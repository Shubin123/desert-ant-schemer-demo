// Download helper for the multi-hundred-MB model files. A plain fetch (or
// ort's internal one) gives no progress and dies on the first network
// hiccup, which on slow mobile connections regularly kills the download
// before it finishes. This reads the body manually so callers get
// byte-level progress, aborts only when the stream genuinely stalls (the
// watchdog resets on every chunk, so a slow-but-moving download never
// times out), and resumes interrupted transfers with a Range request.
//
// Bytes go into a single buffer preallocated from Content-Length rather
// than a chunk list joined at the end: the join briefly doubles the
// file's memory, and on phones the model load already runs close to the
// tab's memory ceiling. The chunk list remains as the fallback for
// responses with no declared length.
export async function fetchBytes(url, { onChunk, stallMs = 90_000, retries = 4 } = {}) {
  let buf = null, chunks = [], loaded = 0, total = 0, canResume = true;
  const reset = () => {
    if (loaded && onChunk) onChunk(-loaded, total);
    chunks = []; loaded = 0; // buf stays; it is simply overwritten
  };
  for (let attempt = 0; ; attempt++) {
    const ctrl = new AbortController();
    let timer, stalled = null;
    const arm = () => {
      clearTimeout(timer);
      timer = setTimeout(() => {
        // Kept aside as well: not every fetch propagates the abort reason.
        stalled = new Error(`download of ${url} stalled for ${stallMs / 1000}s`);
        ctrl.abort(stalled);
      }, stallMs);
    };
    try {
      arm();
      const res = await fetch(url, {
        signal: ctrl.signal,
        headers: loaded ? { Range: `bytes=${loaded}-` } : {},
      });
      if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`);
      if (loaded && res.status !== 206) reset(); // server ignored the Range
      // Range offsets address bytes on the wire; if the response is
      // compressed our decoded byte count doesn't map to a wire offset,
      // so a retry must start over instead of resuming.
      canResume = !res.headers.get("Content-Encoding");
      const len = Number(res.headers.get("Content-Length"));
      if (len) total = loaded + len;
      if (total && !buf) {
        buf = new Uint8Array(total);
        let off = 0;
        for (const c of chunks) { buf.set(c, off); off += c.byteLength; }
        chunks = [];
      }
      const reader = res.body.getReader();
      for (;;) {
        arm();
        const { done, value } = await reader.read();
        if (done) break;
        if (buf) {
          if (loaded + value.byteLength > buf.length) { // more than declared
            const grown = new Uint8Array(loaded + value.byteLength);
            grown.set(buf.subarray(0, loaded));
            buf = grown;
          }
          buf.set(value, loaded);
        } else chunks.push(value);
        loaded += value.byteLength;
        if (onChunk) onChunk(value.byteLength, total);
      }
      clearTimeout(timer);
      // A stream can also end cleanly before all declared bytes arrived
      // (e.g. a proxy cutting the response short): treat it as a drop.
      if (total && loaded < total) throw new Error(`short read: ${loaded}/${total} bytes of ${url}`);
      if (buf) return loaded === buf.length ? buf : buf.subarray(0, loaded);
      const out = new Uint8Array(loaded);
      let off = 0;
      for (const c of chunks) { out.set(c, off); off += c.byteLength; }
      return out;
    } catch (err) {
      clearTimeout(timer);
      if (attempt >= retries) throw stalled || err;
      if (!canResume) reset();
    }
  }
}

