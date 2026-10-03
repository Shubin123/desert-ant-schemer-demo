// Schemer in-browser engine: tokenizer + ONNX graphs + typed decode,
// mirroring the Python reference (predict_for_schema). Runs on
// onnxruntime-web (browser) or onnxruntime-node (conformance tests).

import { Harness } from "./harness.js";
import { fetchBytes } from "./download.js";

const SEP = " ||| ";
const BIO_O = 0, BIO_B = 1, BIO_I = 2;

function argmax(arr) {
  let bi = 0;
  for (let i = 1; i < arr.length; i++) if (arr[i] > arr[bi]) bi = i;
  return bi;
}
const sigmoid = (x) => 1 / (1 + Math.exp(-x));

export class Schemer {
  constructor({ ort, tokenizer, sessions, specs, maxLength = 1216 }) {
    this.ort = ort;
    this.tok = tokenizer;
    this.s = sessions; // {encoder, reader, heads, label, dt}
    this.h = new Harness(specs);
    this.maxLength = maxLength;
    // Schema-side encodings (field queries, label values, dt queries) are
    // independent of the input text and recur on every extract; memoize
    // them. Entries are short strings, so the cache stays small.
    this.queryCache = new Map();
    this._runQ = Promise.resolve();
  }

  // ort sessions are not reentrant ("Session already started" when two
  // run() calls overlap, e.g. prewarm racing the first extract, and the
  // webgpu EP enforces it strictly): funnel every run through one queue.
  runSession(sess, feeds) {
    const r = this._runQ.then(() => sess.run(feeds));
    this._runQ = r.catch(() => {});
    return r;
  }

  static async load({ ort, PreTrainedTokenizer, AutoTokenizer, modelBase, specBase, executionProviders, onProgress, maxLength, lowMemory }) {
    // Download every asset through fetchBytes rather than letting ort and
    // transformers.js fetch internally: that gives byte-level progress for
    // the UI and survives the connection drops that a single 200 MB fetch
    // on a mobile network tends not to. The buffers are then handed to
    // ort/tokenizer directly, so nothing downloads twice.
    const progress = new Map(); // url -> {loaded, total}
    const report = () => {
      if (!onProgress) return;
      let loaded = 0, total = 0;
      for (const p of progress.values()) { loaded += p.loaded; total += p.total; }
      onProgress({ loaded, total });
    };
    // One cheap HEAD per file before any download, so the progress total
    // is exact from the first byte. Summing totals as each GET responds
    // instead spikes the percentage to ~100 while most totals are still
    // unknown, and a display that never goes backwards can't recover.
    const track = async (url) => {
      const p = { loaded: 0, total: 0 };
      progress.set(url, p);
      try {
        const len = Number((await fetch(url, { method: "HEAD" })).headers.get("Content-Length"));
        if (len) { p.total = len; report(); }
      } catch { /* size arrives with the GET instead */ }
    };
    const get = (url) => {
      const p = progress.get(url);
      return fetchBytes(url, {
        onChunk: (delta, fileTotal) => {
          p.loaded += delta;
          p.total = Math.max(p.total, fileTotal, p.loaded);
          report();
        },
      });
    };
    const json = (bytes) => JSON.parse(new TextDecoder().decode(bytes));

    const modelFiles = {
      encoder: "encoder_web4e4.onnx", dt: "dt_head.onnx",
      reader: "reader.onnx", heads: "heads.onnx", label: "label_head.onnx",
    };
    const modelUrls = Object.entries(modelFiles).map(([k, f]) => [k, `${modelBase}/${f}`]);
    // Fetch tokenizer files directly: CDN-hosted transformers.js resolves
    // from_pretrained paths against its own origin, not the page.
    const tokUrls = PreTrainedTokenizer
      ? [`${modelBase}/tokenizer.json`, `${modelBase}/tokenizer_config.json`] : [];
    const specNames = ["relative_dates", "duration_units", "string_trim", "format_gates"];
    const specUrls = specNames.map(n => `${specBase}/${n}.json`);
    await Promise.all([...modelUrls.map(([, u]) => u), ...tokUrls, ...specUrls].map(track));

    const opts = { executionProviders: executionProviders || ["wasm"] };
    const sessions = {};
    let tokenizerFiles = null, specBytes = null;
    const startSmall = () => {
      if (tokUrls.length) tokenizerFiles = Promise.all(tokUrls.map(get));
      specBytes = Promise.all(specUrls.map(get));
    };
    if (lowMemory) {
      // Phones: one model file in memory at a time (download, compile,
      // release, then the next); parallel buffers put the peak past what
      // a phone browser gives a tab before it force-reloads the page.
      // The small files follow after the graphs.
      for (const [k, url] of modelUrls) {
        sessions[k] = await ort.InferenceSession.create(await get(url), opts);
      }
      startSmall();
    } else {
      // Desktop: download everything in parallel and compile each graph
      // the moment its download lands, serialized so the compile working
      // sets never stack. Each buffer is released right after (in proxy
      // mode ort transfers it to the worker rather than copying).
      let compileChain = Promise.resolve();
      const sessionJobs = modelUrls.map(([k, url]) => {
        const bytes = get(url);
        return compileChain = Promise.all([compileChain, bytes]).then(
          async ([, b]) => { sessions[k] = await ort.InferenceSession.create(b, opts); });
      });
      startSmall();
      await Promise.all(sessionJobs);
    }

    // Parse the 34 MB tokenizer JSON only now, after the model buffers
    // are gone; the parse spike would otherwise stack on top of them.
    const tokenizer = tokenizerFiles
      ? await tokenizerFiles.then(([tj, tc]) => new PreTrainedTokenizer(json(tj), json(tc)))
      : await AutoTokenizer.from_pretrained(modelBase);
    const specs = {};
    const specArr = await specBytes;
    specNames.forEach((n, i) => { specs[n] = json(specArr[i]); });
    return new Schemer({ ort, tokenizer, sessions, specs, maxLength });
  }

  async encode(textStr) {
    const enc = this.tok(textStr, { truncation: true, max_length: this.maxLength });
    const ids = Array.from(enc.input_ids.data, Number);
    const T = ids.length;
    const feeds = {
      input_ids: new this.ort.Tensor("int64", BigInt64Array.from(ids.map(BigInt)), [1, T]),
      attention_mask: new this.ort.Tensor("int64", BigInt64Array.from(ids.map(() => 1n)), [1, T]),
    };
    const out = await this.runSession(this.s.encoder, feeds);
    return { ids, hidden: out.hidden }; // hidden: [1, T, 768]
  }

  encodeQuery(str) {
    // Memoized encode for text-independent strings only; joints embed the
    // input text and must not go through this path. Caching the promise
    // also dedupes concurrent requests for the same string.
    let p = this.queryCache.get(str);
    if (!p) {
      p = this.encode(str);
      this.queryCache.set(str, p);
      p.catch(() => this.queryCache.delete(str));
    }
    return p;
  }

  meanPool(hidden, T) {
    const d = 768, data = hidden.data, out = new Float32Array(d);
    for (let t = 0; t < T; t++)
      for (let j = 0; j < d; j++) out[j] += data[t * d + j];
    for (let j = 0; j < d; j++) out[j] /= T;
    return out;
  }

  fieldDesc(name, spec) {
    return spec.describe ? `${name} (${spec.describe})` : `${name} : ${spec.type}`;
  }

  schemaSummary(name, spec) {
    let seg = `${name}:${spec.type}`;
    if (spec.describe) seg += `(${spec.describe})`;
    return seg;
  }

  dtQuery(name, spec) {
    const parts = [`${name}: datetime`];
    if (spec.describe) parts.push("\u2014 " + spec.describe);
    if (spec.nullable) parts.push("or null");
    return parts.join(" ");
  }

  // Encode every schema-derived string ahead of time (e.g. while the user
  // is still typing) so the first extract only pays for the joint passes.
  async prewarm(schema) {
    const jobs = [];
    for (const [field, spec] of Object.entries(schema)) {
      jobs.push(this.encodeQuery(this.fieldDesc(field, spec)));
      if (spec.type === "label")
        for (const v of spec.values || []) jobs.push(this.encodeQuery(String(v)));
      if (spec.type === "datetime")
        jobs.push(this.encodeQuery(this.dtQuery(field, spec)));
    }
    await Promise.all(jobs);
  }

  textStartIn(jointIds, text) {
    // Align by locating the text's own tokens inside the joint encoding.
    // The first text token can merge differently after the SEP (leading
    // space), so anchor on tokens [1..5] of the standalone encoding.
    const t = this.tok(text, { add_special_tokens: false });
    const tids = Array.from(t.input_ids.data, Number);
    const needle = tids.slice(1, Math.min(6, tids.length));
    if (needle.length) {
      for (let i = 1; i + needle.length <= jointIds.length; i++) {
        let ok = true;
        for (let k = 0; k < needle.length; k++)
          if (jointIds[i + k] !== needle[k]) { ok = false; break; }
        if (ok) return i - 1;
      }
    }
    return Math.max(1, jointIds.length - tids.length - 1);
  }

  padQuery(qHidden, qLen) {
    // Reader/dt graphs are exported with a FIXED query size of 32.
    const Q = 32;
    const states = new Float32Array(Q * 768);
    states.set(qHidden.data.slice(0, Math.min(qLen, Q) * 768));
    const mask = new Uint8Array(Q);
    mask.fill(1, 0, Math.min(qLen, Q));
    return { states, mask, Q };
  }

  async runReader(textStates, T, queryStr, textMask) {
    const q = await this.encodeQuery(queryStr);
    const { states, mask, Q } = this.padQuery(q.hidden, q.ids.length);
    // In ort-web proxy mode run() transfers input buffers to the worker,
    // detaching them on this thread; feed copies of any array the caller
    // reuses (textStates and the region mask go into later runs too).
    const feeds = {
      text_states: new this.ort.Tensor("float32", textStates.slice(), [T, 768]),
      query_states: new this.ort.Tensor("float32", states, [Q, 768]),
      text_mask: new this.ort.Tensor("bool", (textMask || new Uint8Array(T).fill(1)).slice(), [T]),
      query_mask: new this.ort.Tensor("bool", mask, [Q]),
    };
    return this.runSession(this.s.reader, feeds);
  }

  bioDecode(tagLogits, T, mask) {
    const spans = [];
    let cur = null;
    for (let i = 0; i < T; i++) {
      if (mask && !mask[i]) { if (cur) { spans.push(cur); cur = null; } continue; }
      const t = argmax([tagLogits[i * 3], tagLogits[i * 3 + 1], tagLogits[i * 3 + 2]]);
      if (t === BIO_B) { if (cur) spans.push(cur); cur = [i, i]; }
      else if (t === BIO_I && cur) cur[1] = i;
      else { if (cur) spans.push(cur); cur = null; }
    }
    if (cur) spans.push(cur);
    return spans;
  }

  spanScore(tagLogits, span) {
    let best = -1e30;
    let sum = 0, n = 0;
    for (let i = span[0]; i <= span[1]; i++) {
      const l = [tagLogits[i * 3], tagLogits[i * 3 + 1], tagLogits[i * 3 + 2]];
      const m = Math.max(...l);
      const p = [Math.exp(l[0] - m), Math.exp(l[1] - m), Math.exp(l[2] - m)];
      const z = p[0] + p[1] + p[2];
      sum += Math.max(p[1], p[2]) / z; n++;
    }
    return n ? sum / n : best;
  }

  decodeSpanText(jointIds, textStart, span) {
    // Trained convention: head position p corresponds to joint token p-1
    // (the Python path encodes this in its offsets arithmetic).
    const ids = jointIds.slice(Math.max(0, span[0] - 1), span[1]);
    if (!ids.length) return "";
    return this.tok.decode(ids, { skip_special_tokens: true }).trim();
  }

  async extract(text, schema, anchorIso, onField) {
    const anchor = anchorIso ? new Date(anchorIso) : new Date();
    const anchorStr = `today=${anchor.getFullYear()}-${String(anchor.getMonth() + 1).padStart(2, "0")}-${String(anchor.getDate()).padStart(2, "0")}`;
    const out = {};
    const hasDtSibling = Object.values(schema).some(s => s.type === "datetime");

    const entries = Object.entries(schema);
    for (const [i, [field, spec]] of entries.entries()) {
      if (onField) { try { onField(field, i, entries.length); } catch { /* UI only */ } }
      const t = spec.type;
      const desc = this.fieldDesc(field, spec);
      const summary = this.schemaSummary(field, spec);
      const prefix = anchorStr + SEP + summary;
      const joint = prefix + SEP + text;
      const enc = await this.encode(joint);
      const T = enc.ids.length;
      const textStates = enc.hidden.data.slice(0, T * 768);
      const textStart = this.textStartIn(enc.ids, text);

      // FULL sequence + region mask, exactly like the Python path (the
      // heads were trained on full joints with masked prefixes; slicing
      // shifts their outputs by one).
      const regionMask = new Uint8Array(T);
      regionMask.fill(1, textStart, T);
      const reader = await this.runReader(textStates, T, desc, regionMask);
      const states = reader.states.data;
      const rep = reader.reader_rep.data;
      const RT = T;

      if (t === "label") {
        const vals = (spec.values || []).map(String);
        if (!vals.length) { out[field] = null; continue; }
        const embs = new Float32Array(vals.length * 768);
        for (let i = 0; i < vals.length; i++) {
          const ve = await this.encodeQuery(vals[i]);
          embs.set(this.meanPool(ve.hidden, ve.ids.length), i * 768);
        }
        const r = await this.runSession(this.s.label, {
          reader_rep: new this.ort.Tensor("float32", rep, [768]),
          value_embs: new this.ort.Tensor("float32", embs, [vals.length, 768]),
        });
        out[field] = vals[argmax(Array.from(r.label_scores.data))];
        continue;
      }

      const heads = await this.runSession(this.s.heads, {
        states: new this.ort.Tensor("float32", states, [RT, 768]),
        reader_rep: new this.ort.Tensor("float32", rep, [768]),
        // copy: regionMask is read by the decoders and fed to dt below
        text_mask: new this.ort.Tensor("bool", regionMask.slice(), [RT]),
      });

      if (t === "boolean") {
        const cls = argmax(Array.from(heads.bool3.data));
        out[field] = cls === 0 ? null : cls === 2;

      } else if (t === "string") {
        if (argmax(Array.from(heads.pres_string.data)) === 0) {
          out[field] = spec.nullable ? null : "";
        } else {
          const bio = heads.bio_string.data;
          const spans = this.bioDecode(bio, RT, regionMask);
          if (!spans.length) out[field] = spec.nullable ? null : "";
          else {
            const best = spans.reduce((a, b) => this.spanScore(bio, a) >= this.spanScore(bio, b) ? a : b);
            let st = this.decodeSpanText(enc.ids, textStart, best);
            st = this.h.trimSpanTail(st, hasDtSibling);
            if (st && !this.h.formatGateOk(field, spec, st)) st = spec.nullable ? null : "";
            out[field] = st;
          }
        }

      } else if (t === "array") {
        if (argmax(Array.from(heads.pres_array.data)) === 0) { out[field] = []; continue; }
        const bio = heads.bio_array.data;
        const items = this.bioDecode(bio, RT, regionMask)
          .map(sp => this.decodeSpanText(enc.ids, textStart, sp))
          .filter(Boolean);
        out[field] = [...new Set(items)];

      } else if (t === "number") {
        const sl = Array.from(heads.span_start.data), el = Array.from(heads.span_end.data);
        const targetUnit = this.h.targetDurUnit(spec);
        const k = targetUnit != null ? Math.min(3, RT) : 1;
        const order = sl.map((v, i) => [v, i]).filter(p => regionMask[p[1]])
          .sort((a, b) => b[0] - a[0]).slice(0, k).map(p => p[1]);
        let parsed = null, unitAttached = false;
        for (let rank = 0; rank < order.length; rank++) {
          const ps = order[rank];
          let be = ps, bv = el[ps];
          for (let j = ps; j < Math.min(ps + 12, RT); j++) if (el[j] > bv) { bv = el[j]; be = j; }
          let spanText = this.decodeSpanText(enc.ids, textStart, [ps, be]);
          const after = be + 1 < RT ? this.decodeSpanText(enc.ids, textStart, [be + 1, Math.min(be + 8, RT - 1)]) : "";
          const contM = after.match(/^(?:[\s  ]?\d{3})+(?!\d)/);
          if (contM && /\d$/.test(spanText)) spanText = spanText + " " + contM[0].trim();
          const cand = this.h.parseNumberLiteral(spanText, spec);
          if (cand == null) continue;
          const conv = this.h.convertDuration(cand, spanText, after, spec);
          if (targetUnit != null && this.h.spanUnit(spanText, after)) { parsed = conv; unitAttached = true; break; }
          if (rank === 0) { parsed = conv; if (targetUnit == null) break; }
        }
        if (targetUnit != null && !unitAttached) {
          const expr = this.h.findDurationExpression(text, spec);
          if (expr != null) parsed = expr;
        }
        out[field] = parsed;

      } else if (t === "datetime") {
        const q = await this.encodeQuery(this.dtQuery(field, spec));
        const pq = this.padQuery(q.hidden, q.ids.length);
        const dtOut = await this.runSession(this.s.dt, {
          text_states: new this.ort.Tensor("float32", textStates.slice(), [RT, 768]),
          query_states: new this.ort.Tensor("float32", pq.states, [pq.Q, 768]),
          text_mask: new this.ort.Tensor("bool", regionMask.slice(), [RT]),
          query_mask: new this.ort.Tensor("bool", pq.mask, [pq.Q]),
        });
        const g = (n) => Array.from(dtOut[`dt_${n}`].data);
        const dt = {
          nullProb: sigmoid(g("null")[0]),
          year: argmax(g("year")), month: argmax(g("month")), day: argmax(g("day")),
          hour: argmax(g("hour")), minute: argmax(g("minute")),
        };
        out[field] = this.h.composeDatetime(dt, anchor, text);
      } else {
        out[field] = null;
      }
    }
    return out;
  }
}



