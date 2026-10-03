// Schemer deterministic harness, JS port. Logic mirrors the Python
// reference (training/v60/score_heldout.py + relative_dates.py); lexicons
// load at runtime from the harness/*.json porting specs.

export class Harness {
  constructor(specs) {
    this.rel = specs.relative_dates;
    this.dur = specs.duration_units;
    this.trim = specs.string_trim;
    this.gates = specs.format_gates;
    this.offsetPhrases = this.rel.offset_phrases
      .slice()
      .sort((a, b) => b[0].length - a[0].length);
    this.weekdays = this.rel.weekdays;
    this.nextWeekdayRe = new RegExp(this.rel.in_n_days_regex ? "" : "", "");
  }

  // ---- numbers ----------------------------------------------------------
  parseNumberLiteral(spanText, spec) {
    if (!spanText) return null;
    let s = String(spanText).trim().toLowerCase();
    let scale = 1;
    const scaleM = s.match(/(\d[\d.,\s]*\d|\d)\s*(k|thousand|tys|mil|mln|million|m|b|bn|billion)\b/);
    const SCALES = { k: 1e3, thousand: 1e3, tys: 1e3, mil: 1e6, mln: 1e6,
                     million: 1e6, m: 1e6, b: 1e9, bn: 1e9, billion: 1e9 };
    if (scaleM) { s = scaleM[1]; scale = SCALES[scaleM[2]] || 1; }
    const m = s.match(/-?\d[\d.,   ]*\d|-?\d/);
    if (!m) return null;
    let tok = m[0].replace(/[   ]/g, "");
    const lastComma = tok.lastIndexOf(","), lastDot = tok.lastIndexOf(".");
    if (lastComma > -1 && lastDot > -1) {
      if (lastComma > lastDot) tok = tok.replace(/\./g, "").replace(",", ".");
      else tok = tok.replace(/,/g, "");
    } else if (lastComma > -1) {
      const frac = tok.length - lastComma - 1;
      tok = (frac === 3 && tok.length > 4) ? tok.replace(/,/g, "") : tok.replace(",", ".");
    } else if (lastDot > -1) {
      const frac = tok.length - lastDot - 1;
      if (frac === 3 && tok.length > 4) tok = tok.replace(/\./g, "");
    }
    let val = parseFloat(tok);
    if (!isFinite(val)) return null;
    val *= scale;
    const lo = spec.min ?? -1e18, hi = spec.max ?? 1e18;
    if (val < lo || val > hi) return null;
    const dec = spec.decimals;
    if (dec === 0) return Math.round(val);
    if (dec != null) return Math.round(val * 10 ** dec) / 10 ** dec;
    return val === Math.trunc(val) ? Math.trunc(val) : Math.round(val * 1e4) / 1e4;
  }

  targetDurUnit(spec) {
    const d = (spec.describe || "").toLowerCase();
    if (/\bminutes?\b|\bmin\b/.test(d)) return 1;
    if (/\bhours?\b/.test(d)) return 60;
    if (/\bdays?\b/.test(d)) return 1440;
    return null;
  }

  spanUnit(spanText, afterSpan) {
    const units = Object.keys(this.dur.units_minutes).sort((a, b) => b.length - a.length);
    const numM = spanText.match(/-?\d[\d.,]*/);
    if (!numM) return null;
    const tail = spanText.slice(numM.index + numM[0].length);
    const adjacent = (str) => {
      const mm = str.match(/^[\s -]{0,2}([a-zA-ZÀ-ɏ一-鿿぀-ヿ]+)/u);
      if (!mm) return null;
      const w = mm[1].toLowerCase().replace(/s$/, "");
      return this.dur.units_minutes[w] != null ? w : null;
    };
    return adjacent(tail) || (!tail.trim() ? adjacent(afterSpan) : null);
  }

  convertDuration(val, spanText, afterSpan, spec) {
    const target = this.targetDurUnit(spec);
    if (target == null) return val;
    const u = this.spanUnit(spanText, afterSpan);
    if (!u) return val;
    const src = this.dur.units_minutes[u];
    const out = (val * src) / target;
    const lo = spec.min ?? -1e18, hi = spec.max ?? 1e18;
    return out >= lo && out <= hi ? (out === Math.trunc(out) ? out : Math.round(out * 100) / 100) : val;
  }

  findDurationExpression(text, spec) {
    const target = this.targetDurUnit(spec);
    if (target == null) return null;
    const qty = this.dur.word_quantities;
    const units = Object.keys(this.dur.units_minutes).sort((a, b) => b.length - a.length).join("|");
    const qwords = Object.keys(qty).sort((a, b) => b.length - a.length).join("|");
    const re = new RegExp(`\\b(half an?|${qwords}|\\d{1,3}(?:[.,]\\d{1,2})?)\\s+(${units})s?\\b`, "giu");
    const vals = new Set();
    let m;
    while ((m = re.exec(text)) !== null) {
      const t = m[1].toLowerCase();
      let q;
      if (t.startsWith("half")) q = 0.5;
      else if (qty[t] != null) q = qty[t];
      else q = parseFloat(t.replace(",", "."));
      if (!isFinite(q)) continue;
      const unit = this.dur.units_minutes[m[2].toLowerCase().replace(/s$/, "")];
      if (unit == null) continue;
      const v = (q * unit) / target;
      const lo = spec.min ?? -1e18, hi = spec.max ?? 1e18;
      if (v >= lo && v <= hi) vals.add(v === Math.trunc(v) ? v : Math.round(v * 100) / 100);
    }
    return vals.size === 1 ? [...vals][0] : null;
  }

  // ---- relative dates ----------------------------------------------------
  hasAbsoluteDate(text) {
    return new RegExp(this.rel.absolute_date_guard_regex, "i").test(text);
  }

  parseTimeOfDay(text) {
    const pats = [
      /\b(\d{1,2}):(\d{2})\b/,
      /\b(\d{1,2})\s*(am|pm)\b/i,
      /\b(\d{1,2})(?::(\d{2}))?\s*uhr\b/i,
      /\b(?:klockan|klokka|kl\.?)\s*(\d{1,2})(?::(\d{2}))?\b/i,
      /\ba las?\s+(\d{1,2})(?::(\d{2}))?\b/i,
      /\balle\s+(\d{1,2})(?::(\d{2}))?\b/i,
      /\bà\s+(\d{1,2})\s*h\s*(\d{2})?\b/i,
      /(\d{1,2})時(?:(\d{1,2})分)?/,
      /(\d{1,2})点/,
      /\b(noon|midnight)\b/i,
    ];
    let best = null;
    for (const p of pats) {
      const m = p.exec(text);
      if (!m) continue;
      let h, mi;
      if (/noon|midnight/i.test(m[1])) { h = m[1].toLowerCase() === "noon" ? 12 : 0; mi = 0; }
      else {
        h = parseInt(m[1], 10);
        const g2 = m[2];
        if (g2 && /^(am|pm)$/i.test(g2)) {
          mi = 0;
          if (g2.toLowerCase() === "pm" && h !== 12) h += 12;
          if (g2.toLowerCase() === "am" && h === 12) h = 0;
        } else mi = g2 && /^\d+$/.test(g2) ? parseInt(g2, 10) : 0;
      }
      if (h < 0 || h > 23 || mi < 0 || mi > 59) continue;
      if (best === null || m.index < best[2]) best = [h, mi, m.index];
    }
    return best;
  }

  resolveRelativeDate(text, anchor) {
    if (this.hasAbsoluteDate(text)) return null;
    const low = text.toLowerCase();
    const cands = [];
    const covered = [];
    for (const [phrase, off] of this.offsetPhrases) {
      let idx = 0;
      while ((idx = low.indexOf(phrase, idx)) !== -1) {
        const j = idx + phrase.length;
        const beforeOk = idx === 0 || !/[a-zÀ-ɏ]/i.test(low[idx - 1]);
        const afterOk = j >= low.length || !/[a-zÀ-ɏ]/i.test(low[j]);
        if (beforeOk && afterOk && !covered.some(([s, e]) => s <= idx && j <= e)) {
          covered.push([idx, j]);
          const d = new Date(anchor);
          d.setDate(d.getDate() + off);
          cands.push([idx, j, d]);
        }
        idx = j;
      }
    }
    const inN = new RegExp(this.rel.in_n_days_regex, "gi");
    let m;
    while ((m = inN.exec(text)) !== null) {
      const tok = m[1].toLowerCase();
      let n = /^\d+$/.test(tok) ? parseInt(tok, 10) : this.rel.word_numbers[tok];
      if (/^(a|an|una|une|un|en|ein|einer|ett|uma|um)$/.test(tok)) n = 1;
      if (!n) continue;
      const weekUnits = /week|woche|semana|semaine|settiman|weken|uger|uker|veckor|tygodni|週間|周/i;
      if (weekUnits.test(m[2])) n *= 7;
      const d = new Date(anchor);
      d.setDate(d.getDate() + n);
      cands.push([m.index, m.index + m[0].length, d]);
    }
    const wdNames = Object.keys(this.weekdays).sort((a, b) => b.length - a.length);
    const nextWords = this.rel.next_words.join("|");
    const reNext = new RegExp(`\\b(${nextWords})\\s*(${wdNames.join("|")})\\b`, "giu");
    while ((m = reNext.exec(text)) !== null) {
      const wd = this.weekdays[m[2].toLowerCase()];
      if (wd == null) continue;
      const d = new Date(anchor);
      const delta = ((wd - ((d.getDay() + 6) % 7) - 1 + 700) % 7) + 1;
      d.setDate(d.getDate() + delta);
      cands.push([m.index, m.index + m[0].length, d]);
    }
    if (cands.length === 0) {
      // bare weekday fallback: single marker, no 'last'-word before it
      const hits = [];
      for (const name of wdNames) {
        const re = new RegExp(`(?<![\\w])${name.replace(/[-]/g, "\\-")}(?![\\w])`, "giu");
        while ((m = re.exec(low)) !== null) {
          const before = text.slice(0, m.index);
          if (/(last|past|previous|letzten?|förra|sidste|forrige|pasado|dernier|scorso|vorige|afgelopen)\s*$/i.test(before)) continue;
          hits.push([m.index, m.index + name.length, this.weekdays[name]]);
        }
      }
      const wds = new Set(hits.map(h => h[2]));
      if (wds.size === 1) {
        const [s, e, wd] = hits[0];
        const d = new Date(anchor);
        const delta = ((wd - ((d.getDay() + 6) % 7) - 1 + 700) % 7) + 1;
        d.setDate(d.getDate() + delta);
        cands.push([s, e, d]);
      }
    }
    const iso = (d) => `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
    const uniq = [...new Set(cands.map(c => iso(c[2])))];
    if (uniq.length === 1) return uniq[0];
    if (uniq.length > 1) {
      const t = this.parseTimeOfDay(text);
      if (t) {
        const tpos = t[2];
        const scored = cands.slice().sort((c1, c2) => {
          const d1 = Math.min(Math.abs(tpos - c1[1]), Math.abs(c1[0] - tpos));
          const d2 = Math.min(Math.abs(tpos - c2[1]), Math.abs(c2[0] - tpos));
          return d1 - d2;
        });
        const nd = Math.min(Math.abs(tpos - scored[0][1]), Math.abs(scored[0][0] - tpos));
        const rd = Math.min(Math.abs(tpos - scored[1][1]), Math.abs(scored[1][0] - tpos));
        if (nd <= 25 && rd - nd >= 10) return iso(scored[0][2]);
      }
    }
    return null;
  }

  // ---- string trims + gates ---------------------------------------------
  trimSpanTail(s, temporalSiblings) {
    const fw = new Set(this.trim.trailing_function_words);
    const particles = this.trim.trailing_particles.join("");
    let temporal = null;
    if (temporalSiblings) {
      temporal = new Set([
        ...this.offsetPhrases.map(p => p[0]),
        ...Object.keys(this.weekdays),
        ...this.rel.next_words.map(w => w.toLowerCase()),
      ]);
    }
    const timeRe = /(\d{1,2}([:.]\d{2})?\s*(am|pm|uhr|h|時|点))$|(kl\.?|klokka|klockan|a las|à|alle|om|um)\s*\d{1,2}([:.]\d{2})?$/i;
    let prev = null;
    while (s && s !== prev) {
      prev = s;
      s = s.replace(/[\s,.;:!?、。·\-–]+$/u, "");
      const parts = s.split(" ");
      const tail = parts[parts.length - 1]?.toLowerCase();
      const tail2 = parts.slice(-2).join(" ").toLowerCase();
      if (parts.length >= 2 && fw.has(tail)) s = parts.slice(0, -1).join(" ");
      else if (temporal && parts.length >= 2 && temporal.has(tail)) s = parts.slice(0, -1).join(" ");
      else if (temporal && parts.length >= 3 && temporal.has(tail2)) s = parts.slice(0, -2).join(" ");
      else if (temporal && timeRe.test(s)) s = s.replace(timeRe, "").trimEnd();
      else if (s && particles.includes(s[s.length - 1])) s = s.slice(0, -1);
    }
    return s;
  }

  formatGateOk(field, spec, value) {
    const hint = `${field} ${spec.describe || ""}`.toLowerCase();
    for (const g of this.gates.gates) {
      if (g.keywords.some(k => hint.includes(k))) {
        const re = new RegExp(g.regex, "i");
        if (!re.test(value.trim())) return false;
        if (g.extra && g.extra.includes("6 digits")) {
          const digits = (value.match(/\d/g) || []).length;
          if (digits < 6) return false;
        }
        return true;
      }
    }
    return true;
  }

  // ---- datetime compose ---------------------------------------------------
  composeDatetime(dt, anchorDate, text) {
    // dt: {null, year, month, day, hour, minute} argmaxes + nullProb
    if (dt.nullProb > 0.5) {
      const rd = this.resolveRelativeDate(text, anchorDate);
      const tod = rd ? this.parseTimeOfDay(text) : null;
      if (rd && tod) return `${rd}T${String(tod[0]).padStart(2, "0")}:${String(tod[1]).padStart(2, "0")}`;
      return null;
    }
    const YEAR_OFFSET_RANGE = 128;
    const year = anchorDate.getFullYear() + (dt.year - YEAR_OFFSET_RANGE);
    const month = dt.month + 1, day = dt.day + 1;
    if (month < 1 || month > 12 || day < 1 || day > 31) return null;
    let composed = `${year}-${String(month).padStart(2, "0")}-${String(day).padStart(2, "0")}T${String(dt.hour).padStart(2, "0")}:${String(dt.minute).padStart(2, "0")}`;
    const rd = this.resolveRelativeDate(text, anchorDate);
    if (rd) composed = `${rd}T${composed.split("T")[1]}`;
    return composed;
  }
}

