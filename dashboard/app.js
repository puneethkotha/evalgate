// EvalGate dashboard — renders report.json into the instrument panel. No framework, no build.
const C = { amber: "#F5A623", pass: "#4FB477", fail: "#E5484D", mid: "#99A1A4",
            lo: "#5F686B", hair: "#272C2F", grid: "#1F2427", hi: "#EAEDED" };
const f3 = (x) => (x === null || x === undefined || Number.isNaN(x)) ? "n/a" : x.toFixed(3);
const el = (id) => document.getElementById(id);

function svgEl(w, h) {
  return `<svg viewBox="0 0 ${w} ${h}" width="100%" preserveAspectRatio="xMidYMid meet" style="display:block">`;
}
function line(x1, y1, x2, y2, stroke, sw = 1, dash = "") {
  return `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${stroke}" stroke-width="${sw}" ${dash ? `stroke-dasharray="${dash}"` : ""}/>`;
}
function text(x, y, s, fill = C.lo, size = 9, anchor = "middle") {
  return `<text x="${x}" y="${y}" fill="${fill}" font-size="${size}" text-anchor="${anchor}">${s}</text>`;
}
function diamond(x, y, r, fill) {
  return `<path d="M ${x} ${y - r} L ${x + r} ${y} L ${x} ${y + r} L ${x - r} ${y} Z" fill="${fill}"/>`;
}

function kappaGauge(kappa, minK) {
  const W = 460, H = 72, pad = 24, y = 40;
  const x = (k) => pad + ((k + 1) / 2) * (W - 2 * pad);
  const ok = kappa >= minK;
  let s = svgEl(W, H);
  s += line(pad, y, W - pad, y, C.grid, 2);
  for (const t of [-1, 0, 0.2, 0.4, 0.6, 0.8, 1]) {
    s += line(x(t), y - 6, x(t), y + 6, C.hair, 1);
    s += text(x(t), y + 20, t, C.lo, 9);
  }
  // Landis–Koch "substantial+" band (0.6..1)
  s += `<rect x="${x(0.6)}" y="${y - 3}" width="${x(1) - x(0.6)}" height="6" fill="${C.amber}" opacity="0.16"/>`;
  s += line(x(minK), y - 16, x(minK), y + 12, C.amber, 1);
  s += text(x(minK), y - 20, `min κ ${minK.toFixed(2)}`, C.amber, 9);
  if (!Number.isNaN(kappa)) {
    s += diamond(x(kappa), y, 6, ok ? C.pass : C.fail);
    s += text(x(kappa), y - 12, kappa.toFixed(2), ok ? C.pass : C.fail, 10);
  }
  return s + "</svg>";
}

function passRateGauge(pr, minPR) {
  const W = 460, H = 60, pad = 30, y = 34;
  const d0 = Math.min(0.8, pr.lower - 0.02, minPR - 0.02), d1 = 1.0;
  const x = (v) => pad + ((v - d0) / (d1 - d0)) * (W - 2 * pad);
  const ok = pr.lower >= minPR;
  let s = svgEl(W, H);
  s += line(pad, y, W - pad, y, C.grid, 2);
  for (const t of [d0, (d0 + d1) / 2, d1]) s += text(x(t), y + 18, t.toFixed(2), C.lo, 9);
  s += line(x(minPR), y - 14, x(minPR), y + 10, C.amber, 1);
  s += text(x(minPR), y - 18, `min ${minPR.toFixed(2)}`, C.amber, 9);
  s += line(x(pr.lower), y, x(pr.upper), y, ok ? C.pass : C.fail, 2);   // whisker
  s += line(x(pr.lower), y - 6, x(pr.lower), y + 6, ok ? C.pass : C.fail, 1);
  s += line(x(pr.upper), y - 6, x(pr.upper), y + 6, ok ? C.pass : C.fail, 1);
  s += diamond(x(pr.point), y, 6, ok ? C.pass : C.fail);
  return s + "</svg>";
}

function confusion(cal) {
  // Reconstruct integer TP/FP/FN/TN from the reported rates + prevalence + n.
  const n = cal.n, P = Math.round(cal.prevalence * n), N = n - P;
  const tp = Math.round(cal.tpr * P), fn = P - tp;
  const tn = Math.round(cal.tnr * N), fp = N - tn;
  return `<table class="cm">
    <tr><th></th><th>human PASS</th><th>human FAIL</th></tr>
    <tr><th>judge PASS</th><td class="diag">${tp}<small>TP</small></td><td class="err">${fp}<small>FP · lenient</small></td></tr>
    <tr><th>judge FAIL</th><td class="err">${fn}<small>FN · strict</small></td><td class="diag">${tn}<small>TN</small></td></tr>
  </table>`;
}

function timeline(history, minK) {
  const W = 1040, H = 230, padL = 40, padR = 20, padT = 20, padB = 34;
  const runs = history.map((h) => h.run);
  const x = (i) => padL + (i / (history.length - 1)) * (W - padL - padR);
  const yd0 = 0.4, yd1 = 1.0;
  const y = (v) => padT + (1 - (v - yd0) / (yd1 - yd0)) * (H - padT - padB);
  let s = svgEl(W, H);
  // y gridlines
  for (const g of [0.4, 0.6, 0.8, 1.0]) {
    s += line(padL, y(g), W - padR, y(g), C.grid, 1);
    s += text(padL - 8, y(g) + 3, g.toFixed(1), C.lo, 9, "end");
  }
  // threshold
  s += line(padL, y(minK), W - padR, y(minK), C.amber, 1, "4 3");
  s += text(W - padR, y(minK) - 5, `min κ ${minK.toFixed(2)}`, C.amber, 9, "end");
  // pass-rate line (grey)
  const prPts = history.map((h, i) => `${x(i)},${y(h.pass_rate)}`).join(" ");
  s += `<polyline points="${prPts}" fill="none" stroke="${C.mid}" stroke-width="1.5" opacity="0.7"/>`;
  // kappa line (green), segments red when either endpoint drifted
  const kPts = history.map((h, i) => `${x(i)},${y(h.kappa)}`).join(" ");
  s += `<polyline points="${kPts}" fill="none" stroke="${C.pass}" stroke-width="2"/>`;
  history.forEach((h, i) => {
    const col = h.drifted ? C.fail : C.pass;
    s += `<circle cx="${x(i)}" cy="${y(h.kappa)}" r="${h.drifted ? 4.5 : 3}" fill="${col}"/>`;
    if (i % 2 === 0 || i === history.length - 1) s += text(x(i), H - 12, `#${h.run}`, C.lo, 9);
    if (h.drifted) s += text(x(i), y(h.kappa) + 18, "drift", C.fail, 9);
  });
  return s + "</svg>";
}

function sparkline(size, max) {
  const cells = 14, filled = Math.max(1, Math.round((size / max) * cells));
  return `<span class="spark">${"█".repeat(filled)}<span style="color:var(--grid)">${"░".repeat(cells - filled)}</span></span>`;
}

function render(data) {
  const g = data.gate, cal = g.calibration, pr = g.pass_rate, d = g.delta;
  const passed = g.passed;
  el("agent").textContent = `live · ${data.agent}`;
  el("lede").textContent = `CI gate · run #${data.run} · ${data.date}`;

  // verdict
  const v = el("verdict"); v.hidden = false; v.classList.toggle("fail", !passed);
  el("partno").textContent = `EVAL–0${data.run} · REV C`;
  el("stamp").textContent = passed ? "PASS" : "FAIL";
  el("exit").textContent = `exit ${passed ? 0 : 1}`;
  const ok = (b) => b ? `<span class="ok">✓</span>` : `<span class="bad">✕</span>`;
  el("ineq").innerHTML =
    `pass-rate <span class="v">${f3(pr.point)}</span> <span style="color:var(--lo)">[${f3(pr.lower)}, ${f3(pr.upper)}]</span> ≥ ${f3(g.min_pass_rate)} ${ok(pr.lower >= g.min_pass_rate)}` +
    `<span class="sep">·</span>judge κ <span class="v">${f3(cal.kappa)}</span> ≥ ${f3(cal.min_kappa)} ${ok(!cal.drifted)}` +
    `<span class="sep">·</span>drift <span class="${cal.drifted ? "bad" : "ok"}">${cal.drifted ? "detected ✕" : "none ✓"}</span>`;
  let meta = `<b>n</b> ${pr.n} traces · <b>judge</b> llama-3.3-70b @ T=0 · ${data.date}`;
  if (d) meta += ` · <b>Δ vs main</b> +${d.c} fixed / -${d.b} regressed (${d.verdict}, McNemar p=${f3(d.p_value)})`;
  el("meta").innerHTML = meta;

  // kappa
  el("kappa-reading").innerHTML = `${f3(cal.kappa)}<span class="unit">κ · ${cal.band}</span>`;
  el("kappa-gauge").innerHTML = kappaGauge(cal.kappa, cal.min_kappa);
  el("kappa-rates").innerHTML =
    `<span>AC1 <b>${f3(cal.ac1)}</b></span><span>raw <b>${f3(cal.raw_agreement)}</b></span>` +
    `<span>prevalence <b>${f3(cal.prevalence)}</b></span>` +
    `<span>paradox <b class="${cal.paradox_flag ? "" : "ok"}">${cal.paradox_flag ? "flag" : "clear"}</b></span>`;

  // pass-rate
  el("pr-reading").innerHTML = `${f3(pr.point)}<span class="unit">[${f3(pr.lower)}, ${f3(pr.upper)}] · ${pr.passed}/${pr.n}</span>`;
  el("pr-gauge").innerHTML = passRateGauge(pr, g.min_pass_rate);
  const corrected = pr.corrected != null ? `<span>bias-corrected <b>${f3(pr.corrected)}</b></span>` : "";
  el("pr-rates").innerHTML = `${corrected}<span>method <b>${pr.method}</b></span><span>n <b>${pr.n}</b></span>`;

  // confusion + rates
  el("cm").innerHTML = confusion(cal);
  el("cm-rates").innerHTML = `<span>TPR <b>${f3(cal.tpr)}</b></span><span>TNR <b>${f3(cal.tnr)}</b></span><span>n <b>${cal.n}</b></span>`;

  // taxonomy
  const tax = data.taxonomy;
  const total = tax.reduce((a, c) => a + c.size, 0);
  el("tax-cap").textContent = `error-analysis-first · ${total} failing traces · KMeans over embeddings`;
  const max = Math.max(...tax.map((c) => c.size), 1);
  el("taxlist").innerHTML = tax.map((c, i) =>
    `<div class="taxrow"><span class="id">C${i + 1}</span>` +
    `<span class="desc">${c.label} ${sparkline(c.size, max)}</span>` +
    `<span class="ct">${c.size}</span></div>`).join("");

  // timeline
  el("timeline").innerHTML = timeline(data.history, data.min_kappa);

  document.title = `EvalGate — ${passed ? "PASS" : "FAIL"}`;
}

fetch("report.json")
  .then((r) => r.json())
  .then(render)
  .catch((e) => { el("lede").textContent = "could not load report.json — run: evalgate report --out dashboard/report.json"; console.error(e); });
