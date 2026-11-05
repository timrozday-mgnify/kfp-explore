// @ts-check
/* global Plotly */
const IDS = ['alphabet', 'k', 'm', 'length', 'identity', 'proteins', 'rep_share', 'index-set', 'floor', 'margin', 'truth', 'mix', 'no_relative', 'coverage', 'targets', 'combo_n', 'combo_lo', 'combo_hi', 'placement', 'draw', 'd', 'snp', 'error', 'b-level', 'b-at', 'pol-family', 'pol-rule', 'pol-truth']; // alphabet first: it sets k's range
const el = (id) => /** @type {HTMLInputElement} */ (document.getElementById(id));

const compact = (x) => (x >= 1e4 || (x > 0 && x < 0.01) ? x.toExponential(1).replace('e+', 'e') : x >= 100 ? x.toFixed(0) : x.toPrecision(2));
const size = (gb) => (gb >= 1000 ? `${(gb / 1000).toPrecision(3)} TB` : gb >= 1 ? `${gb.toPrecision(3)} GB` : `${(gb * 1000).toPrecision(3)} MB`);

function params() {
  return {
    k: +el('k').value,
    m: 2 ** +el('m').value,
    length: +el('length').value,
    identity: +el('identity').value,
    proteins: 10 ** +el('proteins').value,
    rep_share: +el('rep_share').value,
    alphabet: el('alphabet').value,
    paired: el('paired').checked,
    index: el('index-set').value,
    floor: +el('floor').value,
    margin: +el('margin').value,
    truth: +el('truth').value,
    mix: el('mix').value,
    no_relative: +el('no_relative').value,
    coverage: +el('coverage').value,
    combo_n: +el('combo_n').value,
    combo_lo: +el('combo_lo').value,
    combo_hi: +el('combo_hi').value,
    placement: el('placement').value,
    draw: +el('draw').value,
    d: +el('d').value,
    snp: 10 ** +el('snp').value,
    error: 10 ** +el('error').value,
    targets: Math.round(10 ** +el('targets').value),
  };
}

function showInputs(p) {
  el('k-out').textContent = String(p.k);
  el('m-out').textContent = String(p.m);
  el('length-out').textContent = String(p.length);
  el('identity-out').textContent = p.identity.toFixed(2);
  el('proteins-out').textContent = p.proteins.toExponential(1).replace('e+', 'e');
  el('rep_share-out').textContent = p.rep_share.toFixed(3);
  const floor = FLOORS[p.floor];
  el('floor-out').textContent = floor === 0 ? 'none' : `${floor} bits`;
  el('no_relative-out').textContent = p.no_relative.toFixed(2);
  el('targets-out').textContent = p.targets.toExponential(1).replace('e+', 'e');
  el('draw-out').textContent = String(p.draw);
  el('combo_lo-out').textContent = String(p.combo_lo);
  el('combo_hi-out').textContent = String(p.combo_hi);
  el('d-out').textContent = String(p.d);
  el('snp-out').textContent = p.snp.toExponential(1).replace('e+', 'e');
  el('error-out').textContent = p.error.toExponential(1).replace('e+', 'e');
}

const AXIS = { zeroline: false, automargin: true };
const LAYOUT = { margin: { t: 36, r: 12, b: 48, l: 56 }, legend: { orientation: 'h', y: -0.3 } };
const CONFIG = { responsive: true, displaylogo: false };

// Simulated points (measure/sim_grid.py): read pairs of real relatives, at the grid's settings.
const GRID_FILES = { protein: 'sim-grid', murphy10: 'sim-grid-murphy10', dayhoff: 'sim-grid-dayhoff', seb8: 'sim-grid-seb8' };
const grids = {};
const SIMULATED = { name: 'Simulated (mini-slice)', mode: 'markers', marker: { color: '#e08e0b', size: 9, symbol: 'diamond' } };

function gridForIdentity(p) {
  const grid = grids[p.alphabet];
  if (!grid || !p.paired || p.length !== grid.mate) return [];
  const row = grid.rows.find((x) => x.k === p.k && x.m === p.m);
  if (!row) return [];
  const on = row.pairs.map((n, i) => (n > 0 ? i : -1)).filter((i) => i >= 0);
  return [{ ...SIMULATED, x: on.map((i) => grid.mean_identity[i]), y: on.map((i) => row.recall[i]) }];
}

function gridForSparsity(p) {
  const grid = grids[p.alphabet];
  if (!grid || !p.paired || p.length !== grid.mate) return [];
  const b = grid.identity_bins.findIndex((low, i) => p.identity >= low && p.identity < grid.identity_bins[i + 1]);
  const rows = grid.rows.filter((x) => x.k === p.k);
  if (b < 0 || !rows.length || !rows[0].pairs[b]) return [];
  return [{ ...SIMULATED, name: `Simulated (${grid.identity_bins[b]}+)`, x: rows.map((x) => x.m), y: rows.map((x) => x.recall[b]) }];
}

const FLOORS = [0, 5, 10, 20, 30, 40, 60, 80, 120, 160, 200]; // measure/precision_tables.py
const NO_FLOOR_X = 2.5; // where 'no floor' sits on the log axis

// Settings the bundled tables do not cover are computed on demand (live.js).
let liveWorker = null;
let liveId = 0;
let liveAnswered = 0;

function liveProblem(p) {
  if (p.alphabet === 'dna') return 'Precision needs nucleotide relatives, which the bundled sets do not have.';
  if (p.k > p.length) return 'k is longer than the reads.';
  return null;
}

function askLive(p) {
  if (!liveWorker) {
    liveWorker = new Worker('live.js');
    liveWorker.onmessage = ({ data }) => {
      if (data.id !== liveId) return;
      const off = el('precision-unavailable');
      if (data.progress) {
        off.hidden = false;
        off.textContent = `${data.progress}. This takes up to about 30 s the first time; later it is kept in this browser.`;
        el('precision-results').style.opacity = 0.35;
      }
      if (data.error) { off.hidden = false; off.textContent = `Could not compute precision: ${data.error}`; }
      if (data.computed_s !== undefined) el('precision-note-live').textContent = `Computed in this browser in ${data.computed_s.toFixed(1)} s.`;
      if (data.result) { liveAnswered = data.id; drawPrecision(data.result, latest, true); }
    };
  }
  liveId += 1;
  const id = liveId;
  liveWorker.postMessage({ ...p, id });
  setTimeout(() => { // an earlier setting may still be computing, and it cannot be interrupted
    if (id !== liveId || liveAnswered === id) return;
    const off = el('precision-unavailable');
    if (!off.hidden) return;
    off.hidden = false;
    off.textContent = 'Computing…';
    el('precision-results').style.opacity = 0.35;
  }, 300);
}

function drawPrecision(q, p, fromLive = false) {
  const off = el('precision-unavailable');
  if (!q.available) {
    const problem = liveProblem(p);
    el('precision-results').hidden = Boolean(problem);
    off.hidden = !problem;
    if (problem) { off.textContent = problem; el('precision-note-live').textContent = ''; } else askLive(p);
    return;
  }
  off.hidden = true;
  el('precision-results').hidden = false;
  el('precision-results').style.opacity = 1;
  if (!fromLive) el('precision-note-live').textContent = '';
  const value = (x, digits = 3) => (x === null ? '–' : x.toFixed(digits));
  const spread = q.hash_sd === null || q.precision === null ? '' : ` ± ${(q.precision * q.hash_sd).toFixed(3)}`;
  el('p-precision').textContent = `${value(q.precision)}${spread}`;
  el('p-recall').textContent = value(q.recall);
  el('p-auprc').textContent = value(q.auprc);
  el('p-hits').textContent = `${q.key_bits[0].toFixed(1)} (${q.key_bits[1].toFixed(0)}–${q.key_bits[2].toFixed(0)})`;
  el('p-others').textContent = compact(q.others);
  el('p-chance').textContent = q.chance === 0 ? '0' : compact(q.chance);
  el('p-target-precision').textContent = value(q.target_precision);
  el('p-target-recall').textContent = value(q.target_recall);
  el('p-depth').textContent = compact(q.depth);
  el('p-reads').textContent = compact(q.reads);
  el('p-measured').textContent = q.measured_pairs.toLocaleString();
  const label = (x) => (x === 0 ? 'no floor' : `${x} bits`);
  const width = (c) => (c === q.coverage ? 3.5 : 1.25);
  const traces = q.coverages.map((c, i) => ({
    x: q.target_by_floor[i], y: q.curve_precision, text: q.floors.map(label), name: `${c}×`, mode: 'lines+markers', marker: { size: 4 }, line: { width: width(c) },
    hovertemplate: `${c}×, %{text}: recall per target %{x:.3f}, precision %{y:.3f}<extra></extra>`,
  }));
  traces.push({ x: [q.target_recall], y: [q.precision], name: 'This setting', mode: 'markers', marker: { color: '#c0392b', size: 11 } });
  Plotly.react('plot-pr', traces, {
    ...LAYOUT,
    uirevision: 1,
    title: { text: `Precision and recall (${p.index}, k ${p.k}, 1 in ${p.m})`, font: { size: 14 } },
    xaxis: { ...AXIS, title: { text: 'Recall per target' }, range: [0, 1] },
    yaxis: { ...AXIS, title: { text: 'Precision per read' }, range: [0, 1] },
  }, CONFIG);
  const floorX = q.floors.map((x) => (x === 0 ? NO_FLOOR_X : x));
  const axis = {
    ...AXIS, type: 'log', title: { text: 'Information bit floor' }, tickmode: 'array',
    tickvals: [NO_FLOOR_X, 5, 10, 20, 40, 80, 160], ticktext: ['none', '5', '10', '20', '40', '80', '160'],
  };
  const ks = Object.keys(q.by_k).map(Number).sort((a, b) => a - b);
  Plotly.react('plot-floor-precision', ks.map((k) => ({
    x: floorX, y: q.by_k[k].precision, name: `k ${k}`, mode: 'lines+markers', marker: { size: 4 }, line: { width: k === p.k ? 3.5 : 1.25 },
    text: q.floors.map(label), hovertemplate: `k ${k}, %{text}: %{y:.3f}<extra></extra>`,
  })), {
    ...LAYOUT, uirevision: 1, title: { text: `Precision per read by floor (1 in ${p.m})`, font: { size: 14 } }, xaxis: axis, yaxis: { ...AXIS, title: { text: 'Precision per read' }, range: [0, 1] },
  }, CONFIG);
  Plotly.react('plot-floor-recall', q.coverages.map((c, i) => ({
    x: floorX, y: q.target_by_floor[i], name: `${c}×`, mode: 'lines+markers', marker: { size: 4 }, line: { width: width(c) },
    text: q.floors.map(label), hovertemplate: `${c}×, %{text}: %{y:.3f}<extra></extra>`,
  })), {
    ...LAYOUT, uirevision: 1, title: { text: `Recall per target by floor (k ${p.k}, 1 in ${p.m})`, font: { size: 14 } }, xaxis: axis, yaxis: { ...AXIS, title: { text: 'Recall per target' }, range: [0, 1] },
  }, CONFIG);
  Plotly.react('plot-coverage', ks.map((k) => ({
    x: q.coverages, y: q.by_k[k].target, name: `k ${k}`, mode: 'lines+markers', marker: { size: 5 }, line: { width: k === p.k ? 3.5 : 1.25 },
    hovertemplate: `k ${k}, %{x}×: %{y:.3f}<extra></extra>`,
  })), {
    ...LAYOUT, uirevision: 1, title: { text: `Recall per target by coverage (${label(q.floor)}, 1 in ${p.m})`, font: { size: 14 } },
    xaxis: { ...AXIS, type: 'log', title: { text: 'Coverage of the target' }, tickmode: 'array', tickvals: q.coverages, ticktext: q.coverages.map((c) => `${c}×`) },
    yaxis: { ...AXIS, title: { text: 'Recall per target' }, range: [0, 1] },
  }, CONFIG);
  Plotly.react('plot-target-identity', q.coverages.map((c, i) => ({
    x: q.identity_bins, y: q.target_by_identity[i], name: `${c}×`, mode: 'lines+markers', marker: { size: 4 }, line: { width: width(c) },
    hovertemplate: `${c}×, identity from %{x}: %{y:.3f}<extra></extra>`,
  })), {
    ...LAYOUT, uirevision: 1, title: { text: `Recall per target by identity (${label(q.floor)})`, font: { size: 14 } },
    xaxis: { ...AXIS, title: { text: 'Identity to the nearest relative, lower edge' } }, yaxis: { ...AXIS, title: { text: 'Recall per target' }, range: [0, 1] },
  }, CONFIG);
}

function drawSnp(q, p) {
  el('s-keys').textContent = String(q.first.length);
  el('s-fewest').textContent = String(q.hits.length);
  el('s-at-d').textContent = q.at_d.toFixed(3);
  el('s-mean-at-d').textContent = q.mean_at_d.toFixed(3);
  el('s-no-key').textContent = q.no_key.toFixed(3);
  el('s-identity').textContent = q.identity.toFixed(4);
  el('s-uniform').textContent = q.recall_uniform.toFixed(3);
  el('s-chain').textContent = q.recall_chain.toFixed(3);
  const lanes = 6;
  const xs = q.first.flatMap((a, i) => [a, q.last[i] + 1, null]);
  const ys = q.first.flatMap((_, i) => [i % lanes, i % lanes, null]);
  const top = lanes - 0.5;
  const traces = [
    { x: xs, y: ys, name: 'Keys', mode: 'lines', line: { color: '#2c6fbb', width: 7 }, hoverinfo: 'skip' },
    { x: q.hits.map((h) => h + 0.5), y: q.hits.map(() => top), name: 'Fewest substitutions', mode: 'markers', marker: { color: '#c0392b', size: 10, symbol: 'triangle-down' }, hovertemplate: 'position %{x}<extra></extra>' },
  ];
  const shapes = [
    ...q.hits.map((h) => ({ type: 'line', x0: h + 0.5, x1: h + 0.5, y0: -0.5, y1: top, line: { color: '#c0392b', width: 1 } })),
    ...(q.paired ? [{ type: 'line', x0: q.length, x1: q.length, y0: -0.5, y1: top, line: { color: '#666', width: 1, dash: 'dot' } }] : []),
  ];
  Plotly.react('plot-snp-keys', traces, {
    ...LAYOUT, shapes, title: { text: `Keys on the read (k ${p.k}, 1 in ${p.m}${q.paired ? ', dotted line between mates' : ''})`, font: { size: 14 } },
    xaxis: { ...AXIS, title: { text: 'Position on the read' }, range: [0, q.n] }, yaxis: { ...AXIS, showticklabels: false, range: [-0.8, lanes], title: { text: '' } },
  }, CONFIG);
  Plotly.react('plot-snp-curve', [
    { x: q.ds, y: q.curve, name: 'This placement', mode: 'lines', line: { color: '#2c6fbb' } },
    { x: q.ds, y: q.mean, name: 'Any placement', mode: 'lines', line: { color: '#666', dash: 'dash' } },
    { x: [Math.min(q.d, q.ds.length - 1)], y: [q.at_d], name: 'd', mode: 'markers', marker: { color: '#c0392b', size: 10 } },
  ], {
    ...LAYOUT, title: { text: 'Chance d substitutions remove every key', font: { size: 14 } },
    xaxis: { ...AXIS, title: { text: 'Substitutions on the read, d' } }, yaxis: { ...AXIS, title: { text: 'Probability' }, range: [0, 1.02] },
  }, CONFIG);
}

let comboGrid = null;
fetch('data/sim-combo.json').then((r) => (r.ok ? r.json() : null)).then((g) => { comboGrid = g; }).catch(() => {});

function drawCombo(q, p) {
  const off = el('combo-unavailable');
  off.hidden = q.available;
  el('combo-results').hidden = !q.available;
  if (!q.available) {
    off.textContent = q.reason === 'window'
      ? 'No features: the upper bound must reach the lower bound × (k-mers per feature − 1).'
      : 'The reads are too short for these k-mers at the lower bound.';
    return;
  }
  el('c-g').textContent = String(q.g);
  el('c-bits').textContent = q.bits.toFixed(1);
  el('c-recall').textContent = q.combo[q.at].toFixed(3);
  el('c-equal').textContent = q.single_equal[q.at].toFixed(3);
  el('c-postings').textContent = q.postings_combo.toExponential(1).replace('e+', 'e');
  el('c-chance').textContent = `≥ ${compact(q.chance)}`;
  el('c-chance-single').textContent = `≥ ${compact(q.chance_single)}`;
  const equal = q.equal_m >= 1 ? `1 in ${q.equal_m.toFixed(1)}` : '1 in 1';
  const lines = [
    { x: q.identity, y: q.combo, name: `${p.combo_n} × k ${p.k}, 1 in ${p.m}`, mode: 'lines', line: { color: '#1e8449', width: 3 } },
    { x: q.identity, y: q.single, name: `k ${p.k}, 1 in ${p.m}`, mode: 'lines', line: { color: '#888', dash: 'dot' } },
    { x: q.identity, y: q.single_equal, name: `k ${p.k}, ${equal} (same postings)`, mode: 'lines', line: { color: '#2c6fbb' } },
    { x: q.identity, y: q.single_weight, name: `k ${q.weight}, ${equal} (same postings, same letters)`, mode: 'lines', line: { color: '#c0392b' } },
  ];
  const row = p.alphabet === 'protein' && p.length === (comboGrid?.mate) && p.paired
    ? comboGrid?.rows.find((x) => x.k === p.k && x.n === p.combo_n && x.lo === p.combo_lo && x.hi === p.combo_hi) : null;
  const level = Math.log2(p.m);
  if (row && row.recall[level]) {
    const mids = comboGrid.bins.slice(0, -1).map((b, i) => (b + Math.min(comboGrid.bins[i + 1], 1)) / 2);
    lines.push({ x: mids, y: row.recall[level], ...SIMULATED });
  }
  Plotly.react('plot-combo-identity', lines, {
    ...LAYOUT, uirevision: 1, legend: { orientation: 'h', y: -0.28 }, title: { text: `Anchor recall (read of ${p.length} letters)`, font: { size: 14 } },
    xaxis: { ...AXIS, title: { text: 'Identity' } }, yaxis: { ...AXIS, title: { text: 'Anchor recall' }, range: [0, 1] },
  }, CONFIG);
  const perM = q.postings_combo * p.m;
  Plotly.react('plot-combo-size', [
    { x: q.ms.map((m) => perM / m), y: q.by_m_combo, name: `${p.combo_n} × k ${p.k}`, mode: 'lines+markers', marker: { size: 5 }, line: { color: '#1e8449', width: 3 } },
    { x: q.ms.map((m) => perM / q.g / m), y: q.by_m_single, name: `k ${p.k}`, mode: 'lines+markers', marker: { size: 5 }, line: { color: '#2c6fbb' } },
  ], {
    ...LAYOUT, uirevision: 1, title: { text: `Recall against postings, at identity ${p.identity.toFixed(2)}`, font: { size: 14 } },
    xaxis: { ...AXIS, type: 'log', title: { text: 'Postings' } }, yaxis: { ...AXIS, title: { text: 'Anchor recall' }, range: [0, 1] },
  }, CONFIG);
  const scanAxes = {
    xaxis: { ...AXIS, type: 'category', title: { text: `Window's upper bound (lower bound ${p.combo_lo})` } },
    yaxis: { ...AXIS, type: 'category', title: { text: 'Sparsity, 1 in m' } },
  };
  const hover = q.scan_postings.map((row) => row.map((x) => x.toExponential(1).replace('e+', 'e')));
  Plotly.react('plot-combo-scan-recall', [{
    type: 'heatmap', x: q.scan_hi.map(String), y: q.scan_ms.map(String), z: q.scan_recall, zmin: 0, zmax: 1, colorscale: 'Viridis', customdata: hover,
    hovertemplate: 'window to %{x}, 1 in %{y}: recall %{z:.3f}, %{customdata} postings<extra></extra>', colorbar: { title: { text: 'Recall' } },
  }], { ...LAYOUT, uirevision: 1, title: { text: `Anchor recall at identity ${p.identity.toFixed(2)}`, font: { size: 14 } }, ...scanAxes }, CONFIG);
  const logChance = q.scan_chance.map((row) => row.map((x) => (x > 0 ? Math.log10(x) : null)));
  Plotly.react('plot-combo-scan-chance', [{
    type: 'heatmap', x: q.scan_hi.map(String), y: q.scan_ms.map(String), z: logChance, colorscale: 'Viridis', customdata: hover,
    hovertemplate: 'window to %{x}, 1 in %{y}: 10^%{z:.1f} chance per read, %{customdata} postings<extra></extra>', colorbar: { title: { text: 'log10' } },
  }], { ...LAYOUT, uirevision: 1, title: { text: 'Chance anchors per read, log10', font: { size: 14 } }, ...scanAxes }, CONFIG);
}

function draw(r, p) {
  if (r.snp) drawSnp(r.snp, p);
  if (r.combo) drawCombo(r.combo, p);
  el('recall').textContent = r.recall.toFixed(3);
  el('index').textContent = size(r.index_gb);
  el('postings').textContent = r.postings.toExponential(1).replace('e+', 'e');
  el('bits').textContent = `${r.bits_per_kmer.toFixed(1)} (${r.bits_per_letter.toFixed(2)} per letter)`;
  el('chance').textContent = r.chance === 0 ? '0' : r.chance_corrected === null ? `≥ ${compact(r.chance)}` : compact(r.chance_corrected);

  Plotly.react('plot-identity', [
    { x: r.identity, y: r.by_identity, name: 'Model', mode: 'lines', line: { color: '#1f6fb2' } },
    { x: r.identity, y: r.by_identity_independent, name: 'Independent mutations', mode: 'lines', line: { color: '#888', dash: 'dot' } },
    { x: [p.identity], y: [r.recall], name: 'Selected', mode: 'markers', marker: { color: '#c0392b', size: 10 } },
    ...gridForIdentity(p),
  ], {
    ...LAYOUT,
    uirevision: 1,
    title: { text: `Recall by identity (${p.alphabet}, k ${p.k}, 1 in ${p.m}, ${p.length} letters)`, font: { size: 14 } },
    xaxis: { ...AXIS, title: { text: 'Identity' }, range: [0.5, 1] },
    yaxis: { ...AXIS, title: { text: 'Anchor recall' }, range: [0, 1] },
  }, CONFIG);

  const ks = Object.keys(r.by_m).map(Number).sort((a, b) => a - b);
  Plotly.react('plot-m', ks.map((k) => ({
    x: r.ms,
    y: r.by_m[k],
    name: `k ${k}`,
    mode: 'lines',
    line: { width: k === p.k ? 3.5 : 1.25 },
  })).concat([{
    x: [p.m], y: [r.recall], name: 'Selected', mode: 'markers', marker: { color: '#c0392b', size: 10 }, showlegend: false,
  }, ...gridForSparsity(p)]), {
    ...LAYOUT,
    uirevision: 1,
    title: { text: `Recall by sparsity (${p.alphabet}, identity ${p.identity.toFixed(2)}, ${p.length} letters)`, font: { size: 14 } },
    xaxis: { ...AXIS, title: { text: 'Sparsity, 1 in m' }, type: 'log', dtick: Math.log10(4) },
    yaxis: { ...AXIS, title: { text: 'Anchor recall' }, range: [0, 1] },
  }, CONFIG);
}

// The URL holds the controls, so a view can be shared.
const query = new URLSearchParams(location.search);
for (const id of IDS) {
  if (query.has(id)) el(id).value = /** @type {string} */ (query.get(id));
  if (id === 'alphabet') el('k').max = el('alphabet').value === 'dna' ? '51' : '31';
}
if (query.has('paired')) el('paired').checked = query.get('paired') === '1';

const worker = new Worker('worker.js');
const status = el('status');
let latest = params();
let startup = 0;

worker.onmessage = ({ data }) => {
  if (data.job !== undefined) {
    onAnalysis(data);
  } else if (data.error) {
    status.textContent = `Error: ${data.error}`;
  } else if (data.ready) {
    startup = data.ready;
  } else {
    draw(data.result, latest);
    status.textContent = `Computed in ${data.ms.toFixed(0)} ms. Start-up took ${(startup / 1000).toFixed(1)} s.`;
  }
};

function update() {
  el('k').max = el('alphabet').value === 'dna' ? '51' : '31'; // the browser clamps k to this
  latest = params();
  showInputs(latest);
  const q = new URLSearchParams(IDS.map((id) => [id, el(id).value]));
  q.set('paired', el('paired').checked ? '1' : '0');
  history.replaceState(null, '', `?${q}`);
  worker.postMessage(latest);
  const problem = liveProblem(latest);
  if (problem) drawPrecision({ available: false }, latest); else askLive(latest);
}

el('controls').addEventListener('input', update);
el('precision-form').addEventListener('input', update);
el('snp-form').addEventListener('input', update);
el('combo-form').addEventListener('input', update);
update();

for (const [alphabet, file] of Object.entries(GRID_FILES)) {
  fetch(`data/${file}.json`).then((r) => (r.ok ? r.json() : null)).then((g) => {
    if (g) { grids[alphabet] = g; update(); }
  }).catch(() => {});
}

fetch('data/sim-grid.json').then((r) => (r.ok ? r.json() : null)).then((g) => {
  if (!g) return;
  el('grid-note').textContent = `Simulated points: ${g.pairs.toLocaleString()} read pairs from ${g.queries.toLocaleString()} proteins against `
    + `${g.groups.toLocaleString()} groups, without verification; shown for paired reads of ${g.mate} aa, `
    + `k ${[...new Set(g.rows.map((x) => x.k))].join(', ')}, 1 in 1–${Math.max(...g.rows.map((x) => x.m))}. `
    + 'The model keeps kfp\'s recall ceiling of 0.96, which comes from representatives of larger groups. '
    + 'Also simulated, at fewer settings: murphy10, dayhoff and seb8.';
}).catch(() => {});

// Grouping and key policy, measured on keys built kfp's way
const FAMILIES = {
  policy: (c) => c.k === 6 && c.window === 8 && c.radius === 'tree:0.57' && !c.budget,
  radius: (c) => c.k === 6 && c.window === 8 && !c.budget && (c.policy === 'composite:0.25' || c.radius === 'none'),
  window: (c) => c.k === 6 && c.radius === 'tree:0.57' && c.policy === 'composite:0.25' && !c.budget,
  budget: (c) => c.k === 6 && c.radius === 'tree:0.57' && c.budget,
  k: (c) => c.window === 8 && !c.budget && ((c.radius === 'tree:0.57' && c.policy === 'composite:0.25') || c.radius === 'none'),
};
let policy = null;

const configName = (c) => {
  const group = c.radius === 'none' ? 'no groups' : c.radius;
  const keys = c.policy === 'composite:0' ? 'union' : c.policy.replace('composite:', 'share ≥ ');
  return `${group}, ${keys}, window ${c.window}${c.budget ? `, ${c.budget}` : ''}, k ${c.k}`;
};

function drawPolicy() {
  if (!policy) return;
  const family = FAMILIES[el('pol-family').value];
  const f = +el('pol-floor').value;
  const rule = el('pol-rule').value;
  const truth = +el('pol-truth').value;
  const configs = policy.configs.filter(family);
  const nBands = policy.bands.length;
  const bins = policy.identity_bins.slice(0, -1);
  const rows = bins.map((_, b) => b); // the last bin has no relative, so no true group
  const POL_LAYOUT = { ...LAYOUT, margin: { ...LAYOUT.margin, b: 56 }, legend: { orientation: 'h', y: -0.42 } };
  const lineFor = (c) => ({ name: configName(c), mode: 'lines+markers', marker: { size: 5 } });
  const recall = (c, b) => c[rule][f][b * nBands] / policy.pairs_by_bin[b];
  const precision = (c) => {
    let correct = 0;
    let wrong = (c.decoys[f] / policy.pairs) * 2; // protein reads are in frame; real reads meet chance in 2 more
    for (let b = 0; b < policy.pairs_by_bin.length; b += 1) {
      for (let band = 0; band < nBands; band += 1) {
        const value = c[rule][f][b * nBands + band] / policy.pairs;
        if (band <= truth) correct += value; else wrong += value;
      }
    }
    return correct / Math.max(correct + wrong, 1e-12);
  };
  const axisId = { ...AXIS, title: { text: 'Query identity to its nearest relative, lower edge' } };
  Plotly.react('plot-pol-identity', configs.map((c) => ({ ...lineFor(c), x: bins, y: rows.map((b) => recall(c, b)) })), {
    ...POL_LAYOUT, title: { text: 'True group kept, per read pair', font: { size: 14 } }, xaxis: axisId, yaxis: { ...AXIS, title: { text: 'Recall per read pair' }, range: [0, 1] },
  }, CONFIG);
  Plotly.react('plot-pol-target', configs.map((c) => ({
    ...lineFor(c), x: bins, y: rows.map((b) => c.called_true[f][b] / policy.queries_by_bin[b]),
  })), {
    ...POL_LAYOUT, title: { text: `True group called by any of ${policy.pairs_per_query} read pairs`, font: { size: 14 } }, xaxis: axisId, yaxis: { ...AXIS, title: { text: 'Recall per target' }, range: [0, 1] },
  }, CONFIG);
  const low = rows.filter((b) => bins[b] >= 0.7 && bins[b] < 0.9); // the range where policies differ
  Plotly.react('plot-pol-size', configs.map((c) => ({
    name: configName(c), mode: 'markers', marker: { size: 10 }, x: [c.postings],
    y: [low.reduce((s, b) => s + recall(c, b), 0) / low.length], customdata: [precision(c)],
    hovertemplate: `${configName(c)}<br>%{x:.3s} postings, recall %{y:.3f}, precision %{customdata:.3f}<extra></extra>`,
  })), {
    ...LAYOUT, showlegend: false, title: { text: 'Recall at 70–90% against postings', font: { size: 14 } },
    xaxis: { ...AXIS, type: 'log', title: { text: `Postings, ${(policy.leaves / 1e3).toFixed(0)} k leaves` } }, yaxis: { ...AXIS, title: { text: 'Recall per read pair' }, range: [0, 1] },
  }, CONFIG);
  Plotly.react('plot-pol-decoys', configs.map((c) => {
    const keep = policy.floors.map((x, i) => [x, c.decoys[i] / policy.pairs]).filter((x) => x[1] > 0 && x[0] > 0);
    return { ...lineFor(c), x: keep.map((x) => x[0]), y: keep.map((x) => x[1]) };
  }), {
    ...POL_LAYOUT, title: { text: 'Decoy groups over a floor', font: { size: 14 } }, xaxis: { ...AXIS, title: { text: 'Information bit floor' } }, yaxis: { ...AXIS, type: 'log', title: { text: 'Per read pair' } },
  }, CONFIG);
}

el('policy-form').addEventListener('input', () => { drawPolicy(); update(); });
fetch('data/policy-mini-slice.json').then((r) => (r.ok ? r.json() : null)).then((t) => {
  if (!t) return;
  policy = t;
  el('pol-floor').innerHTML = t.floors.map((x, i) => `<option value="${i}"${x === 40 ? ' selected' : ''}>${x === 0 ? 'none' : `${x} bits`}</option>`).join('');
  drawPolicy();
  el('policy-note').textContent = `Keys built with kfp's functions on ${t.proteins.toLocaleString()} mini-slice proteins (${t.leaves.toLocaleString()} leaves), the `
    + `rarest k-mer per window, then a representative or a share of members, then a key budget. "frequency" keeps each group's most shared keys, kfp's untried proposal. ${t.pairs.toLocaleString()} `
    + 'read pairs from held-out proteins, in frame, with no errors, no base index and no verification; a read\'s true group holds its query\'s nearest relative, so relatives within the radius count as correct. '
    + 'Postings are group and k-mer pairs, without Fitch sharing. The mini-slice is denser than the release, so values do not carry over to it.';
}).catch(() => {});

// kfp's real indexes, from the benchmark table
const METHODS = { 'flat tier 1': '#c0392b', 'tree tier 1': '#2c6fbb', 'tree tier 2': '#1e8449' };
const TIER_DASH = { slim: 'dot', standard: 'dash', large: 'solid' };
let benchmark = null;

function drawBenchmark() {
  if (!benchmark) return;
  const level = el('b-level').value;
  const at = el('b-at').value;
  const levelName = el('b-level').selectedOptions[0].textContent;
  const runs = Object.entries(benchmark.runs);
  const recallOf = (run) => {
    if (at === 'whole') return run.whole[level][0];
    const i = Math.ceil(benchmark.runs[at].whole[level][1] / benchmark.precision_step - 1e-9);
    return run.at_precision[level][i] ?? 0; // the next grid value up, so never above the true recall
  };
  const atLabel = at === 'whole' ? 'whole hit list' : `at ${at}'s precision`;
  Plotly.react('plot-b-size', Object.entries(METHODS).map(([method, color]) => {
    const rows = runs.filter(([, r]) => r.method === method);
    return {
      x: rows.map(([, r]) => r.index_gb), y: rows.map(([, r]) => recallOf(r)), text: rows.map(([, r]) => r.tier),
      name: method, mode: 'lines+markers', line: { color }, marker: { size: 9 },
      hovertemplate: `${method}, %{text}: %{x:.2f} GB, recall %{y:.3f}<extra></extra>`,
    };
  }), {
    ...LAYOUT,
    title: { text: `Recall by index size (${atLabel})`, font: { size: 14 } },
    xaxis: { ...AXIS, type: 'log', title: { text: 'Index size, GB (1-in-100 subset)' } },
    yaxis: { ...AXIS, title: { text: `Recall, ${levelName}` }, rangemode: 'tozero' },
  }, CONFIG);
  const line = (name, r) => ({ name, mode: 'lines', line: { color: METHODS[r.method], dash: TIER_DASH[r.tier], width: 1.8 } });
  Plotly.react('plot-b-pr', runs.map(([name, r]) => {
    const keep = r.at_precision[level].map((x, i) => [x, i * benchmark.precision_step]).filter(([x]) => x > 0);
    return { ...line(name, r), x: keep.map((x) => x[0]), y: keep.map((x) => x[1]), hovertemplate: `${name}: recall %{x:.3f}, precision %{y:.3f}<extra></extra>` };
  }), {
    ...LAYOUT,
    title: { text: 'Best recall at each precision', font: { size: 14 } },
    xaxis: { ...AXIS, title: { text: `Recall, ${levelName}` }, rangemode: 'tozero' },
    yaxis: { ...AXIS, title: { text: 'Precision' }, range: [0, 1] },
  }, CONFIG);
  const idLevel = benchmark.runs['flat-slim'].by_identity[level] ? level : '0.8';
  Plotly.react('plot-b-identity', runs.map(([name, r]) => ({
    ...line(name, r), x: r.identity_bins, y: r.by_identity[idLevel], hovertemplate: `${name}: strains at %{x}, recall %{y:.3f}<extra></extra>`,
  })), {
    ...LAYOUT,
    title: { text: `Recall at ≥ ${+idLevel * 100}% match, by strain identity`, font: { size: 14 } },
    xaxis: { ...AXIS, title: { text: "Strain's identity to its source" }, autorange: 'reversed' },
    yaxis: { ...AXIS, title: { text: 'Recall' }, range: [0, 1] },
  }, CONFIG);
  Plotly.react('plot-b-tail', Object.entries(benchmark.tail_per_pair).map(([tier, ys]) => {
    const keep = ys.map((y, i) => [benchmark.tail_bits[i], y]).filter((x) => x[1] > 0);
    return {
      name: tier, mode: 'lines+markers', marker: { size: 4 }, x: keep.map((x) => x[0]), y: keep.map((x) => x[1]),
      line: { color: METHODS['tree tier 1'], dash: TIER_DASH[tier] }, hovertemplate: `${tier}: %{x} bits, %{y:.2g} per read pair<extra></extra>`,
    };
  }), {
    ...LAYOUT,
    title: { text: 'Decoy leaves over a score bar (tree tier 1)', font: { size: 14 } },
    xaxis: { ...AXIS, title: { text: 'Score bar, bits' } },
    yaxis: { ...AXIS, type: 'log', title: { text: 'Decoy leaves per read pair' } },
  }, CONFIG);
}

el('benchmark-form').addEventListener('input', () => { drawBenchmark(); update(); });
fetch('data/kfp-benchmark.json').then((r) => (r.ok ? r.json() : null)).then((b) => {
  if (!b) return;
  benchmark = b;
  drawBenchmark();
  el('benchmark-note').textContent = `Real indexes, not the model: ${b.source}, ${b.strains.toLocaleString()} `
    + `strains at 100–70% identity and ${b.read_pairs.toLocaleString()} read pairs, with no background reads, so `
    + 'precision only counts near relatives. A reported unit is correct if its best member reaches the identity, which favours flat '
    + 'tiers at ≥ 99%. Recall has about ±0.01 of noise. The ranking is by abundance. '
    + `Tree index size fits ${b.size_fit.fixed_gb.toFixed(1)} GB plus ${b.size_fit.bytes_per_posting.toFixed(0)} bytes per posting. `
    + 'Tree decoys are leaves, so a key shared by a node counts once per leaf; decoys above about 80 bits rest on few counts.';
}).catch(() => {});

// Analysis of a protein set
const KFP_CORRECTION = { 5: 1.2, 6: 1.3, 7: 1.5, 8: 1.4, 9: 3.6 }; // model.py's HOT
const STEPS = 8; // summary, 6 spectra, rarefaction
const analysisStatus = el('analysis-status');
let currentJob = 0;
let done = 0;
let timings = [];
let spectra = [];

function plot(id, traces, title, xaxis, yaxis) {
  Plotly.react(id, traces, {
    ...LAYOUT,
    title: { text: title, font: { size: 14 } },
    xaxis: { ...AXIS, ...xaxis },
    yaxis: { ...AXIS, ...yaxis },
  }, CONFIG);
}

const MEASURED = { mode: 'lines+markers', line: { color: '#1f6fb2' } };
const INDEPENDENT = { mode: 'lines', line: { color: '#888', dash: 'dot' } };

function drawSummary(r) {
  const q = r.length_quantiles;
  const sampled = r.sampled_from ? ` Sampled at random from ${r.sampled_from.toLocaleString()} sequences.` : '';
  el('analysis-summary').textContent = `${r.sequences.toLocaleString()} sequences, ${r.residues.toLocaleString()} residues`
    + ` (${r.other_residues.toLocaleString()} other letters). Length 5–50–95%: ${q[0]} / ${q[2]} / ${q[4]} aa,`
    + ` E[L²]/E[L]² ${r.length_dispersion.toFixed(2)}. Σf² ${r.s2.toFixed(4)} (model ${0.0597}),`
    + ` per protein 5–50–95%: ${r.protein_s2.map((x) => x.toFixed(3)).join(' / ')}.${sampled}`;
  el('analysis-plots').hidden = false;
  plot('plot-lengths', [{ x: r.length_x, y: r.length_y, type: 'bar', marker: { color: '#1f6fb2' } }],
    'Lengths', { title: { text: 'Length, aa' }, type: 'log' }, { title: { text: 'Sequences' } });
  const amino = [...r.amino];
  plot('plot-composition', [
    { x: amino, y: r.composition, name: 'This set', type: 'bar', marker: { color: '#1f6fb2' } },
    { x: amino, y: r.background, name: 'Model background', mode: 'markers', marker: { color: '#c0392b' } },
  ], 'Composition', {}, { title: { text: 'Share of residues' } });
}

function drawSpectra() {
  const rows = spectra.slice().sort((a, b) => a.k - b.k);
  const ks = rows.map((r) => r.k);
  const table = /** @type {HTMLTableElement} */ (/** @type {unknown} */ (el('spectrum-table')));
  table.hidden = false;
  table.textContent = '';
  const add = (cells, tag) => {
    const row = table.insertRow();
    for (const c of cells) { const cell = document.createElement(tag); cell.textContent = c; row.append(cell); }
  };
  add(['k', 'Windows', 'Distinct', 'Distinct / windows', 'Independent', 'Top 1% postings', 'Decoys ÷ composition', 'Top 1% of hits', 'kfp correction', 'Between halves ÷ composition'], 'th');
  for (const r of rows) {
    add([r.k, r.windows.toLocaleString(), r.distinct.toLocaleString(), r.distinct_share.toFixed(3), r.distinct_independent.toFixed(3), r.top1_postings.toFixed(2),
      r.decoy_ratio.toFixed(2), r.decoy_top1.toFixed(2), KFP_CORRECTION[r.k] ?? '–', r.between_ratio >= 100 ? r.between_ratio.toFixed(0) : r.between_ratio.toFixed(2)], 'td');
  }
  plot('plot-distinct', [
    { x: ks, y: rows.map((r) => r.distinct_share), name: 'This set', ...MEASURED },
    { x: ks, y: rows.map((r) => r.distinct_independent), name: 'Independent', ...INDEPENDENT },
  ], 'Distinct k-mers / windows', { title: { text: 'k' }, dtick: 1 }, { title: { text: 'Share' }, range: [0, 1] });
  plot('plot-collisions', [
    { x: ks, y: rows.map((r) => r.decoy_ratio), name: 'Decoys', ...MEASURED },
    { x: ks, y: rows.map((r) => r.between_ratio), name: 'Between halves', mode: 'lines+markers', line: { color: '#c0392b' } },
    { x: ks.filter((k) => KFP_CORRECTION[k]), y: ks.filter((k) => KFP_CORRECTION[k]).map((k) => KFP_CORRECTION[k]), name: 'kfp correction', mode: 'markers', marker: { color: '#2e8b57', size: 9 } },
    { x: ks, y: ks.map(() => 1), name: 'Composition', ...INDEPENDENT },
  ], 'Matches ÷ composition', { title: { text: 'k' }, dtick: 1 }, { title: { text: 'Ratio' }, type: 'log' });
}

function drawRarefaction(r) {
  const n = r.rows.map((x) => x.sequences);
  const axis = { title: { text: 'Sequences' }, type: 'log' };
  plot('plot-rarefy', [
    { x: n, y: r.rows.map((x) => x.distinct), name: 'This set', ...MEASURED },
    { x: n, y: r.rows.map((x) => x.distinct_independent), name: 'Independent', ...INDEPENDENT },
  ], `Distinct ${r.k}-mers by sequences`, axis, { title: { text: 'Distinct k-mers' }, type: 'log' });
  plot('plot-decoys', [
    { x: r.rows.map((x) => x.decoy_windows), y: r.rows.map((x) => x.decoy_hit), name: 'This set', ...MEASURED },
    { x: r.rows.map((x) => x.decoy_windows), y: r.rows.map((x) => x.decoy_hit_independent), name: 'Independent', ...INDEPENDENT },
  ], `Decoy hits per window, k ${r.k}`, { title: { text: 'Windows in the index' }, type: 'log' }, { title: { text: 'Share of decoy windows hit' }, type: 'log' });
}

function onAnalysis(data) {
  if (data.job !== currentJob) return;
  if (data.error) {
    analysisStatus.textContent = `Error: ${data.error}`;
  } else if (data.reading) {
    analysisStatus.textContent = 'Reading…';
  } else {
    done += 1;
    timings.push(`${data.step === 'spectrum' ? `k ${data.result.k}` : data.step} ${(data.ms / 1000).toFixed(1)} s`);
    if (data.step === 'summary') drawSummary(data.result);
    if (data.step === 'spectrum') { spectra.push(data.result); drawSpectra(); }
    if (data.step === 'rarefy') drawRarefaction(data.result);
    analysisStatus.textContent = `${done === STEPS ? 'Done' : `Step ${done} of ${STEPS}`}: ${timings.join(', ')}`;
  }
}

el('sample').addEventListener('change', () => { el('file-label').hidden = el('sample').value !== 'file'; });

el('analysis-controls').addEventListener('submit', (e) => {
  e.preventDefault();
  const k = +el('rarefy-k').value;
  if (!Number.isInteger(k) || k < 1 || k > 13) { analysisStatus.textContent = 'Rarefaction k must be a whole number from 1 to 13.'; return; }
  const file = el('file').files?.[0];
  if (el('sample').value === 'file' && !file) { analysisStatus.textContent = 'Choose a FASTA file first.'; return; }
  currentJob += 1;
  done = 0;
  timings = [];
  spectra = [];
  analysisStatus.textContent = 'Waiting for Pyodide…';
  worker.postMessage({ analyse: { k, source: el('sample').value === 'file' ? { file } : { url: el('sample').value } } });
});
