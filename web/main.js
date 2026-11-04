// @ts-check
/* global Plotly */
const IDS = ['alphabet', 'k', 'm', 'length', 'identity', 'proteins', 'rep_share']; // alphabet first: it sets k's range
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
  };
}

function showInputs(p) {
  el('k-out').textContent = String(p.k);
  el('m-out').textContent = String(p.m);
  el('length-out').textContent = String(p.length);
  el('identity-out').textContent = p.identity.toFixed(2);
  el('proteins-out').textContent = p.proteins.toExponential(1).replace('e+', 'e');
  el('rep_share-out').textContent = p.rep_share.toFixed(3);
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

function draw(r, p) {
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
}

el('controls').addEventListener('input', update);
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
