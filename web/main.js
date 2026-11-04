// @ts-check
/* global Plotly */
const IDS = ['k', 'm', 'length', 'identity', 'proteins', 'rep_share'];
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
const LAYOUT = { margin: { t: 36, r: 12, b: 48, l: 56 }, legend: { orientation: 'h', y: -0.2 } };
const CONFIG = { responsive: true, displaylogo: false };

function draw(r, p) {
  el('recall').textContent = r.recall.toFixed(3);
  el('index').textContent = size(r.index_gb);
  el('postings').textContent = r.postings.toExponential(1).replace('e+', 'e');
  el('chance').textContent = r.chance === 0 ? '0' : r.chance_corrected === null ? `≥ ${compact(r.chance)}` : compact(r.chance_corrected);

  Plotly.react('plot-identity', [
    { x: r.identity, y: r.by_identity, name: 'Conserved blocks (fitted)', mode: 'lines', line: { color: '#1f6fb2' } },
    { x: r.identity, y: r.by_identity_independent, name: 'Independent mutations', mode: 'lines', line: { color: '#888', dash: 'dot' } },
    { x: [p.identity], y: [r.recall], name: 'Selected', mode: 'markers', marker: { color: '#c0392b', size: 10 } },
  ], {
    ...LAYOUT,
    uirevision: 1,
    title: { text: `Recall by identity (k ${p.k}, 1 in ${p.m}, ${p.length} aa)`, font: { size: 14 } },
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
  }]), {
    ...LAYOUT,
    uirevision: 1,
    title: { text: `Recall by sparsity (identity ${p.identity.toFixed(2)}, ${p.length} aa)`, font: { size: 14 } },
    xaxis: { ...AXIS, title: { text: 'Sparsity, 1 in m' }, type: 'log', dtick: Math.log10(2) },
    yaxis: { ...AXIS, title: { text: 'Anchor recall' }, range: [0, 1] },
  }, CONFIG);
}

// The URL holds the controls, so a view can be shared.
const query = new URLSearchParams(location.search);
for (const id of IDS) if (query.has(id)) el(id).value = /** @type {string} */ (query.get(id));
if (query.has('paired')) el('paired').checked = query.get('paired') === '1';

const worker = new Worker('worker.js');
const status = el('status');
let latest = params();
let startup = 0;

worker.onmessage = ({ data }) => {
  if (data.error) {
    status.textContent = `Error: ${data.error}`;
  } else if (data.ready) {
    startup = data.ready;
  } else {
    draw(data.result, latest);
    status.textContent = `Computed in ${data.ms.toFixed(0)} ms. Start-up took ${(startup / 1000).toFixed(1)} s.`;
  }
};

function update() {
  latest = params();
  showInputs(latest);
  const q = new URLSearchParams(IDS.map((id) => [id, el(id).value]));
  q.set('paired', el('paired').checked ? '1' : '0');
  history.replaceState(null, '', `?${q}`);
  worker.postMessage(latest);
}

el('controls').addEventListener('input', update);
update();
