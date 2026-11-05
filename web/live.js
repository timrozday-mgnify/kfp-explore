// Precision on demand, in its own Pyodide so the sliders stay usable. Results are kept in IndexedDB,
// so each setting is computed once per browser. Only the latest request is run next.
const PYODIDE = 'https://cdn.jsdelivr.net/pyodide/v0.27.7/full/';
const VERSION = 3; // raise when live.py or an export changes what a result means
const KS = Array.from({ length: 31 }, (_, i) => i + 1);
importScripts(`${PYODIDE}pyodide.js`);

let py = null;
let live = null;
let pending = null;
let busy = false;
const loaded = new Set();
const scanned = new Set(); // settings whose stored results are already in memory

function store(mode, work) {
  return new Promise((resolve) => {
    const open = indexedDB.open('kfp-explore-live', 1);
    open.onupgradeneeded = () => open.result.createObjectStore('results');
    open.onerror = () => resolve(null);
    open.onsuccess = () => {
      const request = work(open.result.transaction('results', mode).objectStore('results'));
      request.onsuccess = () => resolve(request.result ?? null);
      request.onerror = () => resolve(null);
    };
  }).catch(() => null);
}
const keyOf = (r, k) => [VERSION, r.index, r.alphabet, k, r.length, r.paired, r.m].join('|');
const requestAt = (r, k) => JSON.stringify({ ...r, k });

async function start() {
  if (py) return;
  py = await loadPyodide({ indexURL: PYODIDE });
  await py.loadPackage('numpy');
  for (const file of ['analyse.py', 'model.py', 'live.py']) {
    py.FS.writeFile(`/home/pyodide/${file}`, await (await fetch(file)).text());
  }
  live = py.pyimport('live');
}

async function handle(r) {
  if (!py) postMessage({ id: r.id, progress: 'Starting' });
  await start();
  const settings = keyOf(r, 0);
  if (!scanned.has(settings)) { // what this browser already holds here, for the plots across k
    scanned.add(settings);
    for (const k of KS) {
      const text = await store('readonly', (s) => s.get(keyOf(r, k)));
      if (text) live.add(requestAt(r, k), text);
    }
  }
  if (!live.have(requestAt(r, r.k))) {
    if (!loaded.has(r.index)) {
      postMessage({ id: r.id, progress: 'Downloading sequences' });
      const response = await fetch(`data/live-${r.index}.npz`);
      if (!response.ok) throw new Error(`No data for ${r.index} (${response.status})`);
      live.load(r.index, py.toPy(new Uint8Array(await response.arrayBuffer())));
      loaded.add(r.index);
    }
    postMessage({ id: r.id, progress: `Computing k ${r.k}, 1 in ${r.m}` });
    const t0 = performance.now();
    const text = live.compute(requestAt(r, r.k));
    await store('readwrite', (s) => s.put(text, keyOf(r, r.k)));
    postMessage({ id: r.id, computed_s: (performance.now() - t0) / 1000 });
  }
  postMessage({ id: r.id, result: JSON.parse(live.panel(requestAt(r, r.k))) });
}

async function run() {
  if (busy || !pending) return;
  busy = true;
  const r = pending;
  pending = null;
  try {
    await handle(r);
  } catch (e) {
    postMessage({ id: r.id, error: String(e).split('\n').filter(Boolean).pop() });
  }
  busy = false;
  run();
}

onmessage = ({ data }) => { pending = data; run(); };
