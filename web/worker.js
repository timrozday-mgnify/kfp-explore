// Runs the model and the analysis in Pyodide, off the main thread. Only the latest model request is
// computed, ahead of the analysis steps, so sliders stay usable while a set is analysed.
const PYODIDE = 'https://cdn.jsdelivr.net/pyodide/v0.27.7/full/';
const SPECTRUM_KS = [4, 5, 6, 7, 8, 9];
importScripts(`${PYODIDE}pyodide.js`);

let py = null;
let busy = false;
let pending = null;
let steps = [];
let job = 0;

function run() {
  if (!py || busy || !(pending || steps.length)) return;
  busy = true;
  const request = pending;
  pending = null;
  const step = request ? null : steps.shift();
  // A macrotask, so a newer request queued meanwhile can take the next turn.
  setTimeout(() => {
    const t0 = performance.now();
    try {
      if (request) {
        const result = JSON.parse(py.globals.get('compute')(JSON.stringify(request)));
        postMessage({ result, ms: performance.now() - t0 });
      } else {
        const out = step.call();
        postMessage({ step: step.name, job: step.job, result: JSON.parse(out), ms: performance.now() - t0 });
      }
    } catch (e) {
      const message = String(e).split('\n').filter(Boolean).pop();
      if (step) { steps = []; postMessage({ error: message, job: step.job }); } else postMessage({ error: message });
    }
    busy = false;
    run();
  });
}

async function analyse({ source, k }) {
  const id = ++job;
  steps = [];
  try {
    postMessage({ reading: true, job: id });
    const buffer = source.url
      ? await (await checked(fetch(source.url))).arrayBuffer()
      : await source.file.arrayBuffer();
    if (id !== job) return;
    const analysis = () => py.pyimport('analyse');
    steps = [
      { name: 'summary', job: id, call: () => analysis().load(py.toPy(new Uint8Array(buffer))) },
      ...SPECTRUM_KS.map((j) => ({ name: 'spectrum', job: id, call: () => analysis().spectrum(j) })),
      { name: 'rarefy', job: id, call: () => analysis().rarefy(k) },
    ];
    run();
  } catch (e) {
    postMessage({ error: String(e), job: id });
  }
}

async function checked(response) {
  const r = await response;
  if (!r.ok) throw new Error(`Could not fetch the sample (${r.status})`);
  return r;
}

onmessage = ({ data }) => {
  if (data.analyse) analyse(data.analyse);
  else { pending = data; run(); }
};

(async () => {
  try {
    const t0 = performance.now();
    const pyodide = await loadPyodide({ indexURL: PYODIDE });
    await pyodide.loadPackage('numpy');
    pyodide.runPython(await (await fetch('model.py')).text());
    pyodide.FS.writeFile('/home/pyodide/analyse.py', await (await fetch('analyse.py')).text());
    for (const name of ['mini-slice', 'slice']) {
      const response = await fetch(`data/tables-${name}.json`);
      if (response.ok) pyodide.globals.get('load_table')(name, await response.text());
    }
    py = pyodide;
    postMessage({ ready: performance.now() - t0 });
    run();
  } catch (e) {
    postMessage({ error: String(e) });
  }
})();
