// Runs the model in Pyodide, off the main thread. Only the latest request is computed.
const PYODIDE = 'https://cdn.jsdelivr.net/pyodide/v0.27.7/full/';
importScripts(`${PYODIDE}pyodide.js`);

let compute = null;
let busy = false;
let pending = null;

function run() {
  if (!compute || busy || !pending) return;
  busy = true;
  const request = pending;
  pending = null;
  // A macrotask, so a newer request queued meanwhile replaces this one's successor.
  setTimeout(() => {
    try {
      const t0 = performance.now();
      const result = JSON.parse(compute(JSON.stringify(request)));
      postMessage({ result, ms: performance.now() - t0 });
    } catch (e) {
      postMessage({ error: String(e) });
    }
    busy = false;
    run();
  });
}

onmessage = (e) => { pending = e.data; run(); };

(async () => {
  try {
    const t0 = performance.now();
    const py = await loadPyodide({ indexURL: PYODIDE });
    await py.loadPackage('numpy');
    py.runPython(await (await fetch('model.py')).text());
    compute = py.globals.get('compute');
    postMessage({ ready: performance.now() - t0 });
    run();
  } catch (e) {
    postMessage({ error: String(e) });
  }
})();
