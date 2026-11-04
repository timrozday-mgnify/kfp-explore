// Feasibility check for the plan (docs/plan.md): the two-rate anchor model against Monte Carlo
// and against kfp's measured tables (kmer-functional-profiler docs/tree-index-plan.md).
// Run: node checks/model_check.mjs
import assert from 'node:assert';

const BG = [.0825, .0553, .0406, .0545, .0137, .0393, .0675, .0707, .0227, .0596, .0966, .0584,
  .0242, .0386, .0470, .0656, .0534, .0108, .0292, .0687]; // amino-acid background
const S2 = BG.reduce((a, f) => a + f * f, 0); // chance two residues agree

// Identity of the conserved and variable classes: exp(-r t), t set by the mean identity p.
function classes(p, phi, rho) {
  let lo = 0, hi = 200;
  for (let i = 0; i < 60; i++) {
    const t = (lo + hi) / 2;
    phi * Math.exp(-t) + (1 - phi) * Math.exp(-rho * t) > p ? lo = t : hi = t;
  }
  return [Math.exp(-lo), Math.exp(-rho * lo)];
}

// P(no sampled k-mer match in L residues). Chain state: (site class, match run capped at k).
function pMiss(L, k, m, p, phi, rho, ell) {
  const id = classes(p, phi, rho), keep = 1 - 1 / m;
  const toV = 1 / ell, toC = toV * phi / (1 - phi); // stationary conserved share phi
  let s = [new Float64Array(k + 1), new Float64Array(k + 1)];
  s[0][0] = phi; s[1][0] = 1 - phi;
  for (let i = 0; i < L; i++) {
    const n = [new Float64Array(k + 1), new Float64Array(k + 1)];
    for (let c = 0; c < 2; c++) for (let r = 0; r <= k; r++) {
      const v = s[c][r];
      if (!v) continue;
      for (let d = 0; d < 2; d++) {
        const tr = c === 0 ? (d === 1 ? toV : 1 - toV) : (d === 0 ? toC : 1 - toC);
        n[d][0] += v * tr * (1 - id[d]);
        const nr = Math.min(r + 1, k);
        n[d][nr] += v * tr * id[d] * (nr === k ? keep : 1);
      }
    }
    s = n;
  }
  return s[0].reduce((x, y) => x + y, 0) + s[1].reduce((x, y) => x + y, 0);
}

function mcMiss(L, k, m, p, phi, rho, ell, n = 200000) {
  const id = classes(p, phi, rho), toV = 1 / ell, toC = toV * phi / (1 - phi);
  let miss = 0;
  for (let t = 0; t < n; t++) {
    let c = Math.random() < phi ? 0 : 1, run = 0, hit = false;
    for (let i = 0; i < L && !hit; i++) {
      if (Math.random() < (c === 0 ? toV : toC)) c = 1 - c;
      run = Math.random() < id[c] ? run + 1 : 0;
      if (run >= k && Math.random() < 1 / m) hit = true;
    }
    if (!hit) miss++;
  }
  return miss / n;
}

// 90%-AAI table: k, m, release postings, chance anchors per pair, pair recall at P.
const T = [[6, 8, 3e10, 4.6e5, [.96, .96, .97, .93]], [7, 8, 3e10, 3.3e4, [.95, .95, .96, .90]],
  [8, 8, 3e10, 1870, [.95, .96, .94, .87]], [9, 8, 3e10, 274, [.94, .95, .93, .82]],
  [8, 16, 1.5e10, 773, [.91, .90, .84, .72]], [9, 16, 1.5e10, 129, [.88, .86, .80, .65]]];
const P = [.95, .9, .85, .8];
const IID = [.5, 1, 10];
const pair = (k, m, p, th, c) => c * (1 - pMiss(50, k, m, p, ...th) ** 2); // two independent mates
const rmse = (rows, th, c) => {
  let e = 0, n = 0;
  for (const [k, m, , , o] of rows) P.forEach((p, i) => { e += (pair(k, m, p, th, c) - o[i]) ** 2; n++; });
  return Math.sqrt(e / n);
};

console.log('1. Chain against Monte Carlo, P(miss)');
for (const a of [[50, 9, 8, .9, ...IID], [50, 9, 16, .8, .8, 32, 10], [100, 5, 8, .7, .8, 32, 10]]) {
  const d = pMiss(...a), s = mcMiss(...a);
  console.log(` ${a.join(' ')}: chain ${d.toFixed(4)} mc ${s.toFixed(4)}`);
  assert(Math.abs(d - s) < 0.005);
}

console.log('2. Pair recall: fit conservation on m 8, predict m 16');
const train = T.filter(r => r[1] === 8), test = T.filter(r => r[1] === 16);
let best = { e: 9 };
for (const phi of [.2, .3, .4, .5, .6, .7, .8, .9]) for (const rho of [2, 4, 8, 16, 32, 64])
  for (const ell of [5, 10, 20, 40, 80]) for (const c of [.95, .96, .97, .98, 1]) {
    const e = rmse(train, [phi, rho, ell], c);
    if (e < best.e) best = { e, th: [phi, rho, ell], c };
  }
console.log(` iid rmse ${rmse(T, IID, 1).toFixed(3)}; fitted phi, rho, ell ${best.th}, ceiling ${best.c}`);
console.log(` train ${best.e.toFixed(3)}, held-out m 16 ${rmse(test, best.th, best.c).toFixed(3)}`);
for (const [k, m, , , o] of T)
  console.log(` k${k} m${m} model ${P.map(p => pair(k, m, p, best.th, best.c).toFixed(2)).join(' ')} measured ${o.join(' ')}`);
assert(rmse(test, best.th, best.c) < 0.04 && rmse(T, IID, 1) > 0.06);

console.log('3. Chance anchors per pair at release: 2(50-k+1) x postings x S2^k x 3 frames');
for (const [k, m, post, obs] of T) {
  const v = 2 * (50 - k + 1) * post * S2 ** k * 3;
  console.log(` k${k} m${m} model ${v.toExponential(1)} measured ${obs.toExponential(1)} x${(obs / v).toFixed(1)}`);
  assert(obs / v > 1 && obs / v < 5); // composition is a floor, a hot tail adds to it
}

console.log('4. Index size: representative residues / m');
for (const [k, m, post] of T.slice(3, 5)) console.log(` k${k} m${m} model ${(2e11 / m).toExponential(1)} table ${post.toExponential(1)}`);

console.log('5. Noise floor tail, wide k 5, independent hits against decoys: this one FAILS');
const lambda = 2 * 45 * 3 * 14 * S2 ** 5; // read pair windows x frames x keys on a group x q
const hitsAt = bar => { let h = 1; while (20.35 * h - 8.9 * (h - 1) < bar) h++; return h; };
const fact = h => (h < 2 ? 1 : h * fact(h - 1));
const decoys = { 30: 23364, 40: 1947, 50: 192 };
const pois = b => lambda ** hitsAt(b) / fact(hitsAt(b));
for (const b of [40, 50])
  console.log(` bar ${b - 10} to ${b}: poisson falls x${(pois(b - 10) / pois(b)).toFixed(0)}, decoys x${(decoys[b - 10] / decoys[b]).toFixed(0)}`);

const t0 = performance.now();
let n = 0, sink = 0;
for (const k of [4, 5, 6, 7, 8, 9, 10, 11]) for (const m of [1, 2, 4, 8, 16, 32, 64, 128])
  for (let p = .5; p < 1; p += .025) for (const L of [30, 50, 75, 100, 150, 250]) { sink += pMiss(L, k, m, p, ...best.th); n++; }
console.log(`6. ${n} chains in ${(performance.now() - t0).toFixed(0)} ms (JS, one thread)`);
assert(sink > 0);
