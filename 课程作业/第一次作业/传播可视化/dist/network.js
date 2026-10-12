/* Same computation for browser and numerical verification. No dependencies. */
(function (root) {
  'use strict';
  const defaults = {a: .2, b: .3, y: .5, eta: 1, w: [.6, .2, .1, .7, .5, .8]};
  const names = ['A → 1', 'B → 1', 'A → 2', 'B → 2', '1 → 3', '2 → 3'];
  const relu = z => Math.max(0, z);
  const gate = z => z > 0 ? 1 : 0; // Choose the common subgradient 0 at z=0.
  function forward(a, b, y, w) {
    const z1 = a*w[0] + b*w[1], z2 = a*w[2] + b*w[3];
    const h1 = relu(z1), h2 = relu(z2), z3 = h1*w[4] + h2*w[5];
    const out = relu(z3), loss = .5*(out-y)**2;
    return {z1, z2, z3, h1, h2, out, loss};
  }
  function compute({a, b, y, eta, w}) {
    const old = forward(a,b,y,w), d3 = (old.out-y)*gate(old.z3);
    const d1 = d3*w[4]*gate(old.z1), d2 = d3*w[5]*gate(old.z2);
    const grad = [d1*a, d1*b, d2*a, d2*b, d3*old.h1, d3*old.h2];
    const nextW = w.map((v,i) => v-eta*grad[i]);
    return {old, d1, d2, d3, grad, nextW, next: forward(a,b,y,nextW)};
  }
  const api = {defaults, names, relu, gate, forward, compute};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.SignalLab = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
