'use strict';
const assert = require('node:assert/strict');
const N = require('./dist/network.js');
let checks=0, maxError=0;
function close(a,b,tol=1e-9) { assert.ok(Math.abs(a-b)<tol, `${a} != ${b}`); checks++; }
const c=N.compute(N.defaults);
close(c.old.out,.274); close(c.old.loss,.025538);
close(c.next.out,.32124532712); close(c.next.loss,.0159766165382179037472);
const expectedG=[-.0226,-.0339,-.03616,-.05424,-.04068,-.05198];
const expectedW=[.62260,.23390,.13616,.75424,.54068,.85198];
c.grad.forEach((g,i)=>close(g,expectedG[i]));
c.nextW.forEach((w,i)=>close(w,expectedW[i]));
assert.ok(c.next.loss<c.old.loss); checks++;
const eps=1e-6;
// Include inactive hidden and output ReLUs, all away from their kinks.
for(const p of [N.defaults, {...N.defaults, a:-.4,b:.2}, {...N.defaults,w:[.6,.2,.1,.7,-.5,-.8]}, {...N.defaults,w:[-.6,-.2,.1,.7,.5,.8]}]) {
  const result=N.compute(p);
  for(let i=0;i<6;i++) {
    const plus=p.w.slice(), minus=p.w.slice(); plus[i]+=eps;minus[i]-=eps;
    const numerical=(N.forward(p.a,p.b,p.y,plus).loss-N.forward(p.a,p.b,p.y,minus).loss)/(2*eps);
    maxError=Math.max(maxError,Math.abs(numerical-result.grad[i]));
    close(result.grad[i],numerical);
  }
}
close(N.gate(0),0);
const sigmoid=x=>1/(1+Math.exp(-x));
const first=x=>sigmoid(x)*(1-sigmoid(x));
const second=x=>first(x)*(1-2*sigmoid(x));
for(const x of [-3,-1,0,1,3]) {
  close(first(x),(sigmoid(x+eps)-sigmoid(x-eps))/(2*eps));
  // Differentiate the first derivative; also check sigma itself at a larger step.
  close(second(x),(first(x+eps)-first(x-eps))/(2*eps));
  const h=1e-3;
  close(second(x),(sigmoid(x+h)-2*sigmoid(x)+sigmoid(x-h))/(h*h),2e-8);
}
const logSoftmax=x=>{const m=Math.max(...x),s=x.reduce((s,v)=>s+Math.exp(v-m),0);return x.map(v=>(v-m)-Math.log(s));};
for(const x of [[.2,-.4,1.1],[1000,999,1001],[2]]) {
  const ls=logSoftmax(x),p=ls.map(Math.exp);
  for(let i=0;i<x.length;i++) {
    let sum=0;
    for(let k=0;k<x.length;k++) {
      const plus=x.slice(),minus=x.slice();plus[k]+=eps;minus[k]-=eps;
      const analytical=(i===k?1:0)-p[k];sum+=analytical;
      close(analytical,(logSoftmax(plus)[i]-logSoftmax(minus)[i])/(2*eps),5e-8);
    }
    close(sum,0);close(ls[i],logSoftmax(x.map(v=>v+7))[i]);
  }
}
console.log(`通过 ${checks} 项数值检查；网络梯度中心差分最大误差 ${maxError.toExponential(3)}`);
console.log(`输出 ${c.old.out} → ${c.next.out}；损失 ${c.old.loss} → ${c.next.loss}`);
