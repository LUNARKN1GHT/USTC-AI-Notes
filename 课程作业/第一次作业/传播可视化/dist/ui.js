'use strict';
const $=id=>document.getElementById(id), N=SignalLab;
const f=(v,n=5)=>Math.abs(v)<1e-12?'0':Number(v.toFixed(n)).toString();
const titles=['给网络一组输入','隐藏层：求和，再过一道门','输出层：汇合两条信号','损失：预测离目标有多远','反向：把变化率传回去','更新：同时调整六个权重','复算：这一步学得怎么样？'];
let step=0, selected=-1, timer=null, params, result;
document.querySelector('.steps').innerHTML=titles.map((t,i)=>`<button class="step" data-step="${i}">${String(i+1).padStart(2,'0')} ${['输入','隐藏层','输出','损失','反向','更新','复算'][i]}</button>`).join('');
$('weights').innerHTML=N.names.map((n,i)=>`<div class="field"><label for="w${i}">${n}</label><input id="w${i}" type="number" value="${N.defaults.w[i]}" step="0.1" min="-5" max="5"></div>`).join('');
function read() {
  const values=['a','b','y',...N.names.map((_,i)=>'w'+i)].map(id=>$(id).value.trim()===''?NaN:Number($(id).value));
  if(values.some(v=>!Number.isFinite(v)||Math.abs(v)>5)) { $('error').textContent='请输入 −5 到 5 之间的有效数值。';return false; }
  $('error').textContent='';
  params={a:values[0],b:values[1],y:values[2],eta:Number($('eta').value),w:values.slice(3)};
  result=N.compute(params);$('etaValue').textContent=params.eta.toFixed(2);return true;
}
function stop(){clearInterval(timer);timer=null;$('play').textContent='自动演示';}
function setStep(s){step=Math.max(0,Math.min(6,s));selected=-1;render();}
function draw(){
  const c=step===6?result.next:result.old, w=step>=5?result.nextW:params.w;
  const backwards=step===4;
  $('main').classList.toggle('back',backwards);
  const edges=[['M 150 100 L 320 90',225,75],['M 150 230 L 320 90',210,160],['M 150 100 L 320 240',260,175],['M 150 230 L 320 240',225,263],['M 420 90 L 560 165',488,110],['M 420 240 L 560 165',490,244]];
  let svg='<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#7f9a8b"/></marker></defs><title id="graphTitle">'+titles[step]+'</title><desc id="graphDesc">点击连接查看权重的计算。当前步骤 '+(step+1)+'，共七步。</desc>';
  edges.forEach(([path,x,y],i)=>{
    const hot=backwards||step===6||(step===1&&i<4)||(step===2&&i>=4);
    svg+=`<path d="${path}" class="wire ${hot?'hot':''} ${i===selected?'picked':''}" ${backwards?'marker-start':'marker-end'}="url(#arrow)"/><path d="${path}" class="wirehit" data-edge="${i}" tabindex="0" role="button" aria-label="连接 ${N.names[i]}，查看计算"/><text x="${x}" y="${y}" text-anchor="middle" class="edge-label" data-edge="${i}">${backwards?'∇ '+f(result.grad[i]):'w '+f(w[i])}</text>`;
  });
  const nodes=[{x:105,y:100,title:'输入 A',value:f(params.a),detail:'固定输入',lit:step===0},{x:105,y:230,title:'输入 B',value:f(params.b),detail:'固定输入',lit:step===0},{x:370,y:90,title:'节点 1 · ReLU',value:step===4?'δ '+f(result.d1):step>=1?f(c.h1):'?',detail:step>=1?'z = '+f(c.z1):'等待求和',lit:step===1||step===4||step===6},{x:370,y:240,title:'节点 2 · ReLU',value:step===4?'δ '+f(result.d2):step>=1?f(c.h2):'?',detail:step>=1?'z = '+f(c.z2):'等待求和',lit:step===1||step===4||step===6},{x:610,y:165,title:'节点 3 · ReLU',value:step===4?'δ '+f(result.d3):step>=2?f(c.out):'?',detail:step>=2?'z = '+f(c.z3):'等待汇合',lit:step===2||step===4||step===6}];
  nodes.forEach(n=>svg+=`<g class="node ${n.lit?'lit':''}"><circle cx="${n.x}" cy="${n.y}" r="50"/><text class="title" x="${n.x}" y="${n.y-18}" text-anchor="middle">${n.title}</text><text class="value" x="${n.x}" y="${n.y+6}" text-anchor="middle">${n.value}</text><text class="detail" x="${n.x}" y="${n.y+25}" text-anchor="middle">${n.detail}</text></g>`);
  svg+=`<text x="710" y="152" text-anchor="middle" font-size="12" fill="#61746b">目标 y</text><text x="710" y="176" text-anchor="middle" font-size="20" fill="#183932">${f(params.y)}</text><text x="370" y="325" text-anchor="middle" font-size="12" fill="#61746b">${backwards?'← 反向传播 · 误差信号 δ = ∂L/∂z':'前向传播 · 加权求和 → ReLU → 下一层 →'}</text>`;
  $('graph').innerHTML=svg;
}
function explanation(){
  const {old:c,d1,d2,d3,next:n}=result, {a,b,y,eta,w}=params;
  const gateText=(z)=>z>0?'z > 0，门打开，导数为 1':z<0?'z < 0，门关闭，导数为 0':'z = 0，不可导；本演示采用 0';
  if(selected>=0){
    const i=selected, signal=[d1,d1,d2,d2,d3,d3][i],input=[a,b,a,b,c.h1,c.h2][i];
    let formula=i<4?`δ${i<2?'1':'2'} × ${i%2===0?'A':'B'}`:`δ3 × h${i===4?'1':'2'}`;
    $('explain').innerHTML=`<h3>连接 ${N.names[i]} 的责任</h3><p>权重梯度 = 下游误差信号 × 这条连接的输入。所有数值来自更新前的网络。</p><p class="math">∂L/∂w = ${formula} = ${f(signal)} × ${f(input)} = ${f(result.grad[i])}</p><p class="math">w′ = ${f(w[i])} − ${f(eta)} × (${f(result.grad[i])}) = ${f(result.nextW[i])}</p>`;
    return;
  }
  const content=[
    ['从一个样本开始',`A = ${f(a)}，B = ${f(b)}，目标 y = ${f(y)}。圆圈中的 1、2、3 是节点编号。网络要通过调整权重，让预测更接近目标。`,`z = Σ（权重 × 输入），h = max(0, z)`],
    ['两次加权求和，两道 ReLU 门',`节点 1：${gateText(c.z1)}；节点 2：${gateText(c.z2)}。`,`z1 = ${f(w[0])} × ${f(a)} + ${f(w[1])} × ${f(b)} = ${f(c.z1)} → h1 = ${f(c.h1)}<br>z2 = ${f(w[2])} × ${f(a)} + ${f(w[3])} × ${f(b)} = ${f(c.z2)} → h2 = ${f(c.h2)}`],
    ['隐藏层输出成为下一层的输入',`输出节点：${gateText(c.z3)}。网络预测值 ŷ 与目标 y 是两个不同的量。`,`z3 = ${f(w[4])} × ${f(c.h1)} + ${f(w[5])} × ${f(c.h2)} = ${f(c.z3)}<br>ŷ = max(0, z3) = ${f(c.out)}`],
    ['把距离变成可优化的损失',`平方损失对预测的导数是 ŷ − y。当预测小于目标时，这个导数为负，表示适当提高预测可降低损失。`,`L = ½ × (${f(y)} − ${f(c.out)})² = ${f(c.loss,8)}<br>∂L/∂ŷ = ŷ − y = ${f(c.out-y)}`],
    ['链式法则：沿路相乘',`δ 是损失对节点加权输入 z 的导数。反传先乘输出 ReLU 的导数，再乘旧权重和隐藏层 ReLU 的导数。点击一条连接查看权重梯度。`,`δ3 = (ŷ − y) × ReLU′(z3) = ${f(d3)}<br>δ1 = δ3 × ${f(w[4])} × ${N.gate(c.z1)} = ${f(d1)}；δ2 = δ3 × ${f(w[5])} × ${N.gate(c.z2)} = ${f(d2)}`],
    ['同一组旧参数，一次统一更新',`学习率 η = ${f(eta)}。六个梯度都已经算完，现在同时更新六个权重。图上是新权重；节点中保留的是上一次前向传播的值，下一步重新计算。`,`θ′ = θ − η × ∂L/∂θ<br>例如 v1′ = ${f(w[4])} − ${f(eta)} × (${f(result.grad[4])}) = ${f(result.nextW[4])}`],
    ['用新参数检验这一步',`${n.loss<c.loss?'损失降低，预测更接近目标。':n.loss>c.loss?'损失升高。沿负梯度走，步长过大仍可能越过目标。':'损失没有变化。检查学习率是否为 0，或 ReLU 是否截断了梯度。'} 比较结果比仅凭更新方向判断更可靠。`,`ŷ′ = ${f(n.out,10)}<br>L′ = ${f(n.loss,10)}；初始 L = ${f(c.loss,10)}`]
  ][step];
  $('explain').innerHTML=`<h3>${content[0]}</h3><p>${content[1]}</p><p class="math">${content[2]}</p>`;
}
function render(){
  $('sceneTitle').textContent=titles[step];$('phase').textContent=step===4?'BACKWARD / 反向':step===5?'UPDATE / 更新':'FORWARD / 前向';
  document.querySelectorAll('[data-step]').forEach(el=>{const s=Number(el.dataset.step);el.className='step '+(s===step?'active':s<step?'done':'');el.setAttribute('aria-current',s===step?'step':'false');});
  draw();explanation();$('prev').disabled=step===0;$('next').disabled=step===6;$('count').textContent=`${step+1} / 7`;
  const {old,next,grad,nextW}=result,delta=next.loss-old.loss;
  const ratio=old.loss===0?null:next.loss/old.loss;
  $('verdict').textContent=delta< -1e-14?'损失降低':delta>1e-14?'损失升高':'损失不变';
  $('verdict').style.color=delta>1e-14?'var(--orange)':'var(--green)';
  $('change').textContent=ratio===null?'初始损失为 0，不计算百分比':`相比初始${delta>1e-14?'增加':delta< -1e-14?'减少':'变化'} ${f(Math.abs(ratio-1)*100,2)}%`;
  $('lossFill').style.width=(ratio===null?0:Math.min(1,ratio)*100)+'%';$('lossFill').style.background=delta>0?'var(--orange)':'var(--green)';
  $('prediction').textContent=`${f(old.out,5)} → ${f(next.out,5)}`;$('target').textContent=`目标 y = ${f(params.y)}`;
  $('loss').textContent=`${f(old.loss,6)} → ${f(next.loss,6)}`;
  $('tableIntro').textContent=step<4?'先观察前向计算。梯度将在第 5 步揭晓，新权重将在第 6 步揭晓。':'梯度计算使用旧权重和旧激活；更新 θ′ = θ − η∇L。点击任意连接可查看它的计算。';
  $('tbody').innerHTML=N.names.map((n,i)=>`<tr class="${i===selected?'selected':''}"><td>${n}</td><td>${f(params.w[i])}</td><td>${step>=4?f(grad[i]):'待反传'}</td><td>${step>=5?f(-params.eta*grad[i]):'待更新'}</td><td>${step>=5?f(nextW[i]):'待更新'}</td></tr>`).join('');
}
document.querySelector('.steps').addEventListener('click',e=>{const button=e.target.closest('[data-step]');if(button){stop();setStep(Number(button.dataset.step));}});
$('prev').onclick=()=>{stop();setStep(step-1);};$('next').onclick=()=>{stop();setStep(step+1);};
$('play').onclick=()=>{if(timer){stop();return;}if(step===6)setStep(0);$('play').textContent='暂停';timer=setInterval(()=>{if(document.hidden){stop();return;}setStep(step+1);if(step===6)stop();},2600);};
function selectEdge(e){const el=e.target.closest('[data-edge]');if(el){selected=Number(el.dataset.edge);draw();explanation();document.querySelectorAll('#tbody tr').forEach((el,i)=>el.classList.toggle('selected',i===selected));}}
$('graph').addEventListener('click',selectEdge);
$('graph').addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();selectEdge(e);}});
document.querySelectorAll('input').forEach(el=>el.addEventListener('input',()=>{stop();if(read())render();}));
function reset(){stop();['a','b','y','eta'].forEach(id=>$(id).value=N.defaults[id]);N.defaults.w.forEach((v,i)=>$('w'+i).value=v);step=0;selected=-1;read();render();}
$('reset').onclick=reset;
document.querySelectorAll('[data-preset]').forEach(el=>el.onclick=()=>{reset();if(el.dataset.preset==='large')$('eta').value=20;if(el.dataset.preset==='off'){$('w0').value=-.6;$('w1').value=-.2;}read();setStep(el.dataset.preset==='off'?4:6);});
read();render();
