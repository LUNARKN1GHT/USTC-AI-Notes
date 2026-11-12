"""数学核验：python 验证参考答案.py；完整执行：追加 --execute；第三种子：--repeat。"""
from __future__ import annotations
import argparse
import ast
import copy
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
for key,name in [("MPLCONFIGDIR","lab7-mpl"),("IPYTHONDIR","lab7-ipython"),("XDG_CACHE_HOME","lab7-cache")]:
    os.environ.setdefault(key,str(Path(tempfile.gettempdir())/name))
os.environ.setdefault("MPLBACKEND","Agg")
import nbformat
import torch
from torch.nn import functional as F
import lab_utils as u


def definitions():
    import 制作notebook as maker
    ns={"__name__":"reference"}
    for reference,name in [(False,"实验七_学生版.ipynb"),(True,"实验七_参考答案.ipynb")]:
        nb=nbformat.read(ROOT/name,as_version=4)
        nbformat.validate(nb)
        fresh=maker.build(reference)
        sources=lambda n:[c.source for c in n.cells if "reference-run-notes" not in c.metadata.get("tags",[])]
        assert sources(nb)==sources(fresh)
        for cell in nb.cells:
            assert not any(ord(c)<32 and c not in "\n\r\t" for c in cell.source)
            if cell.cell_type=="code":
                ast.parse(cell.source)
                assert "GAT" not in cell.source and "attention" not in cell.source
                if reference and set(cell.metadata.get("tags",[])) & {"setup","definitions"}:
                    exec(compile(cell.source,"definitions","exec"),ns)
        if not reference:
            code="\n".join(c.source for c in nb.cells if c.cell_type=="code")
            remaining = [node.args[0].value for node in maker.todo_calls(code)]
            assert len(remaining)==len(set(remaining))
            assert set(remaining)<=set(maker.ANSWERS)
            assert "PARAM_NAME" in code and "tuning_results" in code
            print("学生版尚待填写的小空:",len(remaining))
        if reference:
            assert not any(maker.todo_calls(c.source) for c in nb.cells if c.cell_type=="code")
    print("PASS 当前学生底稿、参考表达式、语法与代码范围")
    return ns


def check_data(ns):
    raw,cites=u.read_raw_cora()
    ids,classes,x,y,edges=ns["encode_cora"](raw,cites)
    assert x.shape==(2708,1433) and edges.shape==(2,13264) and len(cites)==5429
    audit=u.graph_audit(edges,len(y))
    assert (audit["undirected_edges"],audit["isolates"],audit["components"],audit["largest_component"])==(5278,0,78,2485)
    split=u.stratified_split(y)
    assert [len(split[k]) for k in ["train","val","test"]]==[140,500,2068]
    assert torch.cat(list(split.values())).unique().numel()==2708
    assert all(torch.equal(v,u.stratified_split(y)[k]) for k,v in split.items())
    sets=ns["nested_train_sets"](split["train"],y)
    for budget,indices in sets.items():
        assert torch.equal(torch.bincount(y[indices]),torch.full((7,),budget))
        assert torch.isin(indices,split["train"]).all()
        assert not torch.isin(indices,torch.cat([split["val"],split["test"]])).any()
    assert torch.isin(sets[5],sets[10]).all() and torch.equal(sets[20],split["train"])
    graph75=u.retain_undirected_edges(edges,len(y),.75)
    graph50=u.retain_undirected_edges(edges,len(y),.5)
    for fraction in [0.0,0.5,0.75,1.0]:
        assert torch.equal(ns["retain_undirected_edges"](edges,len(y),fraction),
                           u.retain_undirected_edges(edges,len(y),fraction))
    pairs=lambda e:set(map(tuple,e.T.tolist()))
    p50,p75,p100=pairs(graph50),pairs(graph75),pairs(edges)
    assert p50<=p75<=p100
    assert all((j,i) in p50 for i,j in p50)
    assert all((i,i) in p50 for i in range(len(y)))
    assert u.graph_audit(graph75,len(y))["undirected_edges"]==3958
    assert u.graph_audit(graph50,len(y))["undirected_edges"]==2639
    assert torch.equal(u.retain_undirected_edges(edges,len(y),0),u.self_edges(len(y)))
    expected={f"{m}_L{b}_E100" for m in ["MLP","GCN"] for b in [5,10,20]}|{"GCN_L20_E75","GCN_L20_E50"}
    assert {u.case_key(s) for s in ns["label_experiment_plan"]()+ns["structure_experiment_plan"]()}==expected
    print("PASS 数据、连通性、固定划分、嵌套标签、双向删边与八配置")
    return ids,classes,x,y,edges,split,sets


def check_math(ns):
    adjacency=torch.zeros(5,5,dtype=torch.bool)
    for a,b in [(0,1),(0,2),(2,3)]:
        adjacency[a,b]=adjacency[b,a]=True
    edges=torch.stack(torch.where(adjacency|torch.eye(5,dtype=torch.bool)))
    op=ns["normalized_operator"](edges,5)
    degree=torch.tensor([3.,2.,3.,2.,1.])
    expected=(adjacency.float()+torch.eye(5))/torch.sqrt(degree[:,None]*degree[None,:])
    assert torch.allclose(op.to_dense(),expected)
    assert torch.allclose(ns["dense_normalization"](adjacency),expected)
    assert torch.allclose(ns["dense_normalization"](adjacency,"mean").sum(1),torch.ones(5))
    dropped=edges[:,~(((edges[0]==0)&(edges[1]==1))|((edges[0]==1)&(edges[1]==0)))]
    changed=ns["normalized_operator"](dropped,5).to_dense()
    assert abs(float(changed[0,0])-.5)<1e-6 and changed[1,1]==1
    torch.manual_seed(17)
    x=torch.randn(5,3,dtype=torch.float64,requires_grad=True)
    sparse=op.double()
    layer=ns["GraphLayer"](3,2).double()
    other=copy.deepcopy(layer)
    other_x=x.detach().clone().requires_grad_()
    actual=layer(x,sparse)
    expected=sparse.to_dense()@other.linear(other_x)
    assert torch.allclose(actual,expected,atol=1e-12)
    actual.square().sum().backward()
    expected.square().sum().backward()
    assert torch.allclose(x.grad,other_x.grad,atol=1e-12)
    assert torch.allclose(layer.linear.weight.grad,other.linear.weight.grad,atol=1e-12)
    weights=torch.randn(3,2,dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(lambda a,b:u.propagate(sparse,a@b),(x,weights))
    perm=torch.tensor([3,1,4,0,2]); inverse=torch.empty_like(perm); inverse[perm]=torch.arange(5)
    changed_op=ns["normalized_operator"](inverse[edges],5).double()
    assert torch.allclose(layer(x[perm],changed_op),actual[perm],atol=1e-12)
    assert torch.allclose(layer(x,u.identity_operator(5).double()),layer.linear(x))
    a,b=ns["make_model"](42),ns["make_model"](42)
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    assert u.parameter_counts(a)=={"stored":23040,"trainable":23040}
    print("PASS 手算、重算度数、稠密／稀疏梯度、有限差分、置换与 MLP 退化")


def check_training(ns):
    torch.manual_seed(31)
    x=torch.randn(5,3); op=u.identity_operator(5)
    model=ns["GraphClassifier"](3,4,2,0)
    manual,other=copy.deepcopy(model),copy.deepcopy(model)
    train,val=torch.tensor([0,2]),torch.tensor([1,4])
    ty,vy=torch.tensor([1,0]),torch.tensor([0,1])
    optimizer=torch.optim.Adam(manual.parameters(),lr=.01,weight_decay=.0005)
    optimizer.zero_grad(set_to_none=True)
    F.cross_entropy(manual(x,op)[train],ty).backward(); optimizer.step()
    u.fit_model(model,x,op,train,ty,val,vy,epochs=1,log_every=0)
    u.fit_model(other,x,op,train,ty,val,1-vy,epochs=1,log_every=0)
    for key,value in model.state_dict().items():
        assert torch.equal(value,manual.state_dict()[key]) and torch.equal(value,other.state_dict()[key])
    frozen=u.snapshot(model)
    with torch.no_grad(): next(model.parameters()).add_(1)
    assert not torch.equal(next(model.parameters()),frozen["layer1.linear.weight"])
    model.load_state_dict(frozen)
    run=u.fit_model(model,x,op,train,ty,val,vy,epochs=6,log_every=0)
    best=max(run["history"],key=lambda r:(r["val_acc"],-r["val_loss"]))
    assert best["epoch"]==run["best_epoch"]
    assert abs(float(F.cross_entropy(model(x,op)[val],vy).detach())-best["val_loss"])<1e-7
    metrics=u.classification_metrics(F.one_hot(torch.tensor([0,1,1,1]),3).float(),torch.tensor([0,0,1,2]),3)
    assert metrics["accuracy"]==.5 and abs(metrics["macro_f1"]-7/18)<1e-12
    print("PASS 训练标签掩码、验证标签隔离、独立快照、最佳轮次恢复与 Macro-F1")


def execute_notebook():
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    notebook=nbformat.read(ROOT/"实验七_参考答案.ipynb",as_version=4)
    with tempfile.TemporaryDirectory(prefix="lab7-kernel-") as temporary:
        manager=KernelManager(kernel_name="python3",transport="ipc",ip=str(Path(temporary)/"ipc"))
        manager.kernel_spec.argv[0]=sys.executable
        client=NotebookClient(notebook,km=manager,timeout=600,resources={"metadata":{"path":str(ROOT)}})
        client.on_cell_start=lambda cell,cell_index,**kwargs:print(f"EXEC {cell_index+1}: {cell.source.splitlines()[0][:60]}",flush=True)
        try:
            client.execute(env=dict(os.environ))
        finally:
            if manager.has_kernel: manager.shutdown_kernel(now=True)
            manager.cleanup_resources()
    assert all(c.execution_count is not None and not any(o.output_type=="error" for o in c.outputs)
               for c in notebook.cells if c.cell_type=="code")
    nbformat.write(notebook,ROOT/"实验七_参考答案.ipynb")
    print("PASS 参考 notebook 从头运行；基线、两项研究与Q7单变量调参均通过")
    print("Q7只使用验证结果，不写入旧版最终测试记录。")


def repeat_third_seed(ns,data):
    ids,classes,x,y,edges,split,sets=data
    main=json.loads((ROOT/"参考运行记录.json").read_text())
    assert main["version"]=="GCN-160min" and main["config"]==ns["CONFIG"]
    assert ns["DEVICE"].type=="cpu"
    records=dict(main["runs"])
    specs={u.case_key(s):s for s in ns["label_experiment_plan"]()+ns["structure_experiment_plan"]()}
    for case,spec in specs.items():
        graph=ns["retain_undirected_edges"](edges,len(y),spec["keep"],ns["CONFIG"]["edge_seed"])
        operator=u.identity_operator(len(y)) if spec["model"]=="MLP" else ns["normalized_operator"](graph,len(y))
        indices=sets[spec["labels"]]
        run=u.fit_model(ns["make_model"](3407),x,operator,indices,y[indices],split["val"],y[split["val"]],epochs=200,log_every=0)
        with torch.no_grad():
            run["test"]=u.classification_metrics(run["model"](x,operator)[split["test"]],y[split["test"]])
        run.update(spec=spec,seed=3407)
        records[f"{case}_s3407"]={k:v for k,v in run.items() if k!="model"}
        print("REPEAT",case,round(run["test"]["accuracy"],4),flush=True)
    report={"version":"GCN-160min","seeds":[42,2026,3407],"config":main["config"],"environment":main["environment"],
            "runs":records,"test_accuracy_summary":u.summarize_cases(records,"accuracy"),
            "test_macro_f1_summary":u.summarize_cases(records,"macro_f1"),
            "note":"固定划分、标签样本与缺失图；样本标准差 ddof=1，仅描述训练随机性。"}
    (ROOT/"重复运行记录.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--execute",action="store_true"); parser.add_argument("--repeat",action="store_true")
    args=parser.parse_args()
    ns=definitions(); data=check_data(ns); check_math(ns); check_training(ns)
    if args.execute: execute_notebook()
    if args.repeat: repeat_third_seed(ns,data)
