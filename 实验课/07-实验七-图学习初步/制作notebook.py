"""以当前学生版为唯一底稿导出参考答案；本脚本不会写入学生版。

python 制作notebook.py             导出填好 TODO 的参考版，清空其旧输出
python 验证参考答案.py --execute   从头执行参考版

教师直接编辑学生 notebook；删除的单元格和文案不会从历史模板恢复。
"""
from __future__ import annotations

import argparse
import ast
import copy
import textwrap
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent
STUDENT = ROOT / "实验七_学生版.ipynb"
REFERENCE = ROOT / "实验七_参考答案.ipynb"

# 只保存各小空的参考表达式，不保存旧课程文案、Q8 或整份学生模板。
ANSWERS = {
    "Q1.1": "[row[0] for row in contents]",
    "Q1.2": "row[-1]",
    "Q1.3": "row[1:-1]",
    "Q1.4": "(id_to_idx[cited], id_to_idx[citing])",
    "Q1.5": "(v, u)",
    "Q2a.1": "a_hat.sum(dim=1)",
    "Q2a.2": "degree.pow(-0.5)",
    "Q2a.3": "inv_sqrt[:, None] * a_hat * inv_sqrt[None, :]",
    "Q2b.1": "inv_sqrt[receiver] * inv_sqrt[sender]",
    "Q2c.1": "(0, 2)",
    "Q2c.2": "0",
    "Q3.1": "self.linear(x)",
    "Q3.2": "propagate(operator, transformed)",
    "Q5a.1": "pool[labels[pool] == category]",
    "Q5a.2": "shuffled[:count]",
    "Q5b.1": "model_name",
    "Q5b.2": "count",
    "Q6a.1": "edges[0] < edges[1]",
    "Q6a.2": "int(pairs.shape[1] * keep)",
    "Q6a.3": "chosen.flip(0)",
    "Q6b.1": "fraction",
}


def todo_calls(source):
    return [node for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "TODO"]


def solve_source(source):
    """按 AST 位置替换表达式，保留教师写的注释、文档字符串与其他代码。"""
    raw = source.encode("utf-8")
    lines = raw.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    replacements = []
    for node in todo_calls(source):
        assert len(node.args) == 1 and isinstance(node.args[0], ast.Constant)
        key = node.args[0].value
        if key not in ANSWERS:
            raise KeyError(f"参考表达式未定义：{key}；请在 ANSWERS 中补充")
        start = offsets[node.lineno - 1] + node.col_offset
        end = offsets[node.end_lineno - 1] + node.end_col_offset
        replacements.append((start, end, ANSWERS[key].encode("utf-8")))
    for start, end, value in sorted(replacements, reverse=True):
        raw = raw[:start] + value + raw[end:]
    result = raw.decode("utf-8")
    ast.parse(result)
    return result


def add_reference_explanations(notebook):
    """仅回答当前仍保留的题目；对应提示删除后，不再附加该项解答。"""
    originals = {cell.id: cell.source for cell in notebook.cells}
    additions = {}

    def add(anchor, source, code=False):
        factory = nbf.v4.new_code_cell if code else nbf.v4.new_markdown_cell
        cell = factory(textwrap.dedent(source).strip(), metadata={"tags": ["reference-answer"]})
        additions.setdefault(anchor, []).append(cell)

    q2 = originals.get("14ac4011", "")
    if "运行前手算节点 0" in q2:
        placeholder = next((c for c in notebook.cells if c.id == "3ec26615"), None)
        explanation = """**Q2 参考解答**

含自环的度数为 `(3,2,3,2,1)`。对节点 0：

- 行均值系数 `P[0,1]=1/3`，输出为 `(2/3,2/3)`。
- 对称归一化系数 `S[0,1]=1/√6`，输出为 `(2/3,1/3+1/√6)≈(0.6667,0.7416)`。

P 的每行和为 1，但不同端点度数不同时一般不对称；S 对称，但每行和不一定为 1。
S 中一条边的系数同时取决于两个端点的度数，因此不能直接理解为普通邻居平均。
节点 4 只有自环，归一化后保留自身特征。

节点 3 到节点 0 的最短路径是两跳。一次传播只接收一跳邻居，修改节点 3 不影响节点 0；两次传播可以经节点 2 产生影响。
这是对本题无中间非线性的传播检查，不能据此保证任意训练网络中每次扰动都产生可见的最终预测变化。"""
        if placeholder is not None:
            placeholder.source = explanation
        else:
            add("b9ccfc21", explanation)
    if "Q2c" in q2 and "1dd5f4bb" in originals:
        add("1dd5f4bb", """
        **Q2c 填写示例**

        第二条边选 `(0,2)`，检查第 `0` 行。含自环度数为 `(3,2,2)`，第 0 行的系数为
        `(1/3,1/√6,1/√6)`，和约为 `1.1498`。它不是 1，因此这张小图已构成反例。
        选择 `(1,2)` 也可以；那时度数和具体系数不同，应按自己的输出解释。
        """)

    q4 = originals.get("81363bd3", "")
    choices = {
        "Q4-A": "**Q4-A 参考思路：** 先预测 GCN 是否更好，再用验证均值及标准差判断。两模型同为 23,040 个参数，传播算子由 I 换成 S 后，邻居特征能影响节点表示。结论限定在当前图、划分与预算。",
        "Q4-B": "**Q4-B 参考思路：** 训练集分数不能代替验证表现。结合曲线与训练／验证差距，查看最佳验证轮次和最后轮次。按验证结果保存模型，而不预设最后一轮最好。",
        "Q4-C": "**Q4-C 参考思路：** 分别报告两个种子下的模型差值，再看均值与标准差。只挑最好一次会隐藏训练波动，并可能夸大方法效果；两种子也不足以证明普遍稳定。",
    }
    if any(label in q4 for label in choices):
        add("81363bd3", "\n\n".join(["**Q4 选答参考**：下面给出当前保留题目的参考思路，学生任选两题回答。"]
                                     + [text for label, text in choices.items() if label in q4]))
        add("bc4ee834", """
        # Q4-A：普通打印验证结果，不需要额外的显示工具。
        baseline_summary_ref = summarize_cases(records)
        print("Q4-A：比较验证表现")
        for name_ref in ["MLP", "GCN"]:
            items_ref = [r for r in records.values() if r["spec"] == dict(model=name_ref, labels=20, keep=1.0)]
            score_ref = baseline_summary_ref[f"{name_ref}_L20_E100"]
            gaps_ref = [r["history"][r["best_epoch"]-1]["train_acc"] - r["best_val_acc"] for r in items_ref]
            print(name_ref, "验证均值:", round(score_ref["mean"],4),
                  "标准差:", round(score_ref["sample_std"],4),
                  "训练与验证的平均差距:", round(float(np.mean(gaps_ref)),4),
                  "参数量:", items_ref[0]["parameters"]["trainable"])
        baseline_delta_ref = (baseline_summary_ref["GCN_L20_E100"]["mean"] - baseline_summary_ref["MLP_L20_E100"]["mean"])*100
        print("GCN−MLP:", round(baseline_delta_ref,2), "个百分点")
        print("两模型只改变传播算子；结果描述当前图和配置，不能推广到所有图任务。")
        """, code=True)

    if "Q4-B" in q4:
        add("bc4ee834", """
        print("Q4-B：最佳轮次与最后轮次")
        for name_ref in ["MLP", "GCN"]:
            for seed_ref in CONFIG["seeds"]:
                run_ref = records[f"{name_ref}_L20_E100_s{seed_ref}"]
                print(name_ref, "种子:", seed_ref, "最佳轮次:", run_ref["best_epoch"],
                      "最佳验证准确率:", round(run_ref["best_val_acc"],4),
                      "最后一轮验证准确率:", round(run_ref["history"][-1]["val_acc"],4))
        print("训练集分数不能替代验证表现；继续训练不保证验证准确率继续提高。")
        print("根据验证集保存模型，不能只看训练分数或固定取最后一轮。")
        """, code=True)
    if "Q4-C" in q4:
        add("bc4ee834", """
        print("Q4-C：逐个种子的比较")
        seed_deltas_ref = []
        for seed_ref in CONFIG["seeds"]:
            mlp_acc_ref = records[f"MLP_L20_E100_s{seed_ref}"]["best_val_acc"]
            gcn_acc_ref = records[f"GCN_L20_E100_s{seed_ref}"]["best_val_acc"]
            difference_ref = (gcn_acc_ref - mlp_acc_ref)*100
            seed_deltas_ref.append(difference_ref)
            print("种子:",seed_ref,"MLP:",round(mlp_acc_ref,4),"GCN:",round(gcn_acc_ref,4),
                  "GCN−MLP:",round(difference_ref,2),"个百分点")
        print("两个种子下是否都由 GCN 领先:", all(d > 0 for d in seed_deltas_ref))
        print("只报告最好一次会隐藏训练波动。两个初始化不足以证明普遍稳定。")
        """, code=True)

    if "每类训练标签数" in originals.get("79df9ce6", ""):
        add("79df9ce6", """
        **Q5 假设与划分参考**

        假设：增加训练标签通常有助于分类，图传播可能在更少标签时提供较大的补充收益；趋势需要实测，不预设严格单调。
        改变每类可用标签数，固定训练池、全图、验证与测试节点、预处理和优化配置。
        每类只随机排序一次，再取前 5、10、20 个，所以训练集合嵌套；未启用训练池节点不参与损失，也不移动到评价集合。
        """)
        add("6ac3af19", """
        label_summary_ref = summarize_cases(records)
        print("Q5 填表参考：以下均为验证结果")
        for count_ref in [5,10,20]:
            mlp_ref = label_summary_ref[f"MLP_L{count_ref}_E100"]
            gcn_ref = label_summary_ref[f"GCN_L{count_ref}_E100"]
            print("每类标签:",count_ref,"训练节点:",len(label_sets[count_ref]))
            print("  MLP 均值:",round(mlp_ref["mean"],4),"标准差:",round(mlp_ref["sample_std"],4))
            print("  GCN 均值:",round(gcn_ref["mean"],4),"标准差:",round(gcn_ref["sample_std"],4))
            print("  GCN−MLP:",round((gcn_ref["mean"]-mlp_ref["mean"])*100,2),"个百分点")
        for name_ref in ["MLP","GCN"]:
            gain_ref = (label_summary_ref[f"{name_ref}_L20_E100"]["mean"] - label_summary_ref[f"{name_ref}_L5_E100"]["mean"])*100
            print(name_ref,"从每类5到20标签的变化:",round(gain_ref,2),"个百分点")
        print("如果每个预算都重划数据，还会改变训练和评价样本，不能单独归于标签数。")
        print("结论仅针对这套固定嵌套样本、图和初始化，未覆盖所有标签抽样。")
        """, code=True)

    q6 = originals.get("0119cf34", "")
    if "没有外部邻居的节点数" in q6:
        add("0119cf34", """
        **Q6 假设与对照参考**

        假设：引用缺失可能减少有用的传播路径，使 GCN 分类变差，但也可能移除干扰关系，因此不预设每次删边都使性能下降。
        固定模型、标签、优化配置与评价节点，只改变可获得的引用关系；每张图重新计算度数并独立训练。
        无向关系按同一乱序截取前缀，两个方向一起处理，自环和全部节点保留。
        """)
    if "没有外部邻居的节点数" in q6:
        add("fb0e16d4", """
        structure_summary_ref = summarize_cases(records)
        print("Q6 填表参考")
        full_score_ref = structure_summary_ref["GCN_L20_E100"]["mean"]
        for keep_ref in [1.0,0.75,0.5]:
            audit_ref = graph_audit(graph_versions[keep_ref],len(y))
            score_ref = structure_summary_ref[f"GCN_L20_E{round(keep_ref*100)}"]
            print("引用保留比例:",keep_ref,"无向边:",audit_ref["undirected_edges"],
                  "无外部邻居节点:",audit_ref["isolates"],"连通分量:",audit_ref["components"])
            print("  验证均值:",round(score_ref["mean"],4),"标准差:",round(score_ref["sample_std"],4),
                  "相对完整图变化:",round((score_ref["mean"]-full_score_ref)*100,2),"个百分点")
        print("自环保留自身信息，但不能阻止外部连接消失或图分裂。")
        print("每张缺失图都重新训练；这与在固定模型推理时删边不同。")
        print("应报告全部种子，固定图上的两次初始化没有覆盖图抽样本身的波动。")
        """, code=True)
    if "用 Q2 的小图说明" in q6:
        add("fb0e16d4", """
        # Q6 小图解答：删除 0—1 后，仅将原 S 的对应两项置零是不够的。
        keep_mask_ref = ~(((toy_edges[0] == 0) & (toy_edges[1] == 1)) |
                          ((toy_edges[0] == 1) & (toy_edges[1] == 0)))
        reduced_edges_ref = toy_edges[:, keep_mask_ref]
        wrong_s_ref = symmetric_op.clone()
        wrong_s_ref[0, 1] = wrong_s_ref[1, 0] = 0
        correct_s_ref = normalized_operator(reduced_edges_ref, 5).to_dense()
        print("删边前含自环度数:", torch.bincount(toy_edges[0], minlength=5).tolist())
        print("删边后含自环度数:", torch.bincount(reduced_edges_ref[0], minlength=5).tolist())
        print("错误做法的自环系数:", wrong_s_ref.diag().tolist())
        print("重新归一化的自环系数:", correct_s_ref.diag().tolist())
        assert not torch.allclose(wrong_s_ref, correct_s_ref)
        assert torch.allclose(correct_s_ref[0, 0], torch.tensor(0.5))
        assert torch.allclose(correct_s_ref[1, 1], torch.tensor(1.0))
        # 节点 0 的自环系数由 1/3 变为 1/2，节点 1 由 1/2 变为 1。
        # 不仅被删边本身，剩余相邻边的系数也应随新度数改变。
        """, code=True)

    q7 = originals.get("c3f67230", "")
    if "调参" in q7:
        add("c3f67230", """
        **Q7 参考思路**

        以默认学习率为例，可以先猜：较大学习率可能下降更快，但波动也可能更大；较小学习率在固定轮数下可能来不及学好。
        运行后用曲线和两次验证结果判断，而不要求新值一定优于原设置。研究隐藏层宽度或 dropout 时也一次只改一个变量。
        """)
        add("750033f1", """
        print("Q7 参考分析")
        print("本次研究的参数:", PARAM_NAME, "原设置:", CONFIG[PARAM_NAME])
        original_score_ref = tuning_results[CONFIG[PARAM_NAME]]["val_mean"]
        for value_ref in PARAM_VALUES:
            result_ref = tuning_results[value_ref]
            print("参数值:",value_ref,"验证均值:",round(result_ref["val_mean"],4),
                  "标准差:",round(result_ref["val_std"],4),
                  "相对原设置变化:",round((result_ref["val_mean"]-original_score_ref)*100,2),"个百分点")
        best_value_ref = max(PARAM_VALUES,key=lambda value:tuning_results[value]["val_mean"])
        print("本次验证均值最高的参数值:",best_value_ref)
        print("还要看两个种子的结果与曲线，不能只凭某一次训练判断。")
        print("这些候选值和初始化数量有限，不能声称找到了最优参数。")
        """, code=True)

    result = []
    for cell in notebook.cells:
        result.append(cell)
        result.extend(additions.get(cell.id, []))
    notebook.cells = result
    return notebook


def build(reference=False):
    notebook = copy.deepcopy(nbf.read(STUDENT, as_version=4))
    if reference:
        for cell in notebook.cells:
            if cell.cell_type == "code":
                cell.source = solve_source(cell.source)
                cell.outputs = []
                cell.execution_count = None
        notebook = add_reference_explanations(notebook)
    nbf.validate(notebook)
    return notebook


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-notes", action="store_true", help="兼容旧命令；不恢复旧模板文案")
    args = parser.parse_args()
    if args.refresh_notes:
        print("当前以学生版为唯一底稿，没有独立旧文案需要刷新；未修改任何 notebook。")
    else:
        before = STUDENT.read_bytes()
        nbf.write(build(reference=True), REFERENCE)
        assert STUDENT.read_bytes() == before
        print("已导出参考答案；学生版及其删减保持不变。")
