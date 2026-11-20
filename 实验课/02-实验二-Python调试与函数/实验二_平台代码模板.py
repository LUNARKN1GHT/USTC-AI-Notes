"""实验二平台代码填空模板。

学生只需要填写标有 TODO 的位置，不要修改输入和输出代码。
"""


def parse_record(line):
    """将一行文本转换为记录字典；无效时返回 None。"""
    sample_id, true_text, pred_text, confidence_text = line.split()

    # TODO 1：使用 try/except 将标签转为 int、置信度转为 float。
    # 转换失败时返回 None。
    try:
        true_label = __________________
        pred_label = __________________
        confidence = __________________
    except __________________:
        return None

    # TODO 2：标签只能为 0 或 1，置信度必须位于 [0, 1]。
    if __________________:
        return None

    return {
        "id": sample_id,
        "true": true_label,
        "pred": pred_label,
        "confidence": confidence,
    }


def get_status(accuracy, valid_count):
    """根据准确率和有效记录数返回评估状态。"""
    # TODO 3：按题目给出的优先级判断状态。
    if __________________:
        return "reliable"
    elif __________________:
        return "usable"
    else:
        return "needs_review"


n = int(input())
records = []
invalid_count = 0

for _ in range(n):
    record = parse_record(input())
    # TODO 4：无效记录计数后跳过；有效记录加入 records。
    if __________________:
        invalid_count += 1
        continue
    __________________

valid_count = len(records)
correct_count = 0
wrong_count = 0
tp = 0
fp = 0
fn = 0
tn = 0
confidence_sum = 0.0

max_wrong_confidence = -1.0
max_wrong_id = "NONE"

for record in records:
    confidence_sum += record["confidence"]

    # TODO 5：累计正确数和错误数。
    if __________________:
        correct_count += 1
    else:
        wrong_count += 1

        # TODO 6：更新置信度最高的错误样本；并列时保留先出现的样本。
        if __________________:
            max_wrong_confidence = record["confidence"]
            max_wrong_id = record["id"]

    # TODO 7：累计 TP、FP、FN、TN。
    if record["true"] == 1 and record["pred"] == 1:
        __________________
    elif record["true"] == 0 and record["pred"] == 1:
        __________________
    elif record["true"] == 1 and record["pred"] == 0:
        __________________
    else:
        __________________

# TODO 8：避免有效记录数为 0 时发生除零错误。
if __________________:
    accuracy = correct_count / valid_count * 100
    average_confidence = confidence_sum / valid_count
else:
    accuracy = 0.0
    average_confidence = 0.0

status = get_status(accuracy, valid_count)

# 以下输出代码不要修改，也不要增加额外输出。
print(valid_count, invalid_count)
print(correct_count, wrong_count)
print(tp, fp, fn, tn)
print(f"{accuracy:.1f}")
print(f"{average_confidence:.2f}")
print(max_wrong_id)
print(status)
