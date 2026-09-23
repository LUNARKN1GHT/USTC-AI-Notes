"""实验二平台自动判题参考答案，供教师录入与验收使用。"""


def parse_record(line):
    sample_id, true_text, pred_text, confidence_text = line.split()

    try:
        true_label = int(true_text)
        pred_label = int(pred_text)
        confidence = float(confidence_text)
    except ValueError:
        return None

    if true_label not in (0, 1) or pred_label not in (0, 1):
        return None
    if confidence < 0 or confidence > 1:
        return None

    return {
        "id": sample_id,
        "true": true_label,
        "pred": pred_label,
        "confidence": confidence,
    }


def get_status(accuracy, valid_count):
    if accuracy >= 80 and valid_count >= 5:
        return "reliable"
    if accuracy >= 60:
        return "usable"
    return "needs_review"


n = int(input())
records = []
invalid_count = 0

for _ in range(n):
    record = parse_record(input())
    if record is None:
        invalid_count += 1
        continue
    records.append(record)

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

    if record["true"] == record["pred"]:
        correct_count += 1
    else:
        wrong_count += 1
        if record["confidence"] > max_wrong_confidence:
            max_wrong_confidence = record["confidence"]
            max_wrong_id = record["id"]

    if record["true"] == 1 and record["pred"] == 1:
        tp += 1
    elif record["true"] == 0 and record["pred"] == 1:
        fp += 1
    elif record["true"] == 1 and record["pred"] == 0:
        fn += 1
    else:
        tn += 1

if valid_count > 0:
    accuracy = correct_count / valid_count * 100
    average_confidence = confidence_sum / valid_count
else:
    accuracy = 0.0
    average_confidence = 0.0

status = get_status(accuracy, valid_count)

print(valid_count, invalid_count)
print(correct_count, wrong_count)
print(tp, fp, fn, tn)
print(f"{accuracy:.1f}")
print(f"{average_confidence:.2f}")
print(max_wrong_id)
print(status)
