def parse_score(text: str) -> int:
    return int(text)


def load_score(record: dict[str, str]) -> int:
    raw_score = record["score"]
    return parse_score(raw_score)


def build_report(record: dict[str, str]) -> str:
    score = load_score(record)
    return f"{record['name']}: {score}"


student = {"name": "小科", "score": "NA"}
print(build_report(student))
