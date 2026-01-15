# rag/strategy_writer.py
import json
import os
from datetime import datetime


def summarize_strategy(memory):
    steps = memory.steps   # 假设你内部是 list[dict]

    path = []

    for s in steps:
        tool = s["tool"]
        params = s.get("params", {})
        if params:
            path.append(f"{tool}({params})")
        else:
            path.append(tool)

    return {
        "path": path,
    }


class StrategyWriter:

    def __init__(self, path="/home/kexin/hd1/zkf/VisionManus/rag/strategy_cases.jsonl"):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def append(self, record: dict):
        record = dict(record)
        record["created_at"] = datetime.now().strftime("%Y-%m-%d")

        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
