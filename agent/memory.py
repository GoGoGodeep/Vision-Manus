import json
import numpy as np


# ——————————————————————————— 记忆器设计 ———————————————————————————
# 用于保存每一轮的操作、工具、参数、评分、mask 等关键信息，
# 让 LLM 在后续决策时“知道自己之前干了什么”
class Memory:
    def __init__(self):
        self.steps = []

    def _sanitize(self, obj):
        """把不可 JSON 化的对象转成可读字符串"""
        if isinstance(obj, dict):
            return {k: self._sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._sanitize(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return f"<ndarray shape={obj.shape} dtype={obj.dtype}>"
        elif hasattr(obj, "__dict__"):
            return str(obj)
        else:
            return obj

    def add_step(self, info: dict):
        clean_info = self._sanitize(info)
        self.steps.append(clean_info)

    def summary(self, max_steps=5):
        recent = self.steps[-max_steps:]
        # print(recent)
        return json.dumps(recent, ensure_ascii=False, indent=2)