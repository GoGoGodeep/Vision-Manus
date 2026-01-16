import json, re
import cv2
import numpy as np
from transformers import AutoProcessor, AutoModelForImageTextToText
import streamlit as st


@st.cache_resource
def load_model(model_name: str):
    model = AutoModelForImageTextToText.from_pretrained(
        model_name, 
        dtype="auto", 
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_name)
    return processor, model


class evaluate:

    def __init__(self):
        self.model_name = "/home/kexin/hd1/zkf/Qwen3-VL-4bit"
        self.processor, self.model = load_model(self.model_name)


    def hard_evaluate(self, mask):
        """
        对二值/近似二值的分割 mask 进行启发式质量评估（hard rule）。
        主要从三个方面衡量：
        1. 覆盖率（coverage）：前景像素在整幅图中的占比是否合理
        2. 连通性（connectivity）：前景是否主要集中在一个连通区域
        3. 平滑度（smoothness）：边缘是否过于破碎

        返回：
            score (float): 综合评分，范围大致在 [0, 1]
            msg (str): 评估状态说明
        """

        # -----------------------------
        # 1. 基本统计信息
        # -----------------------------
        h, w = mask.shape           # mask 的高和宽
        area = h * w                # 整幅图像的像素总数
        fg = np.sum(mask > 0)       # 前景像素数量（mask > 0 视为前景）

        # -----------------------------
        # 2. 覆盖率约束（Coverage）
        # -----------------------------
        coverage = fg / area        # 前景占整图的比例

        # 如果前景过少（几乎没有目标）
        # 或前景过多（几乎全是目标），直接判为不合理
        if coverage < 0.01 or coverage > 0.95:
            return 0.0, "bad coverage"

        # -----------------------------
        # 3. 连通性评估（Connectivity）
        # -----------------------------
        # 对前景区域做连通域分析
        # num_labels: 连通域数量（包含背景）
        # labels: 每个像素所属的连通域编号
        binary = (mask > 0).astype("uint8")
        num_labels, labels = cv2.connectedComponents(binary)

        # 计算最大前景连通域的像素数量
        # 从 1 开始是为了跳过背景（背景通常是 0）
        largest = max(
            [np.sum(labels == i) for i in range(1, num_labels)],
            default=0
        )

        # 连通性定义为：最大连通域 / 前景像素总数
        # 越接近 1，说明前景越集中、不碎片化
        connectivity = largest / max(fg, 1)

        # -----------------------------
        # 4. 平滑度评估（Smoothness）
        # -----------------------------
        # 使用 Canny 算子检测边缘
        edges = cv2.Canny(mask, 100, 200)

        # 边缘像素占比越小，说明边界越平滑
        # 这里用 1 - 边缘占比 作为平滑度指标
        smoothness = 1.0 - np.sum(edges > 0) / area

        # -----------------------------
        # 5. 综合评分（加权求和）
        # -----------------------------
        score = (
            0.4 * coverage +        # 覆盖率权重
            0.4 * connectivity +    # 连通性权重
            0.2 * smoothness        # 平滑度权重
        )

        return score, coverage, connectivity, smoothness


    def soft_evaluate(self, img, mask, prompt):

        # 统一 mask 为 3 通道
        if isinstance(mask, np.ndarray):
            if mask.ndim == 2:
                mask = np.stack([mask]*3, axis=-1)
            elif mask.ndim == 3 and mask.shape[0] == 1:
                mask = np.repeat(mask, 3, axis=0)
        else:
            if mask.mode != "RGB":
                mask = mask.convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "image", "image": mask},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Preparation for inference
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=1280)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        text = output_text[0] if isinstance(output_text, list) else output_text
        result = json.loads(text)

        coverage_score, coverage_reason, semantic_score, semantic_reason = result["coverage_score"], result["coverage_reason"], result["semantic_score"], result["semantic_reason"]

        return coverage_score, coverage_reason, semantic_score, semantic_reason


    def run(self, img, mask, prompt, visual_concept):
        hard_score, coverage, connectivity, smoothness = self.hard_evaluate(mask)

        prompt_rag = prompt.format(visual_concept)
        coverage_score, coverage_reason, semantic_score, semantic_reason = self.soft_evaluate(img, mask, prompt_rag)
        soft_score = 0.5 * coverage_score + 0.5 * semantic_score

        total_score = round(0.4 * hard_score + 0.6 * soft_score, 4)

        return {
            "score": total_score,
            "hard_score": round(hard_score, 4),
            "soft_score": round(soft_score, 4),
            "coverage": round(coverage, 4),
            "connectivity": round(connectivity, 4),
            "smoothness": round(smoothness, 4)
        }, coverage_reason, semantic_reason