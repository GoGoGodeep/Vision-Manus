import cv2
import numpy as np


class evaluate:

    def __init__(self):
        pass

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


    def soft_evaluate(self, mask):
        # Implementation of soft evaluation logic
        return 0.0

    def run(self, mask):
        hard_score, coverage, connectivity, smoothness = self.hard_evaluate(mask)
        soft_score = self.soft_evaluate(mask)

        return {
            "score": hard_score + soft_score,
            "hard_score": hard_score,
            "soft_score": soft_score,
            "coverage": coverage,
            "connectivity": connectivity,
            "smoothness": smoothness
        }