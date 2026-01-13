import numpy as np
from iSeg_Plus.demo import run_one_image, load_model

from PIL import Image


class segmenter_iSeg:
    def __init__(self, device="cuda"):
        self.device = device

        print("Loading iSeg model for segmentation...")
        self.model = load_model(device=self.device, use_half=True)
        print("iSeg model loaded.")


    def segment(self, class_name, img):
        mask = run_one_image(self.model, class_name, img)

        return mask


    def patch_segment(self, class_name, img, rows=2, cols=2, overlap=0,
                          run_args=None):
        """
        将输入图片 np_img 拆成 rows×cols 个 patch（可带重叠），
        分别送入 run_one_image(model, patch, ...)，
        最终拼接回原图。
        保证输出：
            - 尺寸与输入一致
            - 仅包含黑白（0 与 255）两种像素值
        """
        H, W = img.shape[:2]
        run_args = {} if run_args is None else run_args.copy()
        ph = H // rows
        pw = W // cols

        stitched = np.zeros((H, W), dtype=np.float32)
        weight = np.zeros((H, W), dtype=np.float32)

        for r in range(rows):
            for c in range(cols):
                # ---- 计算 patch 范围（带重叠）----
                y0 = max(0, r * ph - overlap)
                y1 = min(H, (r + 1) * ph + overlap if r < rows - 1 else H)
                x0 = max(0, c * pw - overlap)
                x1 = min(W, (c + 1) * pw + overlap if c < cols - 1 else W)

                patch = img[y0:y1, x0:x1]

                # ---- 单 patch 推理 ----
                mask = run_one_image(
                    self.model, class_name, patch,
                    iter_count=run_args.get("iter_count", 5),
                    thr=run_args.get("thr", 0.5),
                    ent=run_args.get("ent", 0.5),
                    device=run_args.get("device", None),
                )
                if mask.ndim == 3:
                    mask = mask[..., 0]
                mask = mask.astype(np.float32)
                if mask.max() > 1:
                    mask /= 255.0

                # ---- 融合权重（平滑边界）----
                h, w = mask.shape
                wy = np.linspace(0, 1, h)
                wx = np.linspace(0, 1, w)
                window = np.outer(np.minimum(wy, wy[::-1]),
                                np.minimum(wx, wx[::-1])) + 1e-6

                stitched[y0:y1, x0:x1] += mask * window
                weight[y0:y1, x0:x1] += window

        # ---- 融合并归一化 ----
        weight[weight == 0] = 1e-6
        merged = stitched / weight
        merged = np.clip(merged, 0, 1)

        # ---- 二值化（仅保留 0 / 255）----
        threshold = 0.5  # 可根据任务调整
        binary = (merged >= threshold).astype(np.uint8) * 255

        # ---- 强制校验尺寸一致 ----
        binary = np.array(Image.fromarray(binary).resize((W, H), Image.NEAREST))
        assert binary.shape == (H, W), f"Size mismatch: got {binary.shape}, expected {(H, W)}"

        return binary
