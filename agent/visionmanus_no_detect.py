from agent.segment import segmenter_iSeg
from agent.evaluation import evaluate

from tools.postprocess import postprocess_preserve_small


class vision_manus_segment:
    def __init__(self, device="cuda", step_callback=None):
        self.device = device
        self.segment_full = segmenter_iSeg()
        self.evaluate_mask = evaluate()
        self.postprocess = postprocess_preserve_small

        self.step_callback = step_callback  # 👈 关键

    def _notify(self, step_name, image=None, mask=None, score=None):
        if self.step_callback:
            self.step_callback(
                step=step_name,
                image=image,
                mask=mask,
                score=score
            )

    def run(self, image, score_thresh=0.85):
        # Step 1: 全图分割
        mask = self.segment_full.segment(image)
        score = self.evaluate_mask.run(mask)
        self._notify("Full Image Segmentation", image, mask, score)

        if score >= score_thresh:
            return mask

        # Step 2: 分块分割
        patch_mask = self.segment_full.patch_segment(image)
        score = self.evaluate_mask.run(patch_mask)
        self._notify("Patch Merge", image, patch_mask, score)

        if score >= score_thresh:
            return patch_mask

        # Step 3: 后处理
        refined = self.postprocess(patch_mask)
        score = self.evaluate_mask.run(refined)
        self._notify("Postprocess", image, refined, score)

        return refined


