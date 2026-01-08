from segment import segmenter_iSeg
from evaluation import evaluate


class vision_manus_segment:
    def __init__(self, device="cuda"):
        self.device = device

        self.segment_full = segmenter_iSeg()
        self.evaluate_mask = evaluate()
        

    def vision_manus_segment(self, image, score_thresh=0.85):
        # Step 1: 全图分割
        mask = self.segment_full(image, {})
        score = self.evaluate_mask(mask)
        if score >= score_thresh:
            return mask

        # Step 2: 分块分割
        tiles = tile_image(image)
        tile_masks = [segment_tile(t, {}) for _, _, t in tiles]
        merged = merge_tiles(tiles, tile_masks, image.shape)
        score, _ = evaluate_mask(merged)
        if score >= score_thresh:
            return merged

        # Step 3: 后处理
        refined = postprocess(merged)
        score, _ = evaluate_mask(refined)
        if score >= score_thresh:
            return refined

        # Step 4: 失败兜底
        print("⚠ segmentation failed, returning best-effort result")
        return refined
