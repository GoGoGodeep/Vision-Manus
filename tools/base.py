from agent.segment import segmenter_iSeg
from tools.postprocess import postprocess_preserve_small


TOOL_REGISTRY = {
    "split_image_patches": segmenter_iSeg().patch_segment,
    "postprocess_preserve_small": postprocess_preserve_small,
}
