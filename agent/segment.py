from ...iseg.demo import run_one_image


class segmenter_iSeg:
    def __init__(self, device="cuda"):
        self.device = device

        print("Loading iSeg model for segmentation...")
        self.model = run_one_image(model=None, np_img=None, device=self.device)
        print("iSeg model loaded.")

    def segment(self, np_img):
        mask = run_one_image(self.model, np_img, device=self.device)
        mask_np = (mask.squeeze().cpu().numpy() > 0.5).astype('uint8') * 255
        return mask_np