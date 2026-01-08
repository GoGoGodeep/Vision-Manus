import streamlit as st
import numpy as np
import cv2
from PIL import Image
import time

# =========================
# Tools (模型占位)
# =========================

def segment_full(image):
    h, w = image.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    cv2.rectangle(mask, (w//4, h//4), (3*w//4, 3*h//4), 255, -1)
    return mask

def tile_image(image, tile_size=512, overlap=64):
    h, w = image.shape[:2]
    tiles = []
    for y in range(0, h, tile_size - overlap):
        for x in range(0, w, tile_size - overlap):
            tile = image[y:y+tile_size, x:x+tile_size]
            tiles.append((x, y, tile))
    return tiles

def segment_tile(tile):
    return np.ones(tile.shape[:2], np.uint8) * 255

def merge_tiles(tiles, masks, image_shape):
    H, W = image_shape[:2]
    merged = np.zeros((H, W), np.float32)
    weight = np.zeros((H, W), np.float32)
    for (x, y, _), m in zip(tiles, masks):
        h, w = m.shape
        merged[y:y+h, x:x+w] += m
        weight[y:y+h, x:x+w] += 1
    merged /= np.maximum(weight, 1e-6)
    return (merged > 127).astype(np.uint8) * 255

def postprocess(mask):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# =========================
# Evaluator
# =========================

def evaluate_mask(mask):
    h, w = mask.shape
    area = h * w
    fg = np.sum(mask > 0)

    coverage = fg / area
    if coverage < 0.01 or coverage > 0.95:
        return 0.0, "❌ Coverage abnormal"

    _, labels = cv2.connectedComponents(mask > 0)
    largest = np.max(np.bincount(labels.flat)[1:]) if labels.max() > 0 else 0
    connectivity = largest / max(fg, 1)

    edges = cv2.Canny(mask, 100, 200)
    smoothness = 1 - np.sum(edges > 0) / area

    score = 0.4 * coverage + 0.4 * connectivity + 0.2 * smoothness
    return score, "OK"

# =========================
# Streamlit UI
# =========================

st.set_page_config(layout="wide")
st.title("🧠 Vision Manus Agentx")

with st.sidebar:
    uploaded = st.file_uploader("上传图像", type=["jpg", "png"])
    score_thresh = st.slider("质量阈值", 0.5, 0.95, 0.85)
    tile_size = st.selectbox("Tile Size", [256, 512, 768])
    overlap = st.selectbox("Overlap", [32, 64, 128])
    run = st.button("🚀 运行 Agent")

if uploaded:
    image = np.array(Image.open(uploaded).convert("RGB"))
    st.image(image, caption="输入图像", use_column_width=True)

if uploaded and run:
    log = st.empty()
    progress = st.progress(0)

    # ---------- Step 1 ----------
    log.markdown("### Step 1️⃣ 全图分割")
    mask = segment_full(image)
    score, msg = evaluate_mask(mask)
    st.image(mask, caption=f"Score={score:.3f} | {msg}", clamp=True)
    progress.progress(25)
    time.sleep(0.5)

    if score >= score_thresh:
        st.success("✅ 全图分割通过")
        st.stop()

    # ---------- Step 2 ----------
    log.markdown("### Step 2️⃣ 分块分割 + 融合")
    tiles = tile_image(image, tile_size, overlap)
    masks = []

    for i, (_, _, t) in enumerate(tiles):
        masks.append(segment_tile(t))
        progress.progress(25 + int(50 * (i+1) / len(tiles)))

    merged = merge_tiles(tiles, masks, image.shape)
    score, msg = evaluate_mask(merged)
    st.image(merged, caption=f"Score={score:.3f} | {msg}", clamp=True)
    time.sleep(0.5)

    if score >= score_thresh:
        st.success("✅ 分块融合通过")
        st.stop()

    # ---------- Step 3 ----------
    log.markdown("### Step 3️⃣ 后处理修复")
    refined = postprocess(merged)
    score, msg = evaluate_mask(refined)
    st.image(refined, caption=f"Score={score:.3f} | {msg}", clamp=True)
    progress.progress(100)

    if score >= score_thresh:
        st.success("✅ 后处理通过")
    else:
        st.warning("⚠ 未达标，返回最优结果")
