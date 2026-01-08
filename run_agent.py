import streamlit as st
import cv2
import numpy as np
from agent.visionmanus_no_detect import vision_manus_segment

st.set_page_config(layout="wide")
st.title("Vision-Manus")

uploaded = st.file_uploader("Upload an image", type=["jpg", "png"])

score_thresh = st.slider(
    "Score Threshold",
    min_value=0.5,
    max_value=0.95,
    value=0.85,
    step=0.01
)

# 用于保存 agent 运行轨迹
if "steps" not in st.session_state:
    st.session_state.steps = []

def step_callback(step, image, mask, score):
    st.session_state.steps.append({
        "step": step,
        "image": image,
        "mask": mask,
        "score": score
    })

if uploaded:
    st.session_state.steps.clear()

    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    st.subheader("Input Image")
    st.image(image, use_container_width=True)

    if st.button("Run Vision Manus Agent"):
        agent = vision_manus_segment(step_callback=step_callback)
        final_mask = agent.run(image, score_thresh)

    # === 可视化执行过程 ===
    for i, s in enumerate(st.session_state.steps):
        with st.expander(f"Step {i+1}: {s['step']}", expanded=True):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Input Image**")
                st.image(s["image"], use_container_width=True)

            with col2:
                st.markdown("**Mask Output**")
                st.image(
                    s["mask"],
                    use_container_width=True,
                    clamp=True
                )

            st.metric(
                label="Evaluation Score",
                value=f"{s['score']:.3f}"
            )
