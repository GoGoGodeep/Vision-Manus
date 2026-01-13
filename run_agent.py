# streamlit run run_agent.py --server.address 0.0.0.0
import streamlit as st
import numpy as np
from PIL import Image
import time
import json

from agent.evaluation import evaluate
from agent.planner import Planner
from agent.prompts import task_understanding_prompt, router_prompt
from agent.segment import segmenter_iSeg


# —————————————————————————————— 页面基础 ——————————————————————————————
st.set_page_config(
    page_title="Vision Manus",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.chat-area { 
    display:flex; 
    flex-direction:column; 
    gap:1.2rem; 
}

.chat-row-user, .chat-row-sys {
    display:flex;
    align-items:flex-end;
    gap:0.6rem;
    margin-top: 0.3rem;
    margin-bottom: 0.3rem;
    justify-content:flex-start;   /* 全部左对齐 */
}

.chat-bubble-user {
    background:#e8f0ff;
    padding:0.7rem 0.9rem;
    border-radius:14px 14px 14px 4px;
    max-width:100%;
    box-shadow:0 1px 3px rgba(0,0,0,0.08);
}

.chat-bubble-sys {
    background:#f5f5f5;
    padding:0.7rem 0.9rem;
    border-radius:14px 14px 14px 4px;
    max-width:100%;
    border:1px solid #eee;
}

.chat-avatar {
    width:32px;
    height:32px;
    border-radius:50%;
    object-fit:cover;
}
</style>
""", unsafe_allow_html=True)

USER_AVATAR = "https://cdn-icons-png.flaticon.com/512/149/149071.png"
SYS_AVATAR  = "https://cdn-icons-png.flaticon.com/512/4712/4712109.png"

def render_chat(logs):
    st.markdown('<div class="chat-area">', unsafe_allow_html=True)
    for role, msg in logs:
        if role == "user":
            st.markdown(f"""
            <div class="chat-row-user">
                <img class="chat-avatar" src="{USER_AVATAR}">
                <div class="chat-bubble-user">{msg}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-row-sys">
                <img class="chat-avatar" src="{SYS_AVATAR}">
                <div class="chat-bubble-sys">{msg}</div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# —————————————————————————————— Session State ——————————————————————————————
if "logs" not in st.session_state:
    st.session_state.logs = []
if "image" not in st.session_state:
    st.session_state.image = None
if "final_mask" not in st.session_state:
    st.session_state.final_mask = None
if "running" not in st.session_state:
    st.session_state.running = False


# —————————————————————————————— Sidebar ——————————————————————————————
with st.sidebar:
    st.markdown("### 输入图片")
    file = st.file_uploader("上传图片", type=["png", "jpg", "jpeg"])
    if file:
        img = Image.open(file).convert("RGB")
        st.session_state.image = img
        st.image(img, caption="输入图片", use_container_width=True)

    st.markdown("---")
    st.markdown("### 用户任务目标")
    user_prompt = st.text_input("请输入任务描述", "Segmenting the pantograph in the image.")

    st.markdown("---")
    run = st.button("运行 Vision Manus")


# —————————————————————————————— 主界面 ——————————————————————————————
st.title("Vision Manus")

if run and user_prompt and st.session_state.image is not None:
    st.session_state.running = True

log_box = st.empty()

evaluator = evaluate()


# —————————————————————————————— 执行流程 ——————————————————————————————
if st.session_state.running:
    # ---- 意图识别 ----
    understander = Planner()
    st.session_state.logs.append(("user", user_prompt))
    with log_box.container():
        render_chat(st.session_state.logs)

    thinking, task = understander.run(
        sys_prompt=task_understanding_prompt,
        user_prompt=user_prompt
    )

    content = json.loads(task)
    user_goal = content.get("user_goal")
    task_object = content.get("task_object")

    st.session_state.logs.append(("sys", f"思考: {thinking}"))
    st.session_state.logs.append(("sys", f"用户目标: {user_goal}, 任务对象: {task_object}"))
    with log_box.container():
        render_chat(st.session_state.logs)

    max_retry = 3
    quality_th = 0.85
    task_model = {
        "Segmentation": "iSeg-Plus",
        "Detection": "YOLO-World"
    }

    st.session_state.logs.append(
        ("sys", f"任务模型: {task_model[user_goal]}, 质量阈值：{quality_th}, 最大重试轮数: {max_retry}")
    )
    with log_box.container():
        render_chat(st.session_state.logs)

    image_seg = segmenter_iSeg()
    time.sleep(0.1)

    # —————————————————————————————— 多轮推理 ——————————————————————————————
    attempt = 1
    while attempt < max_retry + 1:
        st.session_state.logs.append(("sys", f"第 {attempt} 轮推理开始"))
        with log_box.container():
            render_chat(st.session_state.logs)

        st.session_state.logs.append(("sys", f"开始调用 {task_model[user_goal]} 分割模型"))
        with log_box.container():
            render_chat(st.session_state.logs)

        if attempt == 1:
            mask = image_seg.segment(
                class_name=task_object, 
                img=np.array(st.session_state.image)
            )
            st.session_state.logs.append(("sys", f"第 {attempt} 轮 Mask"))
            st.image(mask, width=400)
            with log_box.container():
                render_chat(st.session_state.logs)

            result = evaluator.run(mask)
            st.session_state.logs.append(("sys", f"Eval: {result}"))
        else:
            pass

        router_thinking, router_answer = understander.run(
            sys_prompt=router_prompt,
            user_prompt=str(result)
        )
        st.session_state.logs.append(("sys", f"思考: {router_thinking}"))
        st.session_state.logs.append(("sys", f"回答: {router_answer}"))
        with log_box.container():
                render_chat(st.session_state.logs)
        # if result["score"] > quality_th:
        #     st.session_state.final_mask = mask
        #     st.session_state.logs.append(("sys", "LLM 判定通过，生成 Final Mask"))
        #     with log_box.container():
        #         render_chat(st.session_state.logs)
        #     break
        # else:
        #     st.session_state.logs.append(("sys", "质量不达标，进入失败恢复策略"))
        #     with log_box.container():
        #         render_chat(st.session_state.logs)

        attempt += 1

        time.sleep(0.1)

    if attempt == max_retry + 1:
        st.markdown("### 最终分割结果")
        st.image(mask, width=400)

    st.session_state.running = False