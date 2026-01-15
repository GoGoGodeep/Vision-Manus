# streamlit run run_agent.py --server.address 0.0.0.0
import streamlit as st
import numpy as np
from PIL import Image
import time, json

from agent.evaluation import evaluate
from agent.planner import Planner
from agent.prompts import task_understanding_prompt, router_prompt, router_prompt_rag
from agent.segment import segmenter_iSeg
from agent.memory import Memory

from tools.base import TOOL_REGISTRY

from rag.vision_rag import VisionRAG
from rag.strategy_writer import StrategyWriter, summarize_strategy


MAX_RETRY=3


# ——————————————————————————— 页面基础 ———————————————————————————
st.set_page_config(
    page_title="Vision Manus", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    .chat-area { display:flex; flex-direction:column; gap:1.2rem; }
    .chat-row-user,.chat-row-sys{display:flex;gap:.6rem;margin:.3rem 0}
    .chat-bubble-user{background:#e8f0ff;padding:.7rem .9rem;border-radius:14px 14px 14px 4px}
    .chat-bubble-sys{background:#f5f5f5;padding:.7rem .9rem;border-radius:14px 14px 14px 4px;border:1px solid #eee}
    .chat-avatar{width:32px;height:32px;border-radius:50%}
    </style>
    """, 
    unsafe_allow_html=True
)

USER_AVATAR="https://cdn-icons-png.flaticon.com/512/149/149071.png"
SYS_AVATAR="https://cdn-icons-png.flaticon.com/512/4712/4712109.png"

def render_chat(logs):
    st.markdown('<div class="chat-area">', unsafe_allow_html=True)
    for role,msg in logs:
        if role=="user":
            st.markdown(f"""<div class="chat-row-user">
            <img class="chat-avatar" src="{USER_AVATAR}">
            <div class="chat-bubble-user">{msg}</div></div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="chat-row-sys">
            <img class="chat-avatar" src="{SYS_AVATAR}">
            <div class="chat-bubble-sys">{msg}</div></div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ——————————————————————————— Session State ———————————————————————————
if "logs" not in st.session_state: 
    st.session_state.logs=[]
if "running" not in st.session_state: 
    st.session_state.running=False
if "masks" not in st.session_state: 
    st.session_state.masks=[]


# ——————————————————————————— Sidebar ———————————————————————————
with st.sidebar:
    file = st.file_uploader("上传图片", ["png","jpg","jpeg"])
    if file:
        img = Image.open(file).convert("RGB")
        st.session_state.image = img
        st.image(img, caption="输入图片")
    user_prompt = st.text_input("任务描述","Segmenting the pantograph in the image.")
    st.markdown("---")
    run = st.button("运行 Vision Manus")


# ——————————————————————————— 主布局 ———————————————————————————
st.title("Vision Manus")
st.markdown("---")
main_col, right_col = st.columns([3,2])

with main_col:
    log_box = st.empty()
    image_box = st.empty()
with right_col:
    right_box = st.empty()


def render_history():
    with right_box.container():
        st.markdown("## 🧩 历史 Mask")

        # 历史过程
        if not st.session_state.masks:
            st.info("暂无中间结果")
        else:
            for i, m in enumerate(st.session_state.masks, 1):
                st.markdown(f"### 第 {i} 轮 Mask")
                st.image(m, width=400)

        # 最终结果
        if st.session_state.get("final_mask") is not None:
            st.markdown("## ✅ 最终 Mask")
            st.image(st.session_state.final_mask, width=400)


# ——————————————————————————— RAG ———————————————————————————
@st.cache_resource
def load_rag():
    return VisionRAG(
        visual_db_path="/home/kexin/hd1/zkf/VisionManus/rag/visual_concepts.jsonl",
        strategy_db_path="/home/kexin/hd1/zkf/VisionManus/rag/strategy_cases.jsonl"
    )

rag = load_rag()
writer = StrategyWriter()

RAG_WRITE_THRESHOLD = 0.8


# ——————————————————————————— 主流程 ———————————————————————————
if run and user_prompt and st.session_state.get("image") is not None:
    st.session_state.running=True
    st.session_state.masks=[]
    st.session_state.final_mask = None
    render_history()


if st.session_state.running:
    with main_col:
        # 初始化
        evaluator = evaluate()
        understander = Planner()
        memory = Memory()

        # 记录用户输入
        st.session_state.logs.append(("user", user_prompt))
        with log_box.container(): 
            render_chat(st.session_state.logs)

        # 使用 LLM 解析用户意图：返回思考过程和结构化任务
        thinking, task = understander.run(task_understanding_prompt, user_prompt)
        content = json.loads(task)
        user_goal,task_object = content["user_goal"],content["task_object"]

        # —— RAG：视觉概念检索（任务对象视觉知识增强）——
        rag_visual_context, rag_hits = rag.retrieve_visual_concept(task_object, top_k=2)
        strategy_cases, strategy_hits = rag.retrieve_strategy_cases(task_object)

        st.session_state.logs += [
            ("sys",f"RAG hits: {rag_hits}"),
            ("sys",f"Strategy hits: {strategy_hits}"),
            ("sys",f"思考: {thinking}"),
            ("sys",f"用户目标: {user_goal}, 任务对象: {task_object}"),
            ("sys",f"RAG 视觉先验:\n{rag_visual_context}"),
            ("sys",f"调用 iSeg-Plus 分割模型，最大尝试次数 {MAX_RETRY} 次")
        ]
        with log_box.container(): 
            render_chat(st.session_state.logs)

        # 初始化分割模型
        image_seg = segmenter_iSeg()

        # 获取输入图像
        IMG = np.array(st.session_state.image)

        # 初始化尝试次数与工具名
        attempt=1
        tool=""
        params = {}

        # 用于回退：记录历史最优结果
        best_mask = None
        best_score = -1
        best_result = None

        # 进入迭代优化
        while attempt <= MAX_RETRY:
            st.session_state.logs += [("sys",f"进行第 {attempt} 轮操作")]
            with log_box.container(): 
                render_chat(st.session_state.logs)

            # 第一轮：直接分割
            if attempt == 1:
                mask = image_seg.segment(task_object, IMG)

                st.session_state.masks.append(mask)
                render_history()

                # 评估初始分割结果
                result = evaluator.run(mask)
                st.session_state.logs.append(("sys", f"评分：{result}"))
                with log_box.container(): 
                    render_chat(st.session_state.logs)

                # 记录初始分割
                memory.add_step({
                    "round": attempt,
                    "tool": "iSeg-Plus",
                    "params": {"class_name": task_object},
                    "score": result
                })

            # 非第一轮：使用工具微调
            if tool != "Terminate" and attempt > 1:
                mask=TOOL_REGISTRY[tool](**params)

                st.session_state.masks.append(mask)   
                render_history()

                # 对当前 mask 进行质量评估
                result = evaluator.run(mask)
                st.session_state.logs += [("sys",f"评分：{result}")]
                with log_box.container(): 
                    render_chat(st.session_state.logs)

                # ⭐ 记录进记忆器
                memory.add_step({
                    "round": attempt,
                    "tool": tool,
                    "params": params,
                    "score": result
                })

            # print(result)
            # -------- 更新历史最优 --------
            score_val = float(result["score"])
            # print(score_val)

            if score_val > best_score:
                best_score = score_val
                best_mask = mask
                best_result = result

            # ---------- 路由器：结合历史记忆做决策 ----------
            memory_text = memory.summary()   # 最近几步的摘要

            router_input = {
                "current_result": result,
                "history": memory_text,
                "visual_prior": rag_visual_context,
                "historical_strategies": strategy_cases
            }
            print(router_input)

            router_thinking, router_answer = understander.run(
                sys_prompt = router_prompt_rag,
                user_prompt = json.dumps(router_input, ensure_ascii=False)
            )

            st.session_state.logs += [
                ("sys",f"思考: {router_thinking}"),
                ("sys",f"下一步: {router_answer}")
            ]
            with log_box.container(): 
                render_chat(st.session_state.logs)

            # 解析模型输出
            router_answer = json.loads(router_answer)
            tool = router_answer["tool"]

            # 如果模型认为流程应该终止
            if tool == "Terminate":
                st.session_state.logs.append(
                    ("sys",f"流程中止，原因: {router_answer.get('parameters',{}).get('reason','无')}")
                )
                render_history()
                break
            elif tool == "Pass":
                st.session_state.logs.append(
                    ("sys",f"通过，流程中止。")
                )
                render_history()
                break
            else:
                # 否则，准备下一步工具调用参数
                params=router_answer.get("parameters",{})
                if params.get("img") == "IMG": 
                    params["img"] = IMG
                if params.get("class_name") == "task_object":
                    params["class_name"] = task_object
                if params.get("mask") == "MASK":
                    params["mask"] = mask

            attempt += 1
            time.sleep(0.1)

        # ——————————————————————————— 回退机制 ———————————————————————————
        if tool == "Terminate" or attempt == MAX_RETRY + 1:
            st.session_state.logs.append(
                ("sys", f"未在 {MAX_RETRY} 轮内通过，回退到历史最佳结果，评分={best_score}")
            )
            mask = best_mask

        # ——————————————————————————— 最终输出 ———————————————————————————
        st.session_state.final_mask = mask
        render_history()

        st.session_state.logs.append(("sys", "流程结束，输出最终 Mask"))
        with log_box.container():
            render_chat(st.session_state.logs)

        # ——————————————————————————— 写入知识库 ———————————————————————————
        if best_score >= RAG_WRITE_THRESHOLD and tool == "Pass":
            strategy_summary = summarize_strategy(memory)

            writer.append({
                "object": task_object,
                "visual_tags": rag_hits[0]["tags"] if rag_hits else [],
                "image_meta": {
                    "resolution": f"{IMG.shape[1]}x{IMG.shape[0]}"
                },
                "final_score": result,
                "strategy_summary": strategy_summary,
                "confidence": float(best_score)
            })

            st.session_state.logs.append(("sys", "RAG：已将成功策略写入知识库"))
            with log_box.container():
                render_chat(st.session_state.logs)

        st.session_state.running=False
