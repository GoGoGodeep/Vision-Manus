import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM


@st.cache_resource
def load_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype="auto",
        device_map="auto",
        # attn_implementation="flash_attention_2",
    )
    return tokenizer, model


class Planner:
    def __init__(self):
        self.model_name = "/home/kexin/hd1/zkf/Qwen3-4bit"
        self.tokenizer, self.model = load_model(self.model_name)

    def run(self, sys_prompt: str, user_prompt: str):
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=1280*8
        )

        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        # 找 </think> 的 token（151668）
        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        thinking_content = self.tokenizer.decode(
            output_ids[:index], skip_special_tokens=True
        ).strip("\n")

        content = self.tokenizer.decode(
            output_ids[index:], skip_special_tokens=True
        ).strip("\n")

        return thinking_content, content
