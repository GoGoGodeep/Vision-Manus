# rag/vision_rag.py
import json
import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class VisualConcept:
    object: str
    aliases: List[str]
    tags: List[str]
    prior: str
    failure_modes: List[str]
    suggestions: List[str]

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "VisualConcept":
        return VisualConcept(
            object=d.get("object", ""),
            aliases=d.get("aliases", []) or [],
            tags=d.get("tags", []) or [],
            prior=d.get("prior", ""),
            failure_modes=d.get("failure_modes", []) or [],
            suggestions=d.get("suggestions", []) or [],
        )

    def to_prompt_block(self) -> str:
        # 控制长度，避免 prompt 过长
        fm = "；".join(self.failure_modes[:5]) if self.failure_modes else "无"
        sg = "；".join(self.suggestions[:5]) if self.suggestions else "无"
        al = "、".join(self.aliases[:6]) if self.aliases else "无"
        tg = "、".join(self.tags[:8]) if self.tags else "无"

        return (
            f"[Visual Concept]\n"
            f"- object: {self.object}\n"
            f"- aliases: {al}\n"
            f"- tags: {tg}\n"
            f"- prior: {self.prior}\n"
            f"- common failure modes: {fm}\n"
            f"- actionable suggestions: {sg}\n"
        )


class VisionRAG:
    """
    轻量级 RAG：TF-IDF + cosine 相似度，用于“任务对象视觉知识增强”.
    """

    def __init__(
            self, 
            visual_db_path: str,
            strategy_db_path: str
        ):
        self.strategy_db_path = strategy_db_path
        self.visual_db_path = visual_db_path

        self.strategy_items: List[Dict[str, Any]] = []
        self.items: List[VisualConcept] = []
        self.doc_matrix = None
        
        self._load_and_build()
        self._load_strategy_cases()

    def _load_and_build(self) -> None:
        if not os.path.exists(self.visual_db_path):
            raise FileNotFoundError(f"visual concept db not found: {self.visual_db_path}")

        items: List[VisualConcept] = []
        texts: List[str] = []

        with open(self.visual_db_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                vc = VisualConcept.from_dict(d)
                items.append(vc)

                # 构造可检索文本：object + aliases + tags + prior + failure_modes + suggestions
                text = " ".join(
                    [vc.object]
                    + vc.aliases
                    + vc.tags
                    + [vc.prior]
                    + vc.failure_modes
                    + vc.suggestions
                )
                texts.append(text)

        self.items = items
        self.vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            min_df=1
        )
        self.doc_matrix = self.vectorizer.fit_transform(texts)

    def _load_strategy_cases(self) -> None:
        items = []
        texts = []

        with open(self.strategy_db_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                items.append(d)

                ss = d.get("strategy_summary", {})
                text = " ".join(
                    [d.get("object", "")]
                    + ss.get("path", [])
                    + ss.get("key_decisions", [])
                )
                texts.append(text)

        if not items:
            self.strategy_items = []
            self.strategy_vectorizer = None
            self.strategy_doc_matrix = None
            return

        self.strategy_items = items
        self.strategy_vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            min_df=1
        )
        self.strategy_doc_matrix = self.strategy_vectorizer.fit_transform(texts)

    def retrieve_visual_concept(
        self,
        task_object: str,
        top_k: int = 2,
        min_score: float = 0.10
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        返回：
          - prompt_context: 适合直接注入 LLM 的文本块
          - debug_hits: 命中详情（用于日志/调试）
        """
        if not task_object or self.vectorizer is None or self.doc_matrix is None:
            return "[Visual Concept]\n- No prior available.\n", []

        q = task_object.strip()
        q_vec = self.vectorizer.transform([q])
        sims = cosine_similarity(q_vec, self.doc_matrix).reshape(-1)

        # 取 top_k
        idxs = np.argsort(-sims)[:top_k]
        hits = []
        blocks = []

        for idx in idxs:
            score = float(sims[idx])
            if score < min_score:
                continue
            vc = self.items[idx]
            blocks.append(vc.to_prompt_block())
            hits.append({
                "object": vc.object,
                "score": score,
                "aliases": vc.aliases,
                "tags": vc.tags
            })

        if not blocks:
            return (
                "[Visual Concept]\n"
                f"- object: {task_object}\n"
                "- No matched prior in database.\n"
                "- suggestions: try add a new concept entry to visual_concepts.jsonl.\n",
                []
            )

        # 合并为一个 context（可控长度）
        prompt_context = "\n".join(blocks[:top_k])
        return prompt_context, hits
    
    def retrieve_strategy_cases(
        self,
        task_object: str,
        top_k: int = 2,
        min_score: float = 0.15,
        min_confidence: float = 0.6
    ) -> Tuple[str, List[Dict[str, Any]]]:

        # ===== 防御：策略库不可用 =====
        if (
            not task_object
            or self.strategy_vectorizer is None
            or self.strategy_doc_matrix is None
            or not self.strategy_items
        ):
            return (
                "[Historical Strategies]\n"
                f"- object: {task_object}\n"
                "- No strategy database available.\n",
                []
            )

        q = task_object.strip()
        q_vec = self.strategy_vectorizer.transform([q])
        sims = cosine_similarity(q_vec, self.strategy_doc_matrix).reshape(-1)

        # 取 top_k
        idxs = np.argsort(-sims)[:top_k]
        hits = []
        blocks = []

        for idx in idxs:
            sim = float(sims[idx])
            if sim < min_score:
                continue

            case = self.strategy_items[idx]
            if case.get("confidence", 0) < min_confidence:
                continue

            ss = case.get("strategy_summary", {})
            fs = case.get("final_score", {})

            blocks.append(
                "[Historical Strategy]\n"
                f"- object: {case.get('object')}\n"
                f"- rounds: {ss.get('rounds')}\n"
                f"- strategy path: {' -> '.join(ss.get('path', []))}\n"
                f"- key decisions: {'；'.join(ss.get('key_decisions', []))}\n"
                f"- final score: {fs}\n"
            )

            hits.append({
                "object": case.get("object"),
                "similarity": sim,
                "confidence": case.get("confidence"),
                "rounds": ss.get("rounds"),
                "path": ss.get("path")
            })

            if len(blocks) >= top_k:
                break

        if not blocks:
            return (
                "[Historical Strategies]\n"
                f"- object: {task_object}\n"
                "- No high-confidence matched strategies.\n",
                []
            )

        return "\n".join(blocks), hits