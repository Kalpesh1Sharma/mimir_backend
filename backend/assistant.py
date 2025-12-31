import re
import ast
import operator
from typing import List, Dict

from rag.embeddings import EmbeddingModel
from rag.retrieve import Retriever
from backend.file_qa.file_qa import FileQASystem
from backend.personas import PersonaManager
from backend.web_search import WebSearchQA


class MimirAssistant:
    def __init__(self):
        self.embedder = EmbeddingModel()
        self.retriever = Retriever(self.embedder)
        self.file_qa = FileQASystem()
        self.persona_manager = PersonaManager()
        self.web_search = WebSearchQA()

        # 🔹 short-term conversational memory (last N turns)
        self.memory: List[Dict[str, str]] = []

        self.MAX_MEMORY = 8

    # =========================
    # MAIN QUERY (SINGLE INPUT)
    # =========================
    def query(self, text: str, persona="default", mode="factual"):
        text = text.strip()

        if not text:
            return {
                "answer": "No input provided.",
                "sources": [],
                "confidence": 0.2,
            }

        # store user turn
        self._add_memory("user", text)

        # math fast-path
        expr = self._extract_math_expression(text)
        if expr:
            return {
                "answer": str(self._solve_math(expr)),
                "sources": [],
                "confidence": 1.0,
            }

        # file QA
        if self.file_qa.has_files():
            result = self.file_qa.answer(text)
            self._add_memory("mimir", result["answer"])
            return result

        persona_contract = self.persona_manager.load(persona)

        # 🔹 build contextual query from memory
        contextual_query = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}"
            for m in self.memory
        )

        query_vec = self.embedder.embed(contextual_query)[0]
        results = self.retriever.retrieve(query_vec)

        if results:
            context = "\n\n".join(r["text"] for r in results)
            self._add_memory("mimir", context)
            return {
                "answer": context,
                "sources": list({r["metadata"]["source"] for r in results}),
                "confidence": 0.9,
            }

        # web fallback
        web = self.web_search.search(text)
        if web:
            self._add_memory("mimir", web["answer"])
            return web

        fallback = "The realms are silent… the connection failed."
        self._add_memory("mimir", fallback)
        return {
            "answer": fallback,
            "sources": [],
            "confidence": 0.2,
        }

    # =========================
    # MEMORY-AWARE QUERY
    # =========================
    def query_with_memory(self, messages: List[Dict], persona="default", mode="factual"):
        if not messages:
            return {
                "answer": "No input provided.",
                "confidence": 0.2,
            }

        # sync memory from frontend
        self.memory = messages[-self.MAX_MEMORY :]

        last_user_msg = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"),
            "",
        )

        # 🔮 MEMORY-INTENT DETECTION
        if self._is_memory_question(last_user_msg):
            recalled = [
                m["content"]
                for m in self.memory
                if m["role"] == "user"
            ]

            if len(recalled) <= 1:
                answer = "We have only just begun speaking."
            else:
                answer = (
                    "Earlier, you spoke of: "
                    + "; ".join(recalled[:-1])
                )

            self._add_memory("mimir", answer)
            return {
                "answer": answer,
                "confidence": 0.95,
            }

        # otherwise continue normally
        return self.query(last_user_msg, persona, mode)

    # =========================
    # MEMORY HELPERS
    # =========================
    def _add_memory(self, role: str, content: str):
        self.memory.append({"role": role, "content": content})
        self.memory = self.memory[-self.MAX_MEMORY :]

    def _is_memory_question(self, text: str) -> bool:
        triggers = [
            "remember",
            "earlier",
            "before",
            "last time",
            "previous",
            "you said",
            "we talked",
        ]
        return any(t in text.lower() for t in triggers)

    # =========================
    # MATH UTILITIES
    # =========================
    def _extract_math_expression(self, text):
        matches = re.findall(
            r"\(?\d+(?:\.\d+)?(?:\s*[\+\-\*/]\s*\(?\d+(?:\.\d+)?\)?)+",
            text,
        )
        return matches[0] if matches else ""

    def _solve_math(self, expr):
        ops = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
        }

        def eval_node(node):
            if isinstance(node, ast.Num):
                return node.n
            if isinstance(node, ast.BinOp):
                return ops[type(node.op)](
                    eval_node(node.left),
                    eval_node(node.right),
                )
            raise ValueError()

        return eval_node(ast.parse(expr, mode="eval").body)
