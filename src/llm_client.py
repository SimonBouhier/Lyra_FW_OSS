from __future__ import annotations
import json, os, requests
from typing import Dict, Any, List

class LocalLLMClient:
    def __init__(self, model: str = None, timeout: int = 120):
        self.model = model or os.getenv("LYRA_MODEL", "gpt-oss:20b")
        self.timeout = timeout
        self.base_url_v1 = os.getenv("OLLAMA_OPENAI_BASE", "http://localhost:11434/v1")
        self.base_url_native = os.getenv("OLLAMA_NATIVE_BASE", "http://localhost:11434")

    def _call_openai_compatible(self, messages: List[Dict[str,str]]) -> Dict[str, Any]:
        url = f"{self.base_url_v1}/chat/completions"
        headers = {"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY','ollama')}"}
        payload = {"model": self.model, "messages": messages, "temperature": 0.7, "stream": False}
        r = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def _call_native(self, messages: List[Dict[str,str]]) -> Dict[str, Any]:
        url = f"{self.base_url_native}/api/chat"
        payload = {"model": self.model, "messages": messages, "stream": False}
        r = requests.post(url, json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def chat(self, messages: List[Dict[str,str]]) -> str:
        try:
            j = self._call_openai_compatible(messages)
            if "choices" in j and j["choices"]:
                return j["choices"][0]["message"]["content"]
        except Exception:
            pass
        j = self._call_native(messages)
        if "message" in j and "content" in j["message"]:
            return j["message"]["content"]
        if "messages" in j and j["messages"]:
            return j["messages"][-1]["content"]
        return ""

    def propose_action(self, system_prompt: str, user_payload: Dict[str, Any]) -> Dict[str, Any]:
        messages = [
            {"role":"system","content": system_prompt},
            {"role":"user","content": json.dumps(user_payload, ensure_ascii=False)}
        ]
        text = self.chat(messages)
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        return {}
