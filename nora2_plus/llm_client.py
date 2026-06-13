"""
LLMClient —— Ollama API 客户端 + JSON 响应解析
LLMExtractor —— 长期记忆提取器
"""
import json
import re
import threading
import requests as req


# ============================================================
# JSON 解析辅助函数（模块级，供 LLMClient 静态方法使用）
# ============================================================

def _extract_json_braces(text):
    """括号计数法提取最外层 JSON 对象。返回 (json_str, start, end) 或 (None, -1, -1)"""
    start = text.find('{')
    if start == -1:
        return None, -1, -1

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        ch = text[i]

        if escape:
            escape = False
            continue

        if ch == '\\' and in_string:
            escape = True
            continue

        if ch == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return text[start:i + 1], start, i + 1

    return None, -1, -1


def _strip_markdown_fences(text):
    """剥离 markdown 代码块标记 ```json ... ``` 或 ``` ... ```"""
    text = text.strip()
    m = re.match(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text


def _repair_json(json_str):
    """尝试修复常见 JSON 格式错误"""
    json_str = re.sub(r':\s*\+(\d)', r': \1', json_str)
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*]', ']', json_str)
    return json_str


# ============================================================
# LLMClient
# ============================================================

class LLMClient:
    """Ollama API 客户端"""

    # 支持的情绪列表
    VALID_EMOTIONS = {"normal", "happy", "angry", "sad", "shy"}

    def __init__(self, api_url="http://localhost:11434/api/chat", model="llama3.1:8b"):
        self.api_url = api_url
        self.model = model

    def chat(self, messages, stream=True, **options):
        """
        发送聊天请求，返回完整响应文本。
        options 可传入 num_predict, temperature, repeat_penalty, repeat_last_n 等。
        """
        body = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "format": "json",
            "options": options,
        }

        resp = req.post(self.api_url, json=body, stream=True)
        resp.raise_for_status()
        full_response = ""

        for line in resp.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            token = data.get("message", {}).get("content", "")
            if not token:
                if data.get("done"):
                    break
                continue
            full_response += token
            if data.get("done"):
                break

        return full_response.strip()

    def chat_non_stream(self, messages, **options):
        """非流式聊天请求，返回完整响应文本"""
        body = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": options,
        }

        resp = req.post(self.api_url, json=body, timeout=120)
        resp.raise_for_status()
        result = resp.json()
        return result.get("message", {}).get("content", "").strip()

    def parse_response(self, raw_text):
        """
        多策略解析 LLM JSON 响应 → {emotion, favorability_change, reply}
        失败时降级返回 safe defaults。
        """
        if not raw_text:
            return self._fallback("(猫娘沉默了...)")

        text = raw_text.strip()

        # 预处理：剥离 markdown 代码块
        text = _strip_markdown_fences(text)

        # 策略 1: 直接 json.loads
        data = self._try_parse_json(text)
        if data:
            return data

        # 策略 2: 括号计数法提取 JSON
        json_str, _, _ = _extract_json_braces(text)
        if json_str:
            # 2a: 直接解析
            data = self._try_parse_json(json_str)
            if data:
                return data

            # 2b: 修复后解析
            try:
                repaired = _repair_json(json_str)
                data = self._try_parse_json(repaired)
                if data:
                    return data
            except Exception:
                pass

            # 2c: 单引号替换
            try:
                single_quoted = json_str.replace("'", '"')
                if single_quoted != json_str:
                    data = self._try_parse_json(single_quoted)
                    if data:
                        return data
            except Exception:
                pass

        # 降级：尝试正则提取 reply 字段
        reply_only = re.search(r'"reply"\s*:\s*"(.+?)"\s*\}?\s*$', text, re.DOTALL)
        if reply_only:
            reply_text = reply_only.group(1)
            try:
                reply_text = json.loads(f'"{reply_text}"')
            except json.JSONDecodeError:
                pass
            print(f"[警告] JSON 解析失败，正则提取到 reply: {reply_text[:80]}...")
            return {"emotion": "normal", "favorability_change": 0, "reply": reply_text}

        # 最终降级：原始文本作为 reply
        print(f"[警告] JSON 解析全部失败，返回原始文本。长度={len(raw_text)}")
        return self._fallback(text)

    def _try_parse_json(self, text):
        """尝试 json.loads + 验证 + 规范化"""
        try:
            data = json.loads(text)
            if not isinstance(data, dict) or "reply" not in data:
                return None
            return self._normalize(data)
        except json.JSONDecodeError:
            return None

    def _normalize(self, data):
        """规范化响应"""
        emotion = data.get("emotion", "normal")
        if emotion not in self.VALID_EMOTIONS:
            emotion = "normal"

        try:
            fav_change = int(data.get("favorability_change", 0))
        except (ValueError, TypeError):
            fav_change = 0
        fav_change = max(-10, min(10, fav_change))

        reply = str(data.get("reply", ""))
        if not reply:
            reply = "..."

        return {"emotion": emotion, "favorability_change": fav_change, "reply": reply}

    def _fallback(self, reply):
        return {"emotion": "normal", "favorability_change": 0, "reply": reply}


# ============================================================
# LLMExtractor —— 长期记忆提取
# ============================================================

class LLMExtractor:
    """从对话记录中提取长期记忆"""

    def __init__(self, client: LLMClient, extraction_interval=20):
        self.client = client
        self.extraction_interval = extraction_interval

    def extract(self, recent_messages: list) -> list:
        """
        调用 LLM 从最近对话提取长期有效信息。
        返回 [{"content": "...", "importance": 5}, ...]
        """
        conv_lines = []
        for msg in recent_messages:
            role = "用户" if msg["role"] == "user" else "猫娘"
            conv_lines.append(f"{role}：{msg['content']}")
        conv_text = "\n".join(conv_lines)

        prompt = f"""分析以下聊天记录，提取所有关于【用户】和【猫娘】的长期有效信息。

逐条排查以下类别，列出所有有价值的信息：
- 兴趣爱好（喜欢什么电影/音乐/食物/运动等）
- 长期习惯（作息、工作方式、日常活动）
- 项目计划（正在做的工作、学习目标、未来安排）
- 身份背景（职业、技能、所学专业、所在城市）
- 个人偏好（口味、风格、审美、喜欢/不喜欢什么）

规则：
- 用户和猫娘的信息都要提取，每条标明是谁（如"用户喜欢..."或"猫娘喜欢..."）
- 只提取有实质信息的条目，跳过纯闲聊和情绪表达
- 不要遗漏，有多少条就列多少条

聊天记录：
{conv_text}

请用以下格式逐条列出（每条一行）：
用户喜欢xxx | 重要性:7
猫娘喜欢xxx | 重要性:6

如果没有值得长期记忆的信息，回复"无"。"""

        try:
            raw_content = self.client.chat_non_stream(
                [{"role": "user", "content": prompt}],
                num_predict=2048,
                temperature=0.3,
            )
        except Exception as e:
            print(f"[长期记忆提取] LLM 调用失败: {e}")
            return []

        if not raw_content or raw_content == "无":
            print("[长期记忆提取] 没有值得长期记忆的信息")
            return []

        return self._parse_text_response(raw_content)

    def _parse_text_response(self, raw_content):
        """解析文本格式响应：内容 | 重要性:N"""
        new_items = []
        for line in raw_content.split("\n"):
            line = line.strip()
            if not line or line == "无":
                continue
            match = re.match(r'(.+?)\s*\|\s*重要性\s*[:：]\s*(\d+)', line)
            if match:
                content = match.group(1).strip()
                importance = int(match.group(2))
                if content:
                    new_items.append({"content": content, "importance": importance})
            else:
                # 兼容 JSON 格式
                try:
                    item = json.loads(line)
                    if isinstance(item, dict) and "content" in item:
                        new_items.append(item)
                    elif isinstance(item, list):
                        new_items.extend(item)
                except json.JSONDecodeError:
                    pass

        if not new_items:
            print(f"[长期记忆提取] 未解析到有效条目，原始返回: {raw_content[:200]}")
        return new_items
