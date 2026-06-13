"""
VoiceChatWorker —— 对话编排线程
编排 VAD → STT → LLM → TTS 全链路，接收依赖注入。
"""
import json
import time
import queue
import threading

import requests as req

from PySide6.QtCore import QThread, Signal

from config import (
    OLLAMA_API, MODEL, MEMORY_EXTRACTION_INTERVAL, MAX_SAVED_EXCHANGES,
)


class VoiceChatWorker(QThread):
    """后台线程：VAD 录音 → STT 识别 → AI 对话 → TTS 播放"""

    # ---- 信号 ----
    status_signal = Signal(str)
    expression_signal = Signal(str)
    user_msg_signal = Signal(str)
    catgirl_token_signal = Signal(str)
    catgirl_done_signal = Signal(str)
    request_close_signal = Signal()
    favorability_signal = Signal(int)
    favorability_change_signal = Signal(int)
    emotion_signal = Signal(str)
    emotion_decay_signal = Signal()          # 情绪衰减通知
    active_chat_request_signal = Signal()    # 主动聊天请求

    def __init__(self, state, vad, stt, llm_client, extractor, tts):
        """
        Args:
            state: NoraStateManager 实例
            vad: VADRecorder 实例
            stt: STTEngine 实例
            llm_client: LLMClient 实例
            extractor: LLMExtractor 实例
            tts: TTSEngine 实例
        """
        super().__init__()
        self._state = state
        self._vad = vad
        self._stt = stt
        self._llm = llm_client
        self._extractor = extractor
        self._tts = tts

        # 主动聊天
        self._active_chat_queue = queue.Queue()
        self.last_user_input_time = time.time()

    # ================================================================
    # 主循环
    # ================================================================
    def run(self):
        while not self.isInterruptionRequested():
            # ---- 检查主动聊天触发 ----
            try:
                trigger = self._active_chat_queue.get_nowait()
                if trigger:
                    self._do_active_chat()
                    continue
            except queue.Empty:
                pass

            # ---- 1. 聆听 ----
            self.status_signal.emit("聆听中...")
            self.expression_signal.emit(self._state.current_emotion)
            audio = self._vad.record()
            if audio is None:
                continue

            # ---- 用户说话，更新时间戳 ----
            self.last_user_input_time = time.time()

            # ---- 2. 语音识别 ----
            self.status_signal.emit("识别中...")
            self.expression_signal.emit("normal")
            user_text = self._stt.transcribe(audio)
            if not user_text:
                self.status_signal.emit("(没听清，请重试)")
                continue
            self.user_msg_signal.emit(user_text)

            # ---- 3. 检查退出 ----
            if user_text.strip().lower() in ["退出", "exit", "quit"]:
                self.status_signal.emit("待机")
                self.expression_signal.emit("sad")
                goodbye = "呜...要走啦？那要早点回来哦，我会想你的喵~"
                self.catgirl_token_signal.emit(goodbye)
                self.catgirl_done_signal.emit(goodbye)
                self._tts.speak(goodbye)
                self.request_close_signal.emit()
                break

            # ---- 4. AI 对话 ----
            self.status_signal.emit("思考中...")
            self.expression_signal.emit(self._state.current_emotion)
            self._tts.clear()
            parsed = self._chat(user_text)

            reply = parsed["reply"]
            self.catgirl_token_signal.emit(reply)
            self.catgirl_done_signal.emit(reply)

            # ---- 5. TTS 播放 ----
            self._tts.speak(reply)

            # ---- 6. 更新表情 ----
            self.expression_signal.emit(parsed["emotion"])

            print(f"[对话] 用户:{user_text} → 情绪:{parsed['emotion']} "
                  f"好感变化:{parsed['favorability_change']:+d} 回复:{reply}")

    # ================================================================
    # 主动聊天
    # ================================================================
    def _do_active_chat(self):
        emotion_cn = {
            "normal": "平静", "happy": "开心", "angry": "生气",
            "sad": "伤心", "shy": "害羞",
        }

        ltm_section = self._state.get_ltm_text()

        prompt = f"""当前好感度：{self._state.favorability}/100
当前情绪：{emotion_cn.get(self._state.current_emotion, '平静')}

用户已经有一段时间没有和你说话了。

{ltm_section}你是温柔猫娘。请主动发起一句聊天，不要太长，自然一点，用可爱的语气。

必须使用JSON格式回复：
{{"emotion":"<你的情绪>","favorability_change":0,"reply":"<你的主动搭话>"}}"""

        self.status_signal.emit("主动搭话中...")
        self.expression_signal.emit(self._state.current_emotion)
        self._tts.clear()

        parsed = self._chat(prompt, is_active_chat=True)

        reply = parsed["reply"]
        self.catgirl_token_signal.emit(reply)
        self.catgirl_done_signal.emit(reply)
        self._tts.speak(reply)
        self.expression_signal.emit(parsed["emotion"])

        print(f"[主动聊天] 好感度:{self._state.favorability} "
              f"情绪:{self._state.current_emotion} → {reply}")

    # ================================================================
    # 核心对话
    # ================================================================
    def _chat(self, user_text, is_active_chat=False):
        """
        AI 对话。返回 {"emotion":..., "favorability_change":..., "reply":...}
        """
        self._state.refresh_system_prompt()

        max_exchanges = 12
        recent = self._state.get_recent_memory(max_exchanges)

        messages = [{"role": "system", "content": self._state.get_system_prompt()}]
        messages += recent
        messages.append({"role": "user", "content": user_text})

        body = {
            "model": MODEL,
            "messages": messages,
            "stream": True,
            "format": "json",
            "options": {
                "num_predict": 512,
                "repeat_penalty": 1.15,
                "repeat_last_n": 128,
                "temperature": 0.85,
            }
        }

        try:
            resp = req.post(OLLAMA_API, json=body, stream=True)
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

            full_response = full_response.strip()
            print(f"[LLM原始输出] 长度={len(full_response)}")
            print(full_response)
            print("[LLM原始输出结束]")

            # 解析 JSON 响应
            parsed = self._llm.parse_response(full_response)

            # ---- 更新状态 ----
            old_emotion = self._state.current_emotion

            if is_active_chat:
                fav_change = -1
                self._state.update_favorability(-1)
            else:
                fav_change = parsed["favorability_change"]
                self._state.update_favorability(fav_change)

            # 更新情绪
            self._state.update_emotion(parsed["emotion"])

            # 发送信号
            self.favorability_signal.emit(self._state.favorability)
            self.favorability_change_signal.emit(fav_change)
            if parsed["emotion"] != old_emotion:
                self.emotion_signal.emit(self._state.current_emotion)

            # 记录到记忆
            if not is_active_chat:
                self._state.append_memory("user", user_text)
            self._state.append_memory("assistant", parsed["reply"])

            # ---- 长期记忆提取检查 ----
            if self._state.increment_round():
                self._state.mark_extraction_done()
                recent_for_extraction = list(
                    self._state.memory[-(MEMORY_EXTRACTION_INTERVAL * 2):]
                )
                threading.Thread(
                    target=self._run_extraction,
                    args=(recent_for_extraction,),
                    daemon=True
                ).start()

            return parsed

        except Exception as e:
            error_msg = f"(出错了) {e}"
            self.status_signal.emit(error_msg)
            return {
                "emotion": "normal",
                "favorability_change": 0,
                "reply": f"喵？好像出了点问题...{e}",
            }

    def _run_extraction(self, recent_messages):
        """后台线程：执行长期记忆提取"""
        new_items = self._extractor.extract(recent_messages)
        if new_items:
            before = len(self._state.long_term_memory)
            self._state.merge_ltm(new_items)
            after = len(self._state.long_term_memory)
            print(f"[长期记忆提取] 本次提取 {len(new_items)} 条，合并后 {before}→{after} 条")
