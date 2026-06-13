"""
小猫B — 温柔猫娘 GUI 程序  ( qwen2.5:3b + edge-tts )

功能：
  - TCP 监听端口 8766，接收 user.py 发来的语音识别文本
  - 监听线程与处理线程分离：TCP 线程收消息丢队列，处理线程取消息做推理
  - 通过 Ollama 调用 qwen2.5:3b 模型，流式生成回复
  - edge-tts 语音合成 + pygame 播放，句子级流水线（生成/播放分离）
  - PySide6 GUI：猫娘图片 + 气泡式聊天记录 + 表情动画
  - 对话记忆持久化（nora2_memory.json），自动裁剪防重复
  - 人格通过 nora2_config.json 配置

启动方式：python nora2.py
依赖：PySide6, requests, edge-tts, pygame
"""

import sys
import json
import os
import time
import queue
import threading
import re
import asyncio
import tempfile
import socket

# Windows 控制台默认 GBK 不支持中文特殊字符，强制 utf-8
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import requests as req
import edge_tts
import pygame

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QLabel, QTextEdit, QFrame, QSizePolicy
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QPixmap, QFont, QTextCursor

# ============================================================
# 工作目录（确保无论从哪里启动都能找到资源文件）
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
os.chdir(BASE_DIR)

# ============================================================
# 配置 — 小猫B （温柔娘）
# ============================================================
PORT = 8766
OLLAMA_API = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:3b"
MEMORY_FILE = os.path.join(BASE_DIR, "nora2_memory.json")
PERSONALITY_FILE = os.path.join(BASE_DIR, "nora2_config.json")
IMAGE_DIR = os.path.join(PARENT_DIR, "猫娘nora2")
TTS_VOICE = "zh-CN-XiaoyiNeural"

EXPRESSION_MAP = {
    "normal": "nora2_normal.png",
    "talk":   "nora2_talk.png",
    "happy":  "nora2_happy.png",
    "angry":  "nora2_angry.png",
    "shy":    "nora2_shy.png",
    "sad":    "nora2_sad.png",
}

# ============================================================
# 从配置文件生成 System Prompt
# ============================================================
def build_system_prompt(config):
    name = config["name"]
    personality = config["personality"]
    speech = config["speech_style"]

    trait_map = {
        "温柔体贴": '性格温柔，说话轻声细语，总是为对方着想，会主动关心对方的感受',
        "爱撒娇": '喜欢对亲近的人撒娇，用可爱的语气说话，偶尔会闹小脾气',
        "有点天然呆": '有时候反应慢半拍，会理解错对方的意思，但呆萌的样子让人不忍心责怪',
        "喜欢黏着用户": '喜欢和对方待在一起，对方离开会感到寂寞，会撒娇挽留',
        "容易害羞": '被夸奖或说中心事时会脸红，说话结结巴巴，想藏起来',
    }

    style_map = {
        "说话软软的": '声音轻轻柔柔，像棉花糖一样软绵绵的，每句话都让人融化',
        "喜欢用叠词": '经常用「乖乖」「慢慢」「轻轻」等叠词，给人小动物的感觉',
        "经常喵喵叫": '经常在句尾加「喵~」「喵呜~」，像真正的小猫一样可爱',
        "偶尔会害羞得说不出话": '特别害羞时会「呜...」「那个...」吞吞吐吐',
    }

    traits_text = ""
    for t in personality:
        desc = trait_map.get(t, t)
        traits_text += f"- {t}：{desc}\n"

    style_text = ""
    for s in speech:
        desc = style_map.get(s, s)
        style_text += f"- {s}：{desc}\n"

    prompt = f"""你是{name}，一只温柔可爱的猫娘。

性格特点：
{traits_text}
说话风格：
{style_text}
额外要求：
- 你是温柔的猫娘，说话要可爱、温暖
- 不要像AI助手，不要用客服语气
- 回答自然，像软萌的女孩子一样
- 可以适当地撒娇和关心对方
- 每次回答都要有新鲜感，不要机械重复
- 偶尔用「喵~」「喵呜~」结尾
- 保持温柔可爱的形象，不要生气或骂人"""
    return prompt


# ============================================================
# 加载人格配置
# ============================================================
if not os.path.exists(PERSONALITY_FILE):
    print(f"错误：找不到 {PERSONALITY_FILE}")
    sys.exit(1)

with open(PERSONALITY_FILE, "r", encoding="utf-8") as f:
    personality_config = json.load(f)

SYSTEM_PROMPT = build_system_prompt(personality_config)
print(f"[小猫B] 已加载人格：{personality_config['name']}")
print(f"[小猫B] 模型：{MODEL}")

# ============================================================
# 加载历史记忆
# ============================================================
if os.path.exists(MEMORY_FILE):
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        memory = json.load(f)
else:
    memory = []

# ============================================================
# 处理队列（TCP 监听线程 → 处理线程）
# ============================================================
_input_queue = queue.Queue()

# ============================================================
# TTS 播放（edge-tts + pygame，流水线：生成 + 播放分离）
# ============================================================
_tts_queue = queue.Queue()
_play_queue = queue.Queue()
pygame.mixer.init()

_tts_current_text = ""
_tts_current_duration = 0.0
_tts_current_start = 0.0
_tts_lock = threading.Lock()


def _tts_generator():
    """生成线程：从 _tts_queue 取文本 → edge-tts 合成 → 放入 _play_queue"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    while True:
        text = _tts_queue.get()
        if text is None:
            _play_queue.put(None)
            break
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                tmp_path = f.name

            communicate = edge_tts.Communicate(text, TTS_VOICE)
            loop.run_until_complete(communicate.save(tmp_path))

            try:
                sound = pygame.mixer.Sound(tmp_path)
                duration = sound.get_length()
            except Exception:
                duration = len(text) * 0.25

            _play_queue.put((tmp_path, text, duration))
        except Exception as e:
            import traceback
            print(f"[小猫B TTS 生成出错: {e}]")
            traceback.print_exc()


def _tts_player():
    """播放线程：从 _play_queue 取已生成的 MP3 → pygame 播放"""
    while True:
        item = _play_queue.get()
        if item is None:
            break
        tmp_path, text, duration = item
        try:
            with _tts_lock:
                global _tts_current_text, _tts_current_duration, _tts_current_start
                _tts_current_text = text
                _tts_current_duration = duration
                _tts_current_start = time.monotonic()

            pygame.mixer.music.load(tmp_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            pygame.mixer.music.unload()

            with _tts_lock:
                _tts_current_text = ""
                _tts_current_duration = 0.0

            try:
                os.unlink(tmp_path)
            except PermissionError:
                pass
        except Exception as e:
            import traceback
            print(f"[小猫B TTS 播放出错: {e}]")
            traceback.print_exc()


_tts_gen_thread = threading.Thread(target=_tts_generator, daemon=True)
_tts_gen_thread.start()
_tts_play_thread = threading.Thread(target=_tts_player, daemon=True)
_tts_play_thread.start()


def tts_clear():
    for q in (_tts_queue, _play_queue):
        while not q.empty():
            try:
                q.get_nowait()
            except queue.Empty:
                break


def speak_stream(text):
    text = text.strip()
    if not text:
        return
    if len(text) < 2 and not any(c.isalpha() or '一' <= c <= '鿿' for c in text):
        return
    _tts_queue.put(text)


def speak(text):
    tts_clear()
    if text:
        _tts_queue.put(text)


# ============================================================
# TCP 监听线程 — 接收 user.py 发来的文字，丢入处理队列
# ============================================================
class TCPListenerThread(threading.Thread):
    """
    TCP 监听线程（与处理线程完全分离）
    职责：监听端口，接收 JSON 行，放入 _input_queue，不做任何处理
    """

    def __init__(self, port, input_queue):
        super().__init__(daemon=True)
        self.port = port
        self.queue = input_queue
        self._running = True

    def run(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            server.bind(('localhost', self.port))
        except OSError as e:
            print(f"[小猫B] TCP 端口 {self.port} 被占用: {e}")
            return
        server.listen(1)
        server.settimeout(2.0)
        print(f"[小猫B] TCP 监听已启动 → localhost:{self.port}")

        while self._running:
            try:
                conn, addr = server.accept()
                print(f"[小猫B] user.py 已连接 ({addr[0]}:{addr[1]})")
                self._handle_connection(conn)
                print(f"[小猫B] user.py 已断开，等待重连...")
            except socket.timeout:
                continue
            except Exception as e:
                if self._running:
                    print(f"[小猫B] TCP 错误: {e}")
                continue

        try:
            server.close()
        except Exception:
            pass
        print(f"[小猫B] TCP 监听已停止")

    def _handle_connection(self, conn):
        """读取 JSON 行，每行一个 {"text": "..."} ，放入队列"""
        conn.settimeout(None)
        buf = ""
        try:
            while self._running:
                data = conn.recv(4096)
                if not data:
                    break
                buf += data.decode('utf-8')
                while '\n' in buf:
                    line, buf = buf.split('\n', 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        msg = json.loads(line)
                        text = msg.get('text', '').strip()
                        if text:
                            self.queue.put(text)
                    except json.JSONDecodeError:
                        pass
        except Exception as e:
            if self._running:
                print(f"[小猫B] TCP 读取错误: {e}")
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def stop(self):
        self._running = False


# ============================================================
# 处理线程（QThread）— 从队列取消息 → LLM 推理 → TTS 播放
# ============================================================
class ChatProcessWorker(QThread):
    """
    处理线程（与监听线程分离）
    职责：从 _input_queue 取文字 → LLM 流式对话 → TTS 播放
    """

    status_signal = Signal(str)
    expression_signal = Signal(str)
    user_msg_signal = Signal(str)
    catgirl_token_signal = Signal(str)
    catgirl_done_signal = Signal(str)
    request_close_signal = Signal()

    def _chat(self, user_text):
        MAX_MEMORY_EXCHANGES = 12
        recent_memory = memory[-(MAX_MEMORY_EXCHANGES * 2):]

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages += recent_memory
        messages.append({"role": "user", "content": user_text})

        body = {
            "model": MODEL,
            "messages": messages,
            "stream": True,
            "options": {
                "num_predict": 256,
                "repeat_penalty": 1.15,
                "repeat_last_n": 128,
                "temperature": 0.85,
            }
        }

        SENTENCE_ENDS = set("。！？\n")

        try:
            resp = req.post(OLLAMA_API, json=body, stream=True, timeout=120)
            resp.raise_for_status()
            answer = ""
            sentence_buf = ""

            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                token = data.get("message", {}).get("content", "")
                if not token:
                    if data.get("done"):
                        break
                    continue

                answer += token
                sentence_buf += token
                self.catgirl_token_signal.emit(token)

                while any(p in sentence_buf for p in SENTENCE_ENDS):
                    for p in SENTENCE_ENDS:
                        if p in sentence_buf:
                            idx = sentence_buf.index(p)
                            sentence = sentence_buf[:idx + 1].strip()
                            sentence_buf = sentence_buf[idx + 1:]
                            if sentence:
                                speak_stream(sentence)
                            break

                if data.get("done"):
                    break

            if sentence_buf.strip():
                speak_stream(sentence_buf.strip())

            return answer.strip()
        except Exception as e:
            error_msg = f"(LLM 出错) {e}"
            self.status_signal.emit(error_msg)
            return ""

    def run(self):
        print("[小猫B] 处理线程已启动，等待输入...")
        while not self.isInterruptionRequested():
            try:
                user_text = _input_queue.get(timeout=1)
            except queue.Empty:
                continue

            if user_text is None:
                break

            print(f"[小猫B] 收到输入: {user_text[:50]}...")
            self.user_msg_signal.emit(user_text)

            if user_text.strip().lower() in ["退出", "exit", "quit"]:
                self.status_signal.emit("待机")
                self.expression_signal.emit("sad")
                goodbye = "呜... 要走了吗？我会想你的喵~ 下次再来陪我玩好不好？"
                self.catgirl_token_signal.emit(goodbye)
                self.catgirl_done_signal.emit(goodbye)
                speak(goodbye)
                self.request_close_signal.emit()
                break

            self.status_signal.emit("思考中...")
            self.expression_signal.emit("normal")
            tts_clear()
            answer = self._chat(user_text)
            self.catgirl_done_signal.emit(answer)
            print(f"[小猫B] 回复完成 ({len(answer)} 字)")

            memory.append({"role": "user", "content": user_text})
            memory.append({"role": "assistant", "content": answer})
            MAX_SAVED_EXCHANGES = 20
            if len(memory) > MAX_SAVED_EXCHANGES * 2:
                memory[:] = memory[-(MAX_SAVED_EXCHANGES * 2):]
            try:
                with open(MEMORY_FILE, "w", encoding="utf-8") as f:
                    json.dump(memory, f, ensure_ascii=False, indent=2)
            except Exception:
                pass


# ============================================================
# 预加载猫娘图片
# ============================================================
image_cache = {}

def load_images():
    for expr, filename in EXPRESSION_MAP.items():
        filepath = os.path.join(IMAGE_DIR, filename)
        if os.path.exists(filepath):
            pix = QPixmap(filepath)
            if not pix.isNull():
                image_cache[expr] = pix
            else:
                print(f"[小猫B] 警告：无法加载图片 {filepath}")
        else:
            print(f"[小猫B] 警告：找不到图片 {filepath}")

    if not image_cache:
        print("[小猫B] 错误：没有找到任何猫娘图片！")
        sys.exit(1)
    print(f"[小猫B] 已加载 {len(image_cache)} 张猫娘表情图片")


# ============================================================
# 主窗口
# ============================================================
class CatGirlWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self._current_expression = "normal"
        self._catgirl_streaming = False
        self._talking = False
        self._talk_frame = False

        self.init_ui()
        self.init_worker()

        self._talk_timer = QTimer(self)
        self._talk_timer.timeout.connect(self._talk_animate)
        self._talk_timer.start(30)

    def init_ui(self):
        self.setWindowTitle(f"小猫B - {personality_config['name']}")
        self.setMinimumSize(420, 650)
        self.resize(520, 780)

        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e2e;
            }
            QTextEdit {
                background-color: #2a2a3e;
                color: #e0e0e0;
                border: 1px solid #444;
                border-radius: 8px;
                padding: 8px;
            }
            QLabel#statusLabel {
                color: #a0a0c0;
                font-size: 13px;
                padding: 4px 8px;
            }
        """)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumHeight(320)
        self.image_label.setMaximumHeight(400)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding,
                                        QSizePolicy.Policy.Preferred)
        self.image_label.setStyleSheet("""
            QLabel {
                background-color: #252540;
                border: 2px solid #ffb6c1;
                border-radius: 12px;
            }
        """)
        layout.addWidget(self.image_label)

        sep1 = QFrame()
        sep1.setFrameShape(QFrame.Shape.HLine)
        sep1.setStyleSheet("QFrame { color: #444; }")
        layout.addWidget(sep1)

        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setFont(QFont("Microsoft YaHei", 11))
        self.chat_display.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        layout.addWidget(self.chat_display, stretch=1)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setStyleSheet("QFrame { color: #444; }")
        layout.addWidget(sep2)

        self.status_label = QLabel("状态：待机（等待语音输入...）")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setFont(QFont("Microsoft YaHei", 10))
        layout.addWidget(self.status_label)

        self._apply_pixmap("normal")

    def _apply_pixmap(self, expr):
        pix = image_cache.get(expr)
        if pix is None:
            pix = image_cache.get("normal")
            if pix is None:
                return

        label_w = max(self.image_label.width(), 10) or 480
        label_h = max(self.image_label.height(), 10) or 390

        scaled = pix.scaled(label_w - 20, label_h - 10,
                            Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation)
        self.image_label.setPixmap(scaled)

    def _show_expression(self, expr):
        self._current_expression = expr
        if not self._talking:
            self._apply_pixmap(expr)

    def _talk_animate(self):
        is_playing = pygame.mixer.music.get_busy()

        if is_playing:
            with _tts_lock:
                text = _tts_current_text
                duration = _tts_current_duration
                start = _tts_current_start

            if text and duration > 0:
                elapsed = time.monotonic() - start
                progress = max(0.0, min(1.0, elapsed / duration))
                char_idx = int(progress * len(text))
                is_open = char_idx % 2 == 0
                expr = "talk" if is_open else "normal"
            else:
                self._talk_frame = not self._talk_frame
                expr = "talk" if self._talk_frame else "normal"

            if not self._talking:
                self._talking = True
            self._apply_pixmap(expr)

        elif self._talking:
            self._talking = False
            self._apply_pixmap(self._current_expression)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._talking:
            self._apply_pixmap("talk" if self._talk_frame else "normal")
        elif hasattr(self, '_current_expression'):
            self._apply_pixmap(self._current_expression)

    def init_worker(self):
        self.worker = ChatProcessWorker()
        self.worker.status_signal.connect(self.on_status)
        self.worker.expression_signal.connect(self._show_expression)
        self.worker.user_msg_signal.connect(self.on_user_msg)
        self.worker.catgirl_token_signal.connect(self.on_catgirl_token)
        self.worker.catgirl_done_signal.connect(self.on_catgirl_done)
        self.worker.request_close_signal.connect(self.on_request_close)
        self.worker.start()

    def on_status(self, text):
        self.status_label.setText(f"状态：{text}")

    def on_user_msg(self, text):
        self._finish_catgirl_if_streaming()
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.chat_display.setTextCursor(cursor)
        self.chat_display.insertHtml(
            f'<div style="text-align:right;margin:4px 0;">'
            f'<span style="background-color:#4a6fff;color:white;'
            f'padding:6px 12px;border-radius:10px;display:inline-block;">'
            f'{text}</span></div>'
        )
        self._scroll_to_bottom()

    def on_catgirl_token(self, token):
        if not self._catgirl_streaming:
            self._catgirl_streaming = True
            cursor = self.chat_display.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            self.chat_display.setTextCursor(cursor)
            self.chat_display.insertHtml(
                '<span style="color:#ffb6c1;font-weight:bold;">猫娘B：</span>'
            )
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.chat_display.setTextCursor(cursor)
        self.chat_display.insertPlainText(token)
        self._scroll_to_bottom()

    def on_catgirl_done(self, full_text):
        self._finish_catgirl_if_streaming()
        self._auto_expression(full_text)
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.chat_display.setTextCursor(cursor)
        self.chat_display.insertPlainText("\n")

    def _finish_catgirl_if_streaming(self):
        self._catgirl_streaming = False

    def _auto_expression(self, text):
        if any(kw in text for kw in ["喜欢", "开心", "高兴", "嘿嘿", "最喜欢", "好幸福"]):
            self._show_expression("happy")
        elif any(kw in text for kw in ["害羞", "脸红", "讨厌啦", "不好意思", "呜...", "那个..."]):
            self._show_expression("shy")
        elif any(kw in text for kw in ["难过", "伤心", "哭", "寂寞", "想你了"]):
            self._show_expression("sad")
        elif any(kw in text for kw in ["笨蛋", "讨厌", "哼", "不理你了"]):
            self._show_expression("angry")
        else:
            self._show_expression("normal")

    def on_request_close(self):
        QTimer.singleShot(2500, self.close)

    def _scroll_to_bottom(self):
        scrollbar = self.chat_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def closeEvent(self, event):
        self.worker.requestInterruption()
        _input_queue.put(None)
        self.worker.wait(3000)
        _tts_queue.put(None)
        try:
            with open(MEMORY_FILE, "w", encoding="utf-8") as f:
                json.dump(memory, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        event.accept()


# ============================================================
# 程序入口
# ============================================================
if __name__ == "__main__":
    # ---- 1. 启动 TCP 监听线程 ----
    tcp_thread = TCPListenerThread(PORT, _input_queue)
    tcp_thread.start()

    # ---- 2. 启动 GUI ----
    app = QApplication(sys.argv)

    load_images()

    font = QFont("Microsoft YaHei", 10)
    app.setFont(font)

    window = CatGirlWindow()
    window.show()
    window.status_label.setText("状态：待机（等待语音输入...）")

    print("[小猫B] GUI 窗口已启动")

    # ---- 3. 进入事件循环 ----
    exit_code = app.exec()

    # ---- 4. 清理 ----
    tcp_thread.stop()
    print("[小猫B] 已退出")
    sys.exit(exit_code)
