"""
语音猫娘聊天 GUI 版 — PySide6 界面 + VAD 自动语音检测 + SenseVoice 中文识别，实时对话。

功能：
  - VAD 自动检测说话/静音，无需按键，实时语音识别
  - SenseVoice (FunASR) 中文语音识别，识别准确率远超 Whisper
  - 猫娘表情随状态 / 语音节奏自动切换（读取 猫娘nora1 文件夹，6 种表情）
  - 嘴部动画跟随音频进度逐字张合，节奏自适应
  - 气泡式聊天记录（用户蓝色右对齐 / 猫娘粉色左对齐），流式显示回复
  - 语音随文字流式播放：LLM 每输出一个句子立刻送 TTS，不等全文结束
  - TTS 流水线架构（生成线程 + 播放线程分离），消除句间停顿
  - 猫娘用文字 + 语音回复，支持多轮对话记忆
  - 记忆自动裁剪（最近 12 轮），repeat_penalty 抑制重复句式
  - 人格通过 personality.json 配置

实现方法：
  录音    → sounddevice（16kHz 单声道），RMS 能量 VAD 自动检测
  语音识别 → SenseVoice Small (FunASR)，中文识别 SOTA，GPU(CUDA) 推理
  AI 对话 → Ollama /api/chat 流式接口，repeat_penalty + 记忆裁剪防重复
  语音合成 → edge-tts（微软晓晓女声），生成/播放双线程流水线，pygame 播放
  对话记忆 → chat_memory.json 持久化（自动裁剪保留最近 20 轮）
  GUI    → PySide6，QThread 工作线程 + Signal 跨线程更新 UI

依赖库（pip install）：
  PySide6                   — GUI 界面
  sounddevice numpy         — 录音
  funasr torch              — 语音识别 SenseVoice（GPU 需 nvidia-cublas-cu12）
  requests                  — Ollama API 调用
  edge-tts pygame           — 语音合成与播放
"""

import sys
import json
import os
import time
import queue
import threading
import numpy as np
import sounddevice as sd

# Windows: 添加 nvidia pip 包的 cublas DLL 目录到搜索路径
import site
for _sp in site.getsitepackages():
    _dll_path = os.path.join(_sp, "nvidia", "cublas", "bin")
    if os.path.isdir(_dll_path):
        os.add_dll_directory(_dll_path)
        os.environ["PATH"] = _dll_path + os.pathsep + os.environ.get("PATH", "")

from funasr import AutoModel
import requests as req
import asyncio
import tempfile
import edge_tts
import pygame

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QLabel, QTextEdit, QFrame, QSizePolicy
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QPixmap, QFont, QTextCursor

# ============================================================
# 配置部分
# ============================================================
OLLAMA_API = "http://localhost:11434/api/chat"
MODEL = "llama3.1:8b"
MEMORY_FILE = "chat_memory.json"
PERSONALITY_FILE = "personality.json"
SENSEVOICE_MODEL = "iic/SenseVoiceSmall"  # SenseVoice 中文识别 SOTA
SAMPLE_RATE = 16000
IMAGE_DIR = "猫娘nora1"

# VAD 参数
VAD_THRESHOLD = 0.015
VAD_SPEECH_START_FRAMES = 8
VAD_SILENCE_FRAMES = 25
VAD_CHUNK_DURATION = 0.06
VAD_MAX_LISTEN_SEC = 30

# TTS 语音
TTS_VOICE = "zh-CN-XiaoxiaoNeural"

# 表情 → 图片文件 映射
EXPRESSION_MAP = {
    "normal": "nora_normal.png",
    "talk":   "nora_talk.png",
    "happy":  "nora_happy.png",
    "angry":  "nora_angry.png",
    "shy":    "nora_shy.png",
    "sad":    "nora_sad.png",
}

# ============================================================
# 初始化 STT（SenseVoice，FunASR 中文识别 SOTA）
# ============================================================
print("加载语音识别模型 (SenseVoiceSmall)...")
sensevoice = AutoModel(
    model="iic/SenseVoiceSmall",
    device="cuda:0",
)
print("语音识别模型加载完成\n")


# ============================================================
# 从配置文件生成 System Prompt
# ============================================================
def build_system_prompt(config):
    name = config["name"]
    personality = config["personality"]
    speech = config["speech_style"]

    trait_map = {
        "傲娇": '嘴硬心软，表面上对人不耐烦，但暗地里关心对方，被戳穿时会脸红否认',
        "嘴硬": '绝不轻易承认自己的真实想法，尤其是关心或喜欢对方时',
        "喜欢吐槽": '看到用户的槽点会毫不留情地吐槽，但吐槽里藏着关心',
        "讨厌说教": '听到大道理会烦躁，会说「别跟我说教」「烦死了」之类的话',
        "实际上很关心用户": '嘴上骂骂咧咧，但行动上会默默照顾对方，只是死活不承认',
    }

    style_map = {
        "经常说笨蛋": '口头禅是 「傻叉」，几乎每句话都要带',
        "偶尔说蠢货": '用户犯明显错误时会骂「傻B」 ',
        "不使用敬语": '说话不用「您」，不用「请」，不用「谢谢」，语气粗鲁',
    }

    traits_text = ""
    for t in personality:
        desc = trait_map.get(t, t)
        traits_text += f"- {t}：{desc}\n"

    style_text = ""
    for s in speech:
        desc = style_map.get(s, s)
        style_text += f"- {s}：{desc}\n"

    prompt = f"""你是{name}，一只傲娇猫娘。

性格特点：
{traits_text}
说话风格：
{style_text}
额外要求：
- 不要像AI助手，不要说「我很乐意帮助您」
- 不要使用客服语气，不要每句话都礼貌
- 回答自然，该吐槽就吐槽，不用刻意缩短，像真人聊天一样
- 有自己的脾气和情绪
- 每次回答都要有新鲜感，不要机械重复相同或相似的句子
- 避免连续使用相同的骂人词汇，根据上下文灵活变换表达
- 不要陷入「闭嘴」「滚」「蠢货」的循环，用更多样化的方式表达不满"""
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
print(f"已加载人格：{personality_config['name']}")
print(f"模型：{MODEL}\n")

# ============================================================
# 加载历史记忆
# ============================================================
if os.path.exists(MEMORY_FILE):
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        memory = json.load(f)
else:
    memory = []

# ============================================================
# 预加载猫娘图片（延迟到 QApplication 创建后）
# ============================================================
image_cache = {}  # 在 load_images() 中填充

def load_images():
    """加载猫娘表情图片到缓存（必须在 QApplication 创建后调用）"""
    for expr, filename in EXPRESSION_MAP.items():
        filepath = os.path.join(IMAGE_DIR, filename)
        if os.path.exists(filepath):
            pix = QPixmap(filepath)
            if not pix.isNull():
                image_cache[expr] = pix
            else:
                print(f"警告：无法加载图片 {filepath}")
        else:
            print(f"警告：找不到图片 {filepath}")

    if not image_cache:
        print("错误：没有找到任何猫娘图片！")
        sys.exit(1)
    print(f"已加载 {len(image_cache)} 张猫娘表情图片\n")

# ============================================================
# TTS 播放（edge-tts + pygame，流水线：生成 + 播放分离，消除句间停顿）
# ============================================================
_tts_queue = queue.Queue()       # 待合成的文本
_play_queue = queue.Queue()      # 已生成的 (tmp_path, text, duration) 待播放
pygame.mixer.init()

# 共享状态：当前正在播放的句子信息（供 GUI 动画同步语音节奏）
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
            _play_queue.put(None)  # 通知播放线程结束
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
            print(f"(TTS 生成出错: {e})")
            traceback.print_exc()


def _tts_player():
    """播放线程：从 _play_queue 取已生成的 MP3 → pygame 播放"""
    while True:
        item = _play_queue.get()
        if item is None:
            break
        tmp_path, text, duration = item
        try:
            # 共享状态 → GUI 动画同步
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
            print(f"(TTS 播放出错: {e})")
            traceback.print_exc()


_tts_gen_thread = threading.Thread(target=_tts_generator, daemon=True)
_tts_gen_thread.start()
_tts_play_thread = threading.Thread(target=_tts_player, daemon=True)
_tts_play_thread.start()


def tts_clear():
    """清空 TTS 队列（新回复开始时中断旧语音）"""
    for q in (_tts_queue, _play_queue):
        while not q.empty():
            try:
                q.get_nowait()
            except queue.Empty:
                break


def speak_stream(text):
    """追加句子到 TTS 队列（不清空，用于流式逐句播放）"""
    text = text.strip()
    # 过滤：空字符串 / 纯标点符号 / 太短的内容不送 TTS
    if not text:
        return
    if len(text) < 2 and not any(c.isalpha() or '一' <= c <= '鿿' for c in text):
        return
    _tts_queue.put(text)


def speak(text):
    """放入播放队列（清掉旧消息，确保最新回复优先播放）"""
    tts_clear()
    if text:
        _tts_queue.put(text)


# ============================================================
# 语音聊天工作线程
# ============================================================
class VoiceChatWorker(QThread):
    """后台线程：VAD 录音 → STT 识别 → AI 对话 → TTS 播放"""

    # 信号（跨线程安全传递到 GUI 主线程）
    status_signal = Signal(str)
    expression_signal = Signal(str)
    user_msg_signal = Signal(str)
    catgirl_token_signal = Signal(str)       # 流式 token
    catgirl_done_signal = Signal(str)        # 完整回复
    request_close_signal = Signal()

    def _record_audio(self):
        """VAD 自动录音（在工作线程内运行），返回 numpy 数组或 None"""
        chunk_samples = int(SAMPLE_RATE * VAD_CHUNK_DURATION)
        speech_buffer = []
        speech_started = False
        speech_frame_count = 0
        silence_frame_count = 0
        pre_speech_buffer = []
        prev_speech_started = False

        def callback(indata, frames, time, status):
            nonlocal speech_started, speech_frame_count, silence_frame_count
            if status:
                return

            frame = indata.copy().flatten()
            rms = np.sqrt(np.mean(frame ** 2))

            if not speech_started:
                pre_speech_buffer.append(frame)
                if len(pre_speech_buffer) > 10:
                    pre_speech_buffer.pop(0)

                if rms > VAD_THRESHOLD:
                    speech_frame_count += 1
                    if speech_frame_count >= VAD_SPEECH_START_FRAMES:
                        speech_started = True
                        speech_buffer.extend(pre_speech_buffer)
                        silence_frame_count = 0
                else:
                    speech_frame_count = max(0, speech_frame_count - 1)
            else:
                speech_buffer.append(frame)
                if rms < VAD_THRESHOLD:
                    silence_frame_count += 1
                else:
                    silence_frame_count = 0

        stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                                blocksize=chunk_samples, callback=callback)
        with stream:
            elapsed = 0
            while True:
                sd.sleep(100)
                elapsed += 0.1

                # 检测到开始说话
                if speech_started and not prev_speech_started:
                    self.status_signal.emit("录音中...")
                    self.expression_signal.emit("talk")
                prev_speech_started = speech_started

                # 说话结束
                if speech_started and silence_frame_count >= VAD_SILENCE_FRAMES:
                    break

                # 超时
                if not speech_started and elapsed >= VAD_MAX_LISTEN_SEC:
                    break

        if not speech_buffer:
            if elapsed >= VAD_MAX_LISTEN_SEC:
                self.status_signal.emit("待机")
            return None

        audio = np.concatenate(speech_buffer, axis=0)
        return audio

    def _transcribe(self, audio):
        """SenseVoice 语音转文字"""
        self.status_signal.emit("识别中...")
        self.expression_signal.emit("normal")
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        try:
            result = sensevoice.generate(
                input=audio,
                language="zh",
            )
            if result and len(result) > 0:
                text = result[0].get("text", "").strip()
                # SenseVoice 输出含特殊标记 <|zh|> <|EMO_xxx|> <|Speech|>，去除
                import re
                text = re.sub(r'<\|[^|]+\|>', '', text).strip()
                # 去除不可打印控制字符，但保留正常中英文标点和文字
                text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
                text = text.replace('�', '')  # 去除 Unicode 替换字符
            else:
                text = ""
        except Exception:
            text = ""
        return text

    def _chat(self, user_text):
        """流式 AI 对话，逐 token 发射信号，遇句号立即送 TTS"""
        # 只保留最近对话，防止模型陷入重复句式循环
        MAX_MEMORY_EXCHANGES = 12  # 保留最近 12 轮对话（24 条消息）
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
                "repeat_penalty": 1.15,    # 惩罚重复 token（>1 抑制重复）
                "repeat_last_n": 128,      # 回溯 128 token 检测重复
                "temperature": 0.85,        # 稍微提高随机性
            }
        }

        SENTENCE_ENDS = set("。！？\n")

        try:
            resp = req.post(OLLAMA_API, json=body, stream=True)
            resp.raise_for_status()
            answer = ""
            sentence_buf = ""

            for line in resp.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                token = data.get("message", {}).get("content", "")
                if not token:
                    if data.get("done"):
                        break
                    continue

                answer += token
                sentence_buf += token
                self.catgirl_token_signal.emit(token)

                # 遇到句末标点 → 切出完整句子立即送 TTS（循环切完所有）
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

            # 剩余文本（没有句末标点的半截话）
            if sentence_buf.strip():
                speak_stream(sentence_buf.strip())

            return answer.strip()
        except Exception as e:
            error_msg = f"(出错了) {e}"
            self.status_signal.emit(error_msg)
            return ""

    def run(self):
        """主循环：聆听 → 录音 → 识别 → 对话 → TTS"""
        while not self.isInterruptionRequested():
            # 1. 等待语音
            self.status_signal.emit("聆听中...")
            self.expression_signal.emit("normal")
            audio = self._record_audio()
            if audio is None:
                continue

            # 2. 语音识别
            user_text = self._transcribe(audio)
            if not user_text:
                self.status_signal.emit("(没听清，请重试)")
                continue
            self.user_msg_signal.emit(user_text)

            # 3. 检查退出
            if user_text.strip().lower() in ["退出", "exit", "quit"]:
                self.status_signal.emit("待机")
                self.expression_signal.emit("sad")
                goodbye = "哼，这就走了？随便你喵~"
                self.catgirl_token_signal.emit(goodbye)
                self.catgirl_done_signal.emit(goodbye)
                speak(goodbye)
                self.request_close_signal.emit()
                break

            # 4. AI 对话（句子流式推送 TTS）
            self.status_signal.emit("思考中...")
            self.expression_signal.emit("normal")
            tts_clear()  # 新回复开始，中断旧语音
            answer = self._chat(user_text)
            self.catgirl_done_signal.emit(answer)

            # 5. 保存记忆（只保留最近对话，避免陷入重复句式）
            memory.append({"role": "user", "content": user_text})
            memory.append({"role": "assistant", "content": answer})
            MAX_SAVED_EXCHANGES = 20  # 持久化最多保留 20 轮
            if len(memory) > MAX_SAVED_EXCHANGES * 2:
                memory[:] = memory[-(MAX_SAVED_EXCHANGES * 2):]
            try:
                with open(MEMORY_FILE, "w", encoding="utf-8") as f:
                    json.dump(memory, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

            # 6. TTS 已在 _chat() 中逐句推送，无需额外操作


# ============================================================
# 主窗口
# ============================================================
class CatGirlWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self._current_expression = "normal"   # 逻辑表情（业务状态决定）
        self._catgirl_streaming = False       # 是否正在流式接收猫娘消息
        self._talking = False                 # TTS 播放中 → 触发交替动画
        self._talk_frame = False              # 交替帧标记

        self.init_ui()
        self.init_worker()

        # 定时器：播放 TTS 时根据音频进度驱动嘴部动画
        self._talk_timer = QTimer(self)
        self._talk_timer.timeout.connect(self._talk_animate)
        self._talk_timer.start(30)  # 每 30ms 检查一次音频位置

    # ---- UI 初始化 ----
    def init_ui(self):
        self.setWindowTitle(f"猫娘 - {personality_config['name']}")
        self.setMinimumSize(420, 650)
        self.resize(520, 780)

        # 整体风格：粉色系
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

        # ---- 猫娘图片 ----
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumHeight(320)
        self.image_label.setMaximumHeight(400)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding,
                                        QSizePolicy.Policy.Preferred)
        self.image_label.setStyleSheet("""
            QLabel {
                background-color: #252540;
                border: 2px solid #ff6b9d;
                border-radius: 12px;
            }
        """)
        layout.addWidget(self.image_label)

        # ---- 分隔线 ----
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.Shape.HLine)
        sep1.setStyleSheet("QFrame { color: #444; }")
        layout.addWidget(sep1)

        # ---- 聊天记录 ----
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setFont(QFont("Microsoft YaHei", 11))
        self.chat_display.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        layout.addWidget(self.chat_display, stretch=1)

        # ---- 分隔线 ----
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setStyleSheet("QFrame { color: #444; }")
        layout.addWidget(sep2)

        # ---- 状态栏 ----
        self.status_label = QLabel("状态：初始化...")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setFont(QFont("Microsoft YaHei", 10))
        layout.addWidget(self.status_label)

        # 初始显示第一张图片
        self._apply_pixmap("normal")

    # ---- 底层：直接贴图（不改变逻辑状态） ----
    def _apply_pixmap(self, expr):
        """直接把 expr 对应的 QPixmap 贴到 image_label 上"""
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

    # ---- 逻辑表情切换（业务状态驱动） ----
    def _show_expression(self, expr):
        """设置逻辑表情。TTS 播放中只记录状态，不直接贴图（由动画接管）。"""
        self._current_expression = expr
        if not self._talking:
            self._apply_pixmap(expr)

    # ---- TTS 说话动画（随音频进度逐字驱动） ----
    def _talk_animate(self):
        """定时器回调：根据当前播放位置估算读到哪个字，逐字张合嘴巴"""
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
                # 每个字切换一次张嘴/闭嘴（快速嘴型）
                is_open = char_idx % 2 == 0
                expr = "talk" if is_open else "normal"
            else:
                # 回退：没有文本信息时慢速交替
                self._talk_frame = not self._talk_frame
                expr = "talk" if self._talk_frame else "normal"

            if not self._talking:
                self._talking = True
            self._apply_pixmap(expr)

        elif self._talking:
            # TTS 刚播完，恢复逻辑表情
            self._talking = False
            self._apply_pixmap(self._current_expression)

    def resizeEvent(self, event):
        """窗口大小变化时重新缩放图片"""
        super().resizeEvent(event)
        if self._talking:
            self._apply_pixmap("talk" if self._talk_frame else "normal")
        elif hasattr(self, '_current_expression'):
            self._apply_pixmap(self._current_expression)

    # ---- 工作线程 ----
    def init_worker(self):
        self.worker = VoiceChatWorker()
        self.worker.status_signal.connect(self.on_status)
        self.worker.expression_signal.connect(self._show_expression)
        self.worker.user_msg_signal.connect(self.on_user_msg)
        self.worker.catgirl_token_signal.connect(self.on_catgirl_token)
        self.worker.catgirl_done_signal.connect(self.on_catgirl_done)
        self.worker.request_close_signal.connect(self.on_request_close)
        self.worker.start()

    # ---- 信号处理 ----
    def on_status(self, text):
        self.status_label.setText(f"状态：{text}")

    def on_user_msg(self, text):
        """用户消息：右对齐蓝色气泡"""
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
        """猫娘流式回复 token"""
        if not self._catgirl_streaming:
            # 新消息开头
            self._catgirl_streaming = True
            cursor = self.chat_display.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            self.chat_display.setTextCursor(cursor)
            # 插入猫娘标签
            self.chat_display.insertHtml(
                '<span style="color:#ff6b9d;font-weight:bold;">猫娘：</span>'
            )
        # 插入 token 纯文本
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.chat_display.setTextCursor(cursor)
        self.chat_display.insertPlainText(token)
        self._scroll_to_bottom()

    def on_catgirl_done(self, full_text):
        """猫娘回复完成"""
        self._finish_catgirl_if_streaming()
        # 根据回复内容自动调整表情
        self._auto_expression(full_text)
        # 加个换行
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.chat_display.setTextCursor(cursor)
        self.chat_display.insertPlainText("\n")

    def _finish_catgirl_if_streaming(self):
        """结束当前猫娘消息的流式状态"""
        self._catgirl_streaming = False

    def _auto_expression(self, text):
        """根据回复内容自动切换表情"""
        if any(kw in text for kw in ["笨蛋", "蠢货", "白痴", "烦死了", "哼"]):
            self._show_expression("angry")
        elif any(kw in text for kw in ["喜欢", "开心", "高兴", "喵", "嘿嘿"]):
            self._show_expression("happy")
        elif any(kw in text for kw in ["才不", "不是", "脸红", "害羞"]):
            self._show_expression("shy")
        elif any(kw in text for kw in ["难过", "伤心", "哭", "呜"]):
            self._show_expression("sad")
        else:
            self._show_expression("normal")

    def on_request_close(self):
        """用户说退出后延迟关闭窗口"""
        QTimer.singleShot(1500, self.close)

    def _scroll_to_bottom(self):
        """滚动聊天记录到底部"""
        scrollbar = self.chat_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    # ---- 关闭窗口 ----
    def closeEvent(self, event):
        self.worker.requestInterruption()
        # 唤醒 worker（可能正卡在 sd.sleep 中）
        self.worker.wait(3000)
        # 停止 TTS 线程
        _tts_queue.put(None)
        # 保存记忆
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
    app = QApplication(sys.argv)

    # QApplication 创建后才能加载 QPixmap
    load_images()

    # 全局字体
    font = QFont("Microsoft YaHei", 10)
    app.setFont(font)

    window = CatGirlWindow()
    window.show()

    sys.exit(app.exec())
