"""
猫娘语音聊天增强版 — Nora Plus
功能：VAD 语音识别 / Ollama LLM 对话 / edge-tts 语音合成 /
      好感度系统 / 情绪系统 / 表情联动 / 主动聊天 / 长期记忆

入口文件：组装所有模块并启动应用。
"""
import sys
import os
import site

# Windows: 添加 nvidia pip 包的 cublas DLL 目录到搜索路径
for _sp in site.getsitepackages():
    _dll_path = os.path.join(_sp, "nvidia", "cublas", "bin")
    if os.path.isdir(_dll_path):
        os.add_dll_directory(_dll_path)
        os.environ["PATH"] = _dll_path + os.pathsep + os.environ.get("PATH", "")

import pygame

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont

from config import (
    SENSEVOICE_MODEL, OLLAMA_API, MODEL, TTS_VOICE,
    VAD_THRESHOLD, VAD_SPEECH_START_FRAMES,
    VAD_SILENCE_FRAMES, VAD_MAX_LISTEN_SEC,
    MEMORY_EXTRACTION_INTERVAL,
)
from state_manager import NoraStateManager
from audio_vad import VADRecorder
from stt_engine import STTEngine
from llm_client import LLMClient, LLMExtractor
from tts_engine import TTSEngine
from chat_worker import VoiceChatWorker
from ui.main_window import CatGirlWindow


def main():
    # ---- 初始化 pygame mixer（必须在 QApplication 前） ----
    pygame.mixer.init()

    # ---- 状态管理器 ----
    print("加载状态...")
    state = NoraStateManager()
    print(f"已加载人格：{state.personality_config['name']}")
    print(f"初始好感度：{state.favorability}")
    print(f"初始情绪：{state.current_emotion}")
    print(f"模型：{MODEL}\n")

    # ---- 引擎 ----
    stt = STTEngine(SENSEVOICE_MODEL, device="cuda:0")
    llm = LLMClient(OLLAMA_API, MODEL)
    extractor = LLMExtractor(llm, MEMORY_EXTRACTION_INTERVAL)
    tts = TTSEngine(TTS_VOICE)

    # ---- 创建 worker（先创建，window 需要引用） ----
    worker = VoiceChatWorker(state, None, stt, llm, extractor, tts)
    # VAD 回调需要 worker 信号，所以先创建 worker 再创建 VAD

    # ---- Qt 应用 ----
    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei", 10))

    # ---- 窗口 ----
    window = CatGirlWindow(state, worker, tts)
    window.show()

    # ---- VAD（创建在 QApplication 之后，回调连到 window） ----
    vad = VADRecorder(
        threshold=VAD_THRESHOLD,
        speech_start_frames=VAD_SPEECH_START_FRAMES,
        silence_frames=VAD_SILENCE_FRAMES,
        max_listen_sec=VAD_MAX_LISTEN_SEC,
        on_status=lambda s: worker.status_signal.emit(s),
        on_expression=lambda e: worker.expression_signal.emit(e),
    )
    worker._vad = vad  # 注入 VAD

    # ---- 启动 worker ----
    worker.start()

    # ---- 事件循环 ----
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
