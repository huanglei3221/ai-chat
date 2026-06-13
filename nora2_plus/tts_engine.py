"""
TTSEngine —— edge-tts 双队列流水线语音合成
"""
import os
import asyncio
import queue
import threading
import tempfile
import time

import edge_tts
import pygame


class TTSEngine:
    """edge-tts 语音合成引擎（双队列流水线：生成 + 播放分离）"""

    def __init__(self, voice="zh-CN-XiaoyiNeural"):
        self.voice = voice
        self._tts_queue = queue.Queue()   # 待合成的文本
        self._play_queue = queue.Queue()  # 已合成的 MP3

        # 当前播放状态（供嘴部动画读取）
        self._current_text = ""
        self._current_duration = 0.0
        self._current_start = 0.0
        self._lock = threading.Lock()

        # 启动后台线程
        self._gen_thread = threading.Thread(target=self._generator, daemon=True)
        self._gen_thread.start()
        self._play_thread = threading.Thread(target=self._player, daemon=True)
        self._play_thread.start()

    # ================================================================
    # 公开 API
    # ================================================================
    def speak(self, text):
        """放入 TTS 队列（先清空旧消息，确保最新回复优先播放）"""
        self.clear()
        text = text.strip()
        if text:
            self._tts_queue.put(text)

    def clear(self):
        """清空队列，中断当前播放"""
        for q in (self._tts_queue, self._play_queue):
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

    def get_playback_position(self):
        """
        获取当前播放状态（线程安全）。
        返回 (text, duration, elapsed) 或 None（未在播放）。
        供嘴部动画使用。
        """
        with self._lock:
            if not self._current_text or self._current_duration <= 0:
                return None
            elapsed = time.monotonic() - self._current_start
            return (self._current_text, self._current_duration, elapsed)

    def is_playing(self):
        """检查是否正在播放"""
        return pygame.mixer.music.get_busy()

    def shutdown(self):
        """优雅停止后台线程"""
        self._tts_queue.put(None)
        self._play_queue.put(None)

    # ================================================================
    # 后台线程
    # ================================================================
    def _generator(self):
        """生成线程：文本 → edge-tts → MP3 文件"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        while True:
            text = self._tts_queue.get()
            if text is None:
                self._play_queue.put(None)
                break
            try:
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                    tmp_path = f.name

                communicate = edge_tts.Communicate(text, self.voice)
                loop.run_until_complete(communicate.save(tmp_path))

                try:
                    sound = pygame.mixer.Sound(tmp_path)
                    duration = sound.get_length()
                except Exception:
                    duration = len(text) * 0.25

                self._play_queue.put((tmp_path, text, duration))
            except Exception as e:
                import traceback
                print(f"(TTS 生成出错: {e})")
                traceback.print_exc()

    def _player(self):
        """播放线程：MP3 文件 → pygame 播放"""
        while True:
            item = self._play_queue.get()
            if item is None:
                break
            tmp_path, text, duration = item
            try:
                with self._lock:
                    self._current_text = text
                    self._current_duration = duration
                    self._current_start = time.monotonic()

                pygame.mixer.music.load(tmp_path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
                pygame.mixer.music.unload()

                with self._lock:
                    self._current_text = ""
                    self._current_duration = 0.0

                try:
                    os.unlink(tmp_path)
                except PermissionError:
                    pass
            except Exception as e:
                import traceback
                print(f"(TTS 播放出错: {e})")
                traceback.print_exc()
