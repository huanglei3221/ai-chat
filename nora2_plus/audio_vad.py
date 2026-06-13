"""
VADRecorder —— VAD 自动录音器
封装 sounddevice 录音、RMS 能量检测、静音裁剪。
"""
import numpy as np
import sounddevice as sd

from config import SAMPLE_RATE, VAD_CHUNK_DURATION


class VADRecorder:
    """VAD 自动录音器。检测语音活动，自动开始/停止录音。"""

    def __init__(self, threshold=0.015, speech_start_frames=8,
                 silence_frames=25, max_listen_sec=30,
                 on_status=None, on_expression=None):
        self.threshold = threshold
        self.speech_start_frames = speech_start_frames
        self.silence_frames = silence_frames
        self.max_listen_sec = max_listen_sec
        self.on_status = on_status or (lambda s: None)
        self.on_expression = on_expression or (lambda e: None)

        self.chunk_samples = int(SAMPLE_RATE * VAD_CHUNK_DURATION)

    def record(self):
        """阻塞录制，检测到语音返回 numpy 数组，超时返回 None"""
        speech_buffer = []
        speech_started = False
        speech_frame_count = 0
        silence_frame_count = 0
        pre_speech_buffer = []  # 环形缓冲区，保留语音开始前的少量音频
        prev_speech_started = False

        def callback(indata, frames, time_info, status):
            nonlocal speech_started, speech_frame_count, silence_frame_count
            if status:
                return

            frame = indata.copy().flatten()
            rms = np.sqrt(np.mean(frame ** 2))

            if not speech_started:
                pre_speech_buffer.append(frame)
                if len(pre_speech_buffer) > 10:
                    pre_speech_buffer.pop(0)

                if rms > self.threshold:
                    speech_frame_count += 1
                    if speech_frame_count >= self.speech_start_frames:
                        speech_started = True
                        speech_buffer.extend(pre_speech_buffer)
                        silence_frame_count = 0
                else:
                    speech_frame_count = max(0, speech_frame_count - 1)
            else:
                speech_buffer.append(frame)
                if rms < self.threshold:
                    silence_frame_count += 1
                else:
                    silence_frame_count = 0

        stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                                blocksize=self.chunk_samples, callback=callback)
        with stream:
            elapsed = 0
            while True:
                sd.sleep(100)
                elapsed += 0.1

                if speech_started and not prev_speech_started:
                    self.on_status("录音中...")
                    self.on_expression("talk")
                prev_speech_started = speech_started

                if speech_started and silence_frame_count >= self.silence_frames:
                    break

                if not speech_started and elapsed >= self.max_listen_sec:
                    break

        if not speech_buffer:
            if elapsed >= self.max_listen_sec:
                self.on_status("待机")
            return None

        audio = np.concatenate(speech_buffer, axis=0)
        return audio
