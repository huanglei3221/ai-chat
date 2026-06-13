"""
用户语音输入程序 — VAD 自动检测 + SenseVoice 识别 + TCP 发送

架构：
  - VAD 录音 → SenseVoice 转文字 → TCP socket 发送给两只猫娘
  - 与猫娘通过 TCP (localhost:8765 / localhost:8766) 通信
  - 协议：每行一个 JSON {"text": "..."}，以 \n 分隔
  - 持久连接 + 自动重连
  - 键盘后备模式：直接打字按 Enter 输入

启动方式：python user.py
依赖：sounddevice, numpy, funasr, torch, requests
"""

import sys
import json
import os
import time
import re
import select
import socket
import threading
import numpy as np
import sounddevice as sd

# Windows 控制台 utf-8
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Windows: CUDA DLL
import site
for _sp in site.getsitepackages():
    _dll_path = os.path.join(_sp, "nvidia", "cublas", "bin")
    if os.path.isdir(_dll_path):
        os.add_dll_directory(_dll_path)
        os.environ["PATH"] = _dll_path + os.pathsep + os.environ.get("PATH", "")

from funasr import AutoModel

# ============================================================
# 配置
# ============================================================
CAT_A_HOST = "localhost"
CAT_A_PORT = 8765
CAT_B_HOST = "localhost"
CAT_B_PORT = 8766
SAMPLE_RATE = 16000

VAD_THRESHOLD = 0.015
VAD_SPEECH_START_FRAMES = 8
VAD_SILENCE_FRAMES = 25
VAD_CHUNK_DURATION = 0.06
VAD_MAX_LISTEN_SEC = 10

print("=" * 55)
print("  双猫娘语音聊天 -- 用户语音输入程序")
print("=" * 55)
print()

# ============================================================
# 音频设备
# ============================================================
print("[音频] 可用输入设备：")
try:
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        if d['max_input_channels'] > 0:
            print(f"   [{i}] {d['name']}")
except Exception as e:
    print(f"   (无法列出设备: {e})")
default_dev = sd.query_devices(kind='input')
print(f"   当前使用: {default_dev['name']}")
print()

# ============================================================
# TCP 连接管理
# ============================================================
class TCPClient:
    """TCP 客户端：连接到一只猫娘，发送 JSON 行消息，断线自动重连"""

    def __init__(self, name, host, port):
        self.name = name
        self.host = host
        self.port = port
        self._sock = None
        self._lock = threading.Lock()

    def connect(self):
        """建立 TCP 连接"""
        with self._lock:
            if self._sock:
                try:
                    self._sock.close()
                except Exception:
                    pass
            try:
                self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self._sock.settimeout(5)
                self._sock.connect((self.host, self.port))
                self._sock.settimeout(None)  # 发送时阻塞
                print(f"   {self.name}: 已连接 localhost:{self.port}")
                return True
            except Exception as e:
                print(f"   {self.name}: 连接失败 ({e})")
                self._sock = None
                return False

    def send(self, text):
        """发送一行 JSON 消息。失败则尝试重连一次。"""
        with self._lock:
            if self._sock is None:
                if not self._reconnect_locked():
                    return False
            try:
                msg = json.dumps({"text": text}, ensure_ascii=False) + "\n"
                self._sock.sendall(msg.encode("utf-8"))
                return True
            except Exception:
                # 尝试重连后重发
                if not self._reconnect_locked():
                    return False
                try:
                    msg = json.dumps({"text": text}, ensure_ascii=False) + "\n"
                    self._sock.sendall(msg.encode("utf-8"))
                    return True
                except Exception:
                    return False

    def _reconnect_locked(self):
        """已持有 _lock 的重连"""
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.settimeout(5)
            self._sock.connect((self.host, self.port))
            self._sock.settimeout(None)
            print(f"   {self.name}: 已重连")
            return True
        except Exception:
            self._sock = None
            return False

    def check_health(self):
        """健康检查：尝试连接后立即断开"""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(2)
            s.connect((self.host, self.port))
            s.close()
            return True
        except Exception:
            return False

    def close(self):
        with self._lock:
            if self._sock:
                try:
                    self._sock.close()
                except Exception:
                    pass
                self._sock = None


# ============================================================
# 连接两只猫娘
# ============================================================
print("[TCP] 连接猫娘服务...")
cat_a = TCPClient("小猫A", CAT_A_HOST, CAT_A_PORT)
cat_b = TCPClient("小猫B", CAT_B_HOST, CAT_B_PORT)

if cat_a.check_health():
    print("   小猫A: 在线")
    cat_a.connect()
else:
    print("   小猫A: 未启动！请先运行 nora1.py")

if cat_b.check_health():
    print("   小猫B: 在线")
    cat_b.connect()
else:
    print("   小猫B: 未启动！请先运行 nora2.py")
print()


def send_to_cats(text):
    """发送文字给两只猫娘"""
    ok_a = cat_a.send(text)
    ok_b = cat_b.send(text)
    if not ok_a:
        print("   小猫A: 发送失败")
    if not ok_b:
        print("   小猫B: 发送失败")


# ============================================================
# SenseVoice 模型 — 后台线程加载，不阻塞主流程
# ============================================================
sensevoice = None
_sensevoice_ready = threading.Event()
_sensevoice_loading = False

def _load_sensevoice():
    """后台线程：加载 SenseVoice 模型"""
    global sensevoice, _sensevoice_loading
    _sensevoice_loading = True
    try:
        print("[模型] 后台加载 SenseVoiceSmall (首次需下载 ~200MB，请稍候)...")
        sensevoice = AutoModel(
            model="iic/SenseVoiceSmall",
            device="cuda:0",
            disable_update=True,       # 跳过在线更新检查
        )
        print("[模型] SenseVoiceSmall 加载完成！")
        _sensevoice_ready.set()
    except Exception as e:
        print(f"[模型] 加载失败: {e}")
        print("[模型] 将只能使用键盘输入模式。")
        sensevoice = None
    _sensevoice_loading = False

# 启动后台加载
_load_thread = threading.Thread(target=_load_sensevoice, daemon=True)
_load_thread.start()

# ============================================================
# 键盘检测
# ============================================================
def _kbhit():
    if sys.platform == "win32":
        import msvcrt
        return msvcrt.kbhit()
    else:
        return bool(select.select([sys.stdin], [], [], 0)[0])


def read_keyboard_line():
    return input().strip()


def check_keyboard_input():
    if _kbhit():
        return read_keyboard_line()
    return None


# ============================================================
# 噪音校准 — 自动调整 VAD 阈值
# ============================================================
def calibrate_vad(duration=2.0):
    """采样环境噪音，自动设置 VAD 阈值（噪音均值 * 3）"""
    global VAD_THRESHOLD
    print(f"[校准] 采样环境噪音 {duration:.0f} 秒，请保持安静...")
    chunk_samples = int(SAMPLE_RATE * VAD_CHUNK_DURATION)
    rms_values = []

    def callback(indata, frames, time_info, status):
        if not status:
            rms_values.append(float(np.sqrt(np.mean(indata ** 2))))

    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                            blocksize=chunk_samples, callback=callback)
    with stream:
        sd.sleep(int(duration * 1000))

    if rms_values:
        arr = np.array(rms_values)
        noise_median = float(np.median(arr))
        # 中位数抗尖峰，x2 倍，上限 0.030，下限 0.005
        new_threshold = min(noise_median * 2.0, 0.030)
        new_threshold = max(new_threshold, 0.005)
        print(f"[校准] 噪音中位: {noise_median:.6f}, 新阈值: {new_threshold:.6f}")
        VAD_THRESHOLD = float(new_threshold)
    else:
        print(f"[校准] 采样失败，使用默认阈值: {VAD_THRESHOLD}")
def record_audio():
    """VAD 自动录音，返回 numpy 数组或 None"""
    chunk_samples = int(SAMPLE_RATE * VAD_CHUNK_DURATION)
    speech_buffer = []
    speech_started = False
    speech_frame_count = 0
    silence_frame_count = 0
    pre_speech_buffer = []
    prev_speech_started = False
    last_display_time = 0
    current_rms = 0.0

    def callback(indata, frames, time_info, status):
        nonlocal speech_started, speech_frame_count, silence_frame_count, current_rms
        if status:
            return

        frame = indata.copy().flatten()
        current_rms = float(np.sqrt(np.mean(frame ** 2)))

        if not speech_started:
            pre_speech_buffer.append(frame)
            if len(pre_speech_buffer) > 10:
                pre_speech_buffer.pop(0)

            if current_rms > VAD_THRESHOLD:
                speech_frame_count += 1
                if speech_frame_count >= VAD_SPEECH_START_FRAMES:
                    speech_started = True
                    speech_buffer.extend(pre_speech_buffer)
                    silence_frame_count = 0
            else:
                speech_frame_count = max(0, speech_frame_count - 1)
        else:
            speech_buffer.append(frame)
            if current_rms < VAD_THRESHOLD:
                silence_frame_count += 1
            else:
                silence_frame_count = 0

    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                            blocksize=chunk_samples, callback=callback)
    with stream:
        elapsed = 0
        print("\r[聆听中] (打字按 Enter，说\"退出\"结束)  ", end="", flush=True)
        while True:
            sd.sleep(100)
            elapsed += 0.1

            # 实时音量条（每 0.3 秒刷新）
            now = time.time()
            if now - last_display_time > 0.3:
                last_display_time = now
                bar_len = min(int(current_rms * 800), 20)
                bar = "#" * bar_len + "-" * (20 - bar_len)
                tag = "录音" if speech_started else "聆听"
                print(f"\r[音量 |{bar}| {current_rms:.4f}] {tag}...   ", end="", flush=True)

            # 键盘按下 -> 退出 VAD
            if not speech_started and _kbhit():
                break

            if speech_started and not prev_speech_started:
                print(f"\n[检测到语音，录音中...]")
            prev_speech_started = speech_started

            if speech_started and silence_frame_count >= VAD_SILENCE_FRAMES:
                break

            if not speech_started and elapsed >= VAD_MAX_LISTEN_SEC:
                break

    if not speech_buffer:
        return None

    audio = np.concatenate(speech_buffer, axis=0)
    dur = len(audio) / SAMPLE_RATE
    print(f"\r[录音结束，{dur:.1f} 秒]                           ")
    return audio


# ============================================================
# 语音转文字
# ============================================================
def transcribe(audio):
    if sensevoice is None:
        if _sensevoice_loading:
            print("  (模型还在加载中，请稍等片刻...)")
        return ""
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    try:
        result = sensevoice.generate(input=audio, language="zh")
        if result and len(result) > 0:
            text = result[0].get("text", "").strip()
            text = re.sub(r'<\|[^|]+\|>', '', text).strip()
            text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
            text = text.replace('�', '')
        else:
            text = ""
    except Exception as e:
        print(f"  (识别出错: {e})")
        text = ""
    return text


# ============================================================
# 主循环
# ============================================================
def main():
    print("=" * 55)
    print("[语音模式] 直接说话即可")
    print("[键盘模式] 打字按 Enter 发送")
    print("说\"退出\"或输入\"退出\"结束程序")
    print("=" * 55)
    print()

    # ---- 0. 噪音校准 ----
    try:
        calibrate_vad(2.0)
    except Exception as e:
        print(f"[校准] 失败: {e}，使用默认阈值: {VAD_THRESHOLD}")
    print()

    # ---- 1. 发送测试消息确认链路 ----
    print("[自检] 发送测试消息到两只猫娘...")
    test_text = "喵~ 系统启动测试"
    send_to_cats(test_text)
    print("[自检] 测试消息已发送，观察猫娘窗口是否有反应")
    print("[自检] 如果没有反应，请确认：")
    print("       1) Ollama 是否在运行 (http://localhost:11434)")
    print(f"       2) gemma3:4b 和 qwen2.5:3b 模型是否已 pull")
    print()

    while True:
        # ---- 键盘输入检查 ----
        kb_text = check_keyboard_input()
        if kb_text:
            print(f"\n[键盘输入] {kb_text}")
            if kb_text.lower() in ["退出", "exit", "quit"]:
                print("发送退出信号...")
                send_to_cats(kb_text)
                print("再见！")
                break
            send_to_cats(kb_text)
            print()

        # ---- VAD 录音 ----
        audio = record_audio()
        if audio is None:
            continue

        # ---- 语音识别 ----
        print("[识别中...]")
        text = transcribe(audio)

        if not text:
            print("  (没听清，请重试)")
            print()
            continue

        print(f"[识别结果] {text}")

        # ---- 退出 ----
        if text.strip().lower() in ["退出", "exit", "quit"]:
            print("发送退出信号给两只猫娘...")
            send_to_cats(text)
            print("再见！")
            break

        # ---- 发送 ----
        send_to_cats(text)
        print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序已中断")
    finally:
        cat_a.close()
        cat_b.close()
