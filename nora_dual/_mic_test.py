"""
麦克风独立测试 — 验证录音和 VAD 是否正常

运行: python _mic_test.py
输出: 设备列表 → 实时音量条 → 检测到说话就录音 → 显示结果
"""

import sys
import time
import numpy as np
import sounddevice as sd

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

SAMPLE_RATE = 16000
VAD_THRESHOLD = 0.010   # 先用较敏感的阈值
DURATION = 0.06

print("=" * 55)
print("  麦克风 VAD 测试")
print("=" * 55)
print()

# ---- 1. 设备列表 ----
print("[1] 音频输入设备:")
devices = sd.query_devices()
for i, d in enumerate(devices):
    if d['max_input_channels'] > 0:
        print(f"     [{i}] {d['name']}")
default_dev = sd.query_devices(kind='input')
print(f"     默认设备: {default_dev['name']}")
print(f"     采样率: {default_dev['default_samplerate']}")
print()

# ---- 2. 实时音量监视 (10 秒，不说话，看环境噪音) ----
print("[2] 环境噪音采样 (请保持安静 3 秒)...")
chunk_samples = int(SAMPLE_RATE * DURATION)
noise_samples = []

def noise_cb(indata, frames, time_info, status):
    if not status:
        noise_samples.append(float(np.sqrt(np.mean(indata ** 2))))

stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                        blocksize=chunk_samples, callback=noise_cb)
with stream:
    sd.sleep(3000)

if noise_samples:
    arr = np.array(noise_samples)
    noise_median = float(np.median(arr))
    noise_p90 = float(np.percentile(arr, 90))
    noise_max = float(np.max(arr))
    # 用中位数（抗尖峰）* 2.0，上限 0.030
    threshold = min(noise_median * 2.0, 0.030)
    threshold = max(threshold, 0.005)
    print(f"     噪音中位: {noise_median:.6f}")
    print(f"     噪音P90:  {noise_p90:.6f}")
    print(f"     噪音峰值: {noise_max:.6f}")
    print(f"     建议阈值: {threshold:.6f} (中位数 x2, 上限 0.03)")
else:
    threshold = 0.010
    print(f"     采样失败，使用默认阈值: {threshold}")
print()

# ---- 3. VAD 实时测试 (20 秒) ----
print("[3] VAD 实时监视 (20 秒，请说话测试)...")
print("    音量条: # = 有声音  - = 安静")
print("    出现 *** 表示检测到说话")
print("    Ctrl+C 提前退出")
print()

chunk_samples = int(SAMPLE_RATE * DURATION)
speech_started = False
speech_frames = 0
silence_frames = 0
pre_buffer = []
speech_buffer = []
detected_count = 0
current_rms = 0.0

def vad_cb(indata, frames, time_info, status):
    global speech_started, speech_frames, silence_frames, current_rms
    if status:
        return
    frame = indata.copy().flatten()
    current_rms = float(np.sqrt(np.mean(frame ** 2)))

    if not speech_started:
        pre_buffer.append(frame)
        if len(pre_buffer) > 10:
            pre_buffer.pop(0)
        if current_rms > threshold:
            speech_frames += 1
            if speech_frames >= 8:
                speech_started = True
                speech_buffer.extend(pre_buffer)
                silence_frames = 0
        else:
            speech_frames = max(0, speech_frames - 1)
    else:
        speech_buffer.append(frame)
        if current_rms < threshold:
            silence_frames += 1
        else:
            silence_frames = 0

stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                        blocksize=chunk_samples, callback=vad_cb)
start_time = time.time()
last_display = 0

try:
    with stream:
        while time.time() - start_time < 20:
            sd.sleep(50)
            elapsed = time.time() - start_time
            now = time.time()

            if now - last_display > 0.2:
                last_display = now
                bar_len = min(int(current_rms * 800), 30)
                bar = "#" * bar_len + "-" * (30 - bar_len)
                status = "[录音]" if speech_started else "[聆听]"
                print(f"\r  {status} |{bar}| rms={current_rms:.5f}  thr={threshold:.5f}  {elapsed:.0f}s  ", end="", flush=True)

            if speech_started and silence_frames >= 30:
                dur = len(speech_buffer) / SAMPLE_RATE
                detected_count += 1
                print(f"\n  *** 检测到语音 #{detected_count}，时长 {dur:.1f} 秒 ***")
                speech_started = False
                speech_frames = 0
                silence_frames = 0
                speech_buffer.clear()
                pre_buffer.clear()

except KeyboardInterrupt:
    print("\n  用户中断")

print()
print()

# ---- 4. 结果 ----
if detected_count > 0:
    print(f"[OK] 麦克风工作正常！检测到 {detected_count} 次语音。")
    print("    如果猫娘没反应，问题在 user.py 的 SenseVoice 识别或 TCP 发送环节。")
else:
    print("[FAIL] 20 秒内未检测到任何语音。")
    print("    可能原因：")
    print("    1. 麦克风未插好或被禁用")
    print("    2. 麦克风音量太低（Windows 设置中调高）")
    print("    3. 默认设备选错了（查看上面的设备列表）")
    print(f"    4. 阈值 {threshold:.5f} 太高（尝试对着麦克风吹气）")
    print()
    print("    手动测试方法：")
    print("    python -c \"import sounddevice as sd; print(sd.query_devices())\"")

print()
print("=" * 55)
