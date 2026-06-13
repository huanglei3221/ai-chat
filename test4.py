"""
语音猫娘聊天 — 通过语音和 AI 傲娇猫娘实时对话。

功能：
  - 按 Enter 开始/停止录音，实时语音识别
  - 猫娘用文字 + 语音回复，支持多轮对话记忆
  - 人格通过 personality.json 配置，启动时自动生成 System Prompt

实现方法：
  录音    → sounddevice（16kHz 单声道）
  语音识别 → faster-whisper (base)，通过 ModelScope 下载模型，GPU(CUDA) 推理
  AI 对话 → Ollama /api/chat 流式接口，复用人格配置
  语音合成 → edge-tts（微软晓晓女声），pygame 播放
  对话记忆 → chat_memory.json 持久化

依赖库（pip install）：
  sounddevice numpy         — 录音
  faster-whisper modelscope — 语音识别（GPU 需 nvidia-cublas-cu12）
  requests                  — Ollama API 调用
  edge-tts pygame           — 语音合成与播放
"""

import requests
import json
import os
import wave
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

from modelscope import snapshot_download
from faster_whisper import WhisperModel
import asyncio
import tempfile
import edge_tts
import pygame

# --------------------------
# 配置部分
# --------------------------
OLLAMA_API = "http://localhost:11434/api/chat"
MODEL = "llama3.1:8b"
MEMORY_FILE = "chat_memory.json"
PERSONALITY_FILE = "personality.json"
WHISPER_MODEL_SIZE = "base"  # tiny / base / small / medium （越大越准但越慢）
SAMPLE_RATE = 16000  # 录音采样率

# --------------------------
# 初始化 STT（faster-whisper，通过 ModelScope 下载模型）
# --------------------------

print(f"加载语音识别模型 (faster-whisper {WHISPER_MODEL_SIZE})...")
# 用 ModelScope 下载（国内快），下载后返回本地路径
model_path = snapshot_download(f"Systran/faster-whisper-{WHISPER_MODEL_SIZE}")
whisper = WhisperModel(model_path, device="cuda", compute_type="float16")
print("语音识别模型加载完成\n")

# --------------------------
# 从配置文件生成 System Prompt（同 test3.py）
# --------------------------
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
        "经常说笨蛋": '口头禅是「笨蛋」「你这笨蛋」，几乎每句话都要带',
        "偶尔说蠢货": '用户犯明显错误时会骂「蠢货」或「白痴」',
        "不使用敬语": '说话不用「您」，不用「请」，不用「谢谢」，语气随意甚至粗鲁',
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
- 有自己的脾气和情绪"""
    return prompt


# --------------------------
# 加载人格配置
# --------------------------
if not os.path.exists(PERSONALITY_FILE):
    print(f"错误：找不到 {PERSONALITY_FILE}")
    exit(1)

with open(PERSONALITY_FILE, "r", encoding="utf-8") as f:
    personality_config = json.load(f)

SYSTEM_PROMPT = build_system_prompt(personality_config)
print(f"已加载人格：{personality_config['name']}")
print(f"模型：{MODEL}\n")

# --------------------------
# 加载历史记忆
# --------------------------
if os.path.exists(MEMORY_FILE):
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        memory = json.load(f)
else:
    memory = []


# --------------------------
# 录音函数（按 Enter 开始/停止）
# --------------------------
def record_audio():
    """按 Enter 开始录音，再按 Enter 停止，返回音频数据"""
    input("按 Enter 开始说话...")

    audio_chunks = []
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                            callback=lambda *a: audio_chunks.append(a[0].copy()))
    with stream:
        print("🎤 录音中... 按 Enter 停止")
        input()

    if not audio_chunks:
        return None

    audio = np.concatenate(audio_chunks, axis=0).flatten()
    duration = len(audio) / SAMPLE_RATE
    print(f"录音结束，时长 {duration:.1f} 秒")

    # 保存为临时 wav，faster-whisper 可以直接读 numpy 数组
    return audio


# --------------------------
# 语音识别
# --------------------------
def transcribe(audio):
    """faster-whisper 语音转文字"""
    print("识别中...", end="", flush=True)
    segments, _ = whisper.transcribe(audio, language="zh", beam_size=5)
    text = " ".join(seg.text for seg in segments)
    print(f"\r你: {text}")
    return text.strip()


# --------------------------
# 猫娘聊天（流式）
# --------------------------
def chat_with_catgirl(user_text):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += memory
    messages.append({"role": "user", "content": user_text})

    body = {
        "model": MODEL,
        "messages": messages,
        "stream": True,
        "options": {"num_predict": 256}
    }

    try:
        resp = requests.post(OLLAMA_API, json=body, stream=True)
        resp.raise_for_status()
        print("猫娘: ", end="", flush=True)
        answer = ""
        for line in resp.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            token = data.get("message", {}).get("content", "")
            answer += token
            print(token, end="", flush=True)
            if data.get("done"):
                break
        print("\n")
        return answer.strip()
    except Exception as e:
        print(f"\n猫娘: (出错了) {e}\n")
        return ""


# --------------------------
# TTS 播放（edge-tts + pygame，单线程队列）
# --------------------------
_tts_queue = queue.Queue()

# 猫娘语音：微软晓晓（活泼女声）
TTS_VOICE = "zh-CN-XiaoxiaoNeural"
pygame.mixer.init()

def _tts_worker():
    """后台线程：用 edge-tts 生成语音，pygame 播放"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    while True:
        text = _tts_queue.get()
        if text is None:
            break
        try:
            # 生成临时 mp3 文件
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                tmp_path = f.name

            # edge-tts 合成语音
            communicate = edge_tts.Communicate(text, TTS_VOICE)
            loop.run_until_complete(communicate.save(tmp_path))

            # pygame 播放
            pygame.mixer.music.load(tmp_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            pygame.mixer.music.unload()  # 释放文件句柄

            # 清理临时文件
            try:
                os.unlink(tmp_path)
            except PermissionError:
                pass  # Windows 偶发延迟，忽略
        except Exception as e:
            import traceback
            print(f"\n(TTS 出错: {e})")
            traceback.print_exc()

# 启动 TTS 后台线程
_tts_thread = threading.Thread(target=_tts_worker, daemon=True)
_tts_thread.start()

def speak(text):
    """放入播放队列（清掉旧消息，确保最新回复优先播放）"""
    while not _tts_queue.empty():
        try:
            _tts_queue.get_nowait()
        except queue.Empty:
            break
    _tts_queue.put(text)


# --------------------------
# 主循环
# --------------------------
print("=" * 50)
print("语音猫娘已启动！")
print("按 Enter 开始说话，再按 Enter 结束")
print("说「退出」或「exit」结束对话")
print("=" * 50 + "\n")

while True:
    # 1. 录音
    audio = record_audio()

    # 2. 语音识别
    user_text = transcribe(audio)
    if not user_text:
        print("(没听清，请重试)\n")
        continue

    # 3. 检查退出
    if user_text.strip().lower() in ["退出", "exit", "quit"]:
        print("猫娘: 哼，这就走了？随便你喵~")
        speak("哼，这就走了？随便你喵~")
        break

    # 4. 猫娘聊天
    answer = chat_with_catgirl(user_text)

    # 5. 保存记忆
    memory.append({"role": "user", "content": user_text})
    memory.append({"role": "assistant", "content": answer})
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)

    # 6. 猫娘语音回复（后台播放，不阻塞下一轮）
    if answer:
        speak(answer)
