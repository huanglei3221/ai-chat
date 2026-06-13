"""
Nora Plus 配置常量
所有可调参数集中管理，修改配置无需翻代码。
"""
import os

# ============================================================
# 路径配置
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
IMAGE_DIR = os.path.join(PARENT_DIR, "猫娘nora2")

# ============================================================
# API / 模型
# ============================================================
OLLAMA_API = "http://localhost:11434/api/chat"
MODEL = "llama3.1:8b"
SENSEVOICE_MODEL = "iic/SenseVoiceSmall"

# ============================================================
# 文件路径
# ============================================================
MEMORY_FILE = os.path.join(SCRIPT_DIR, "nora2_chat_memory.json")
PERSONALITY_FILE = os.path.join(SCRIPT_DIR, "personality.json")
FAVORABILITY_FILE = os.path.join(SCRIPT_DIR, "nora2_favorability.json")
EMOTION_FILE = os.path.join(SCRIPT_DIR, "nora2_emotion.json")
LONG_TERM_MEMORY_FILE = os.path.join(SCRIPT_DIR, "long_term_memory.json")
EXTRACTION_STATE_FILE = os.path.join(SCRIPT_DIR, "nora2_extraction_state.json")

# ============================================================
# 音频 / VAD 参数
# ============================================================
SAMPLE_RATE = 16000
VAD_THRESHOLD = 0.015
VAD_SPEECH_START_FRAMES = 8
VAD_SILENCE_FRAMES = 25
VAD_CHUNK_DURATION = 0.06
VAD_MAX_LISTEN_SEC = 30

# ============================================================
# TTS 语音
# ============================================================
TTS_VOICE = "zh-CN-XiaoyiNeural"

# ============================================================
# 表情 / 情绪
# ============================================================
EXPRESSION_MAP = {
    "normal": "nora2_normal.png",
    "happy":  "nora2_happy.png",
    "angry":  "nora2_angry.png",
    "sad":    "nora2_sad.png",
    "shy":    "nora2_shy.png",
    "talk":   "nora2_talk.png",
}
VALID_EMOTIONS = {"normal", "happy", "angry", "sad", "shy"}

# ============================================================
# 时间间隔
# ============================================================
EMOTION_DECAY_INTERVAL_MS = 60_000   # 每分钟检查一次情绪衰减
EMOTION_DECAY_TIMEOUT_SEC = 600      # 10 分钟无变化 → 恢复 normal
ACTIVE_CHAT_INTERVAL_MS = 30_000     # 每 30 秒检查主动聊天
ACTIVE_CHAT_IDLE_SEC = 30            # 用户空闲超过此值触发主动搭话

# ============================================================
# 记忆 / 提取
# ============================================================
MEMORY_EXTRACTION_INTERVAL = 20      # 每 20 轮触发一次长期记忆提取
MAX_LONG_TERM_MEMORIES = 100         # 最多保留 100 条长期记忆
MAX_SAVED_EXCHANGES = 20             # 短期记忆最多保留 20 轮（40 条）
