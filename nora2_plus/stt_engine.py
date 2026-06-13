"""
STTEngine —— SenseVoice 语音识别封装
"""
import re
import numpy as np
from funasr import AutoModel


class STTEngine:
    """SenseVoice 语音识别引擎"""

    def __init__(self, model_name="iic/SenseVoiceSmall", device="cuda:0"):
        print(f"加载语音识别模型 ({model_name})...")
        self._model = AutoModel(
            model=model_name,
            device=device,
            disable_update=True,
        )
        print("语音识别模型加载完成\n")

    def transcribe(self, audio: np.ndarray) -> str:
        """语音 → 文字，返回识别文本或空字符串"""
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        try:
            result = self._model.generate(
                input=audio,
                language="zh",
            )
            if result and len(result) > 0:
                text = result[0].get("text", "").strip()
                # 清洗 FunASR 输出的标签和特殊字符
                text = re.sub(r'<\|[^|]+\|>', '', text).strip()
                text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
                text = text.replace('�', '')
            else:
                text = ""
        except Exception:
            text = ""
        return text
