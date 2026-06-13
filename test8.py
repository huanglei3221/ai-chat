"""
QWEN-TTS 流水线测试 — 生成/播放双线程并行，不连 Ollama。

架构：
  主线程 → 拆句，把句子喂入 gen_queue
  生成线程 → 从 gen_queue 取句子 → TTS 推理 → 音频放入 play_queue
  播放线程 → 从 play_queue 取音频 → sounddevice 播放

播放第 N 句的同时，生成线程已经在后台推理第 N+1 句。
"""

# 强制禁用 sox，用 ffmpeg/torchaudio 做音频处理（必须在 import 之前）
import os
os.environ["LIBROSA_NO_SOX"] = "1"
os.environ["TORCHAUDIO_USE_FFMPEG"] = "1"

import re
import time
import queue
import threading
import sounddevice as sd
import torch
from qwen_tts import Qwen3TTSModel


# ============================================================
# 配置
# ============================================================
QWEN_TTS_MODEL_PATH = "D:/codes/AI/modelscope/qwen/Qwen3-TTS-12Hz-0___6B-CustomVoice"
QWEN_TTS_SPEAKER = "Serena"
QWEN_TTS_LANGUAGE = "Chinese"

# ============================================================
# 测试文章（约 1000 字）
# ============================================================
TEST_ARTICLE = """
春天来了，万物复苏，大地重新披上了绿色的新装。柳树抽出了嫩绿的枝条，在微风中轻轻摇曳，仿佛是少女的长发在风中飘舞。桃花、杏花、梨花竞相开放，红的像火，粉的像霞，白的像雪，把整个山坡装点得如同人间仙境。

小溪解冻了，清澈的溪水哗哗地流淌着，像是在唱着一首欢快的歌曲。水中的鱼儿自由自在地游来游去，时而跃出水面，溅起一朵朵水花。岸边的青蛙从冬眠中醒来，呱呱地叫着，宣告着春天的到来。

田野里，农民们开始了一年的忙碌。他们翻耕土地，播下希望的种子，期待着秋天的丰收。小麦从泥土里探出了头，绿油油的一片，远远望去就像是一块巨大的绿色地毯。油菜花也开了，金黄金黄的，像是给大地铺上了一层金色的绸缎。

在城市里，人们脱下了厚重的冬装，换上了轻便的春衣。公园里到处是散步和锻炼的人，老人打着太极拳，孩子们放着风筝，年轻人骑着自行车穿梭在林荫道上。空气中弥漫着花香和泥土的清新味道，让人心旷神怡。

春天也是一个充满希望的季节。新的一年，新的开始，每个人都在为自己的梦想而努力。学生埋头苦读，准备迎接考试；上班族规划着新的职业目标；创业者怀揣着激情，踏上新的征程。春天告诉我们，无论过去经历了什么，都可以在这个季节重新开始。

这真是一个美好的季节，让我们拥抱春天，拥抱希望，拥抱生活中的每一份美好吧。
"""


def split_sentences(text: str) -> list:
    """按中文标点拆句，>80 字且含逗号时在逗号处再切一刀。"""
    raw = re.split(r'(?<=[。！？；\n])', text)
    sentences = []
    for part in raw:
        part = part.strip()
        if not part:
            continue
        if len(part) > 80 and '，' in part:
            sub_parts = re.split(r'(?<=，)', part)
            for sp in sub_parts:
                sp = sp.strip()
                if sp:
                    sentences.append(sp)
        else:
            sentences.append(part)
    return sentences


def print_gpu_memory(label: str):
    """打印当前 GPU 显存使用情况"""
    if not torch.cuda.is_available():
        print(f"[{label}] CUDA 不可用")
        return
    a = torch.cuda.memory_allocated(0) / 1024**3
    r = torch.cuda.memory_reserved(0) / 1024**3
    m = torch.cuda.max_memory_allocated(0) / 1024**3
    t = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"[{label}] allocated={a:.2f}G  reserved={r:.2f}G  peak={m:.2f}G  free≈{t-r:.2f}G")


# ============================================================
# 流水线：生成线程 + 播放线程
# ============================================================
def pipeline_tts(model, sentences):
    """
    双线程流水线：
      - 生成线程：逐句 TTS 推理 → play_queue
      - 播放线程：从 play_queue 取音频 → sd.play

    返回 (total_wall_time, total_gen_time, total_audio_dur, gen_times)
    """
    gen_queue = queue.Queue()   # 待生成的句子
    play_queue = queue.Queue()  # 已生成的音频 (wav, sr, sent_idx, gen_time, audio_dur)

    # 统计（跨线程收集）
    stats = {
        "gen_times": [],       # 每句的生成耗时
        "audio_durs": [],      # 每句的音频时长
        "gen_start": None,     # 首句生成开始时间
        "play_done": None,     # 全部播放完毕时间
    }

    # ---- 生成线程 ----
    def generator():
        while True:
            item = gen_queue.get()
            if item is None:          # 结束信号
                play_queue.put(None)
                break
            idx, sentence = item
            t0 = time.time()
            try:
                audio_wav, out_sr = model.generate_custom_voice(
                    text=sentence,
                    language=QWEN_TTS_LANGUAGE,
                    speaker=QWEN_TTS_SPEAKER,
                )
            except Exception as e:
                print(f"  !! [{idx+1}] 生成失败: {e}")
                gen_queue.task_done()
                continue
            gen_time = time.time() - t0
            wav = audio_wav[0]
            audio_dur = len(wav) / out_sr

            stats["gen_times"].append(gen_time)
            stats["audio_durs"].append(audio_dur)

            print(f"  [生成 {idx+1:2d}] {gen_time:.1f}s  |  音频 {audio_dur:.1f}s  |  "
                  f"实时率 {gen_time/audio_dur:.1f}x  |  \"{sentence[:20]}...\"")

            play_queue.put((wav, out_sr, idx, gen_time, audio_dur))
            gen_queue.task_done()

    # ---- 播放线程 ----
    def player():
        while True:
            item = play_queue.get()
            if item is None:          # 生成全部完成
                break
            wav, out_sr, idx, _gen_time, _audio_dur = item

            # 播放（阻塞到这句播完，生成线程在此期间继续跑）
            sd.play(wav, out_sr)
            sd.wait()

            print(f"  [播放 {idx+1:2d}] 完毕")
            play_queue.task_done()

        stats["play_done"] = time.time()

    # ---- 启动线程 ----
    stats["gen_start"] = time.time()
    gen_thread = threading.Thread(target=generator, daemon=True)
    play_thread = threading.Thread(target=player, daemon=True)
    gen_thread.start()
    play_thread.start()

    # ---- 主线程：喂句子 ----
    for i, s in enumerate(sentences):
        gen_queue.put((i, s))

    # 全部句子已入队，发结束信号
    gen_queue.put(None)

    # 等待两个线程结束
    gen_thread.join()
    play_thread.join()

    wall_time = stats["play_done"] - stats["gen_start"]
    total_gen = sum(stats["gen_times"])
    total_audio = sum(stats["audio_durs"])
    return wall_time, total_gen, total_audio, stats["gen_times"]


def main():
    # ---- 1. 加载前 ----
    print("=" * 60)
    print("【步骤 1】加载模型前的显存")
    print("=" * 60)
    torch.cuda.reset_peak_memory_stats()
    print_gpu_memory("加载前")

    # ---- 2. 加载模型 ----
    print()
    print("=" * 60)
    print("【步骤 2】加载 QWEN-TTS 模型")
    print("=" * 60)
    t0 = time.time()
    model = Qwen3TTSModel.from_pretrained(
        QWEN_TTS_MODEL_PATH,
        device_map="cuda:0",
        dtype=torch.float32,
    )
    print(f"加载耗时: {time.time() - t0:.1f} 秒")
    print_gpu_memory("加载后")

    # ---- 3. 拆句 ----
    article = TEST_ARTICLE.strip()
    sentences = split_sentences(article)
    total_chars = sum(len(s) for s in sentences)
    print(f"\n拆分为 {len(sentences)} 句, 共 {total_chars} 字符:\n")
    for i, s in enumerate(sentences):
        print(f"  [{i+1:2d}] {len(s):2d}字: {s}")

    # ---- 4. 流水线 TTS（生成 / 播放并行） ----
    print()
    print("=" * 60)
    print("【步骤 3】流水线 TTS — 播放的同时后台生成下一句")
    print("=" * 60)

    wall_time, total_gen, total_audio, gen_times = pipeline_tts(model, sentences)

    # ---- 5. 显存 ----
    print()
    print_gpu_memory("全部完毕")

    # ---- 6. 总结 ----
    print()
    print("=" * 60)
    print("【总结 — 双线程流水线 TTS】")
    print("=" * 60)
    print(f"  句数:             {len(sentences)}")
    print(f"  总字符:           {total_chars}")
    print(f"  首句生成耗时:     {gen_times[0]:.1f}s  ← 用户听到第一句的延迟")
    print(f"  累计生成耗时:     {total_gen:.1f}s  (各句推理时间之和)")
    print(f"  累计音频时长:     {total_audio:.0f}s ({total_audio/60:.1f} 分钟)")
    print(f"  流水线总耗时:     {wall_time:.1f}s  ← 实际墙钟时间")
    print(f"  并行节省:         {total_gen + total_audio - wall_time:.1f}s "
          f"(生成与播放重叠)")
    print(f"  模型显存:         {torch.cuda.memory_allocated(0)/1024**3:.2f} GiB")
    print(f"  峰值显存:         {torch.cuda.max_memory_allocated(0)/1024**3:.2f} GiB")


if __name__ == "__main__":
    main()
