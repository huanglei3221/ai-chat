import requests
import json
import os

# --------------------------
# 配置部分
# --------------------------
OLLAMA_API = "http://localhost:11434/api/chat"
MODEL = "llama3.1:8b"
MEMORY_FILE = "chat_memory.json"
PERSONALITY_FILE = "personality.json"

# --------------------------
# 从配置文件自动生成 System Prompt
# --------------------------
def build_system_prompt(config):
    name = config["name"]
    personality = config["personality"]
    speech = config["speech_style"]

    # 性格映射表：把关键词展开成具体的行为指令
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

    # 生成性格描述
    traits_text = ""
    for t in personality:
        desc = trait_map.get(t, t)
        traits_text += f"- {t}：{desc}\n"

    # 生成说话风格描述
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
    print(f"错误：找不到人格配置文件 {PERSONALITY_FILE}")
    exit(1)

with open(PERSONALITY_FILE, "r", encoding="utf-8") as f:
    personality_config = json.load(f)

SYSTEM_PROMPT = build_system_prompt(personality_config)
print(f"已加载人格：{personality_config['name']}")
print(f"性格：{', '.join(personality_config['personality'])}")
print(f"说话风格：{', '.join(personality_config['speech_style'])}")
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
# 聊天循环
# --------------------------
print("猫娘已启动，你可以开始聊天（输入 exit 退出）：\n")

while True:
    user_input = input("你: ")
    if user_input.strip().lower() in ["exit", "quit"]:
        break

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += memory
    messages.append({"role": "user", "content": user_input})

    body = {
        "model": MODEL,
        "messages": messages,
        "stream": True,
        "options": {
            "num_predict": 256  # 最大生成 token 数，默认 128 太短
        }
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
    except Exception as e:
        print("\n猫娘: (出错了)", e)
        continue

    memory.append({"role": "user", "content": user_input})
    memory.append({"role": "assistant", "content": answer})

    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)
