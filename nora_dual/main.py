"""
双猫娘一键启动器 — 同时启动小猫A、小猫B 和用户语音输入

启动方式：python main.py

说明：
  - 自动启动 nora1.py、nora2.py（GUI 窗口）和 user.py（语音监听）
  - 按 Ctrl+C 或说「退出」即可全部停止
"""

import subprocess
import sys
import os
import time
import platform

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 三个子程序
PROGRAMS = [
    ("小猫A (傲娇娘)", os.path.join(SCRIPT_DIR, "nora1.py")),
    ("小猫B (温柔娘)", os.path.join(SCRIPT_DIR, "nora2.py")),
    ("用户语音输入",    os.path.join(SCRIPT_DIR, "user.py")),
]

# 日志目录
LOG_DIR = os.path.join(SCRIPT_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

processes = []
_start_time = 0  # 启动完成的时间戳


def start_all():
    """依次启动三个子程序"""
    global _start_time

    print("=" * 55)
    print("  双猫娘语音聊天 — 一键启动")
    print("=" * 55)
    print()

    is_windows = platform.system() == "Windows"

    for i, (name, script) in enumerate(PROGRAMS):
        print(f"[{i+1}/3] 启动 {name}...")

        # nora1/nora2 GUI 程序 → 输出到日志文件
        # user.py 控制台程序 → 输出留在控制台，用户需要看音量条
        is_gui = (i < 2)

        if is_gui:
            log_name = name.replace(" ", "_").replace("(", "").replace(")", "") + ".log"
            log_path = os.path.join(LOG_DIR, log_name)
            log_fp = open(log_path, "w", encoding="utf-8")
            stdout_target = log_fp
            stderr_target = log_fp
        else:
            log_path = None
            log_fp = None
            stdout_target = None   # 继承父进程 stdout → 控制台可见
            stderr_target = None

        if is_windows:
            p = subprocess.Popen(
                [sys.executable, script],
                cwd=SCRIPT_DIR,
                stdout=stdout_target,
                stderr=stderr_target,
                creationflags=subprocess.CREATE_NEW_CONSOLE,
            )
        else:
            p = subprocess.Popen(
                [sys.executable, script],
                cwd=SCRIPT_DIR,
                stdout=stdout_target,
                stderr=stderr_target,
            )

        processes.append((name, p, log_fp))
        time.sleep(1.5)

    _start_time = time.time()  # 记录启动完成时间
    print()
    print("=" * 55)
    print("  三程序已全部启动！")
    print()
    print("  - 两只猫娘的 GUI 窗口各自独立显示")
    print("  - 语音监听窗口在后台运行")
    print("  - 说「退出」结束会话，或在此按 Ctrl+C")
    print(f"  - 日志文件: {LOG_DIR}")
    print("=" * 55)
    print()


def stop_all():
    """停止所有子程序"""
    print("\n正在关闭所有程序...")
    for name, p, log_fp in processes:
        if p.poll() is None:
            print(f"  关闭 {name}...")
            p.terminate()
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print(f"    强制结束 {name}...")
                p.kill()
        if log_fp is not None:
            log_fp.close()
    print("全部已关闭。")


def check_crash(name, p, log_fp, log_path):
    """检查子进程是否异常退出，如果是则打印日志"""
    returncode = p.returncode
    if log_fp is not None and not log_fp.closed:
        log_fp.close()
    if returncode != 0:
        print(f"\n  ⚠ {name} 异常退出 (返回码: {returncode})")
        if log_path is not None:
            try:
                with open(log_path, "r", encoding="utf-8") as f:
                    log_content = f.read().strip()
                if log_content:
                    lines = log_content.splitlines()
                    for line in lines[-20:]:
                        print(f"     {line}")
            except Exception:
                pass
            print(f"  完整日志: {log_path}\n")


def main():
    global processes
    start_all()

    try:
        while True:
            for name, p, log_fp in processes:
                if p.poll() is not None:
                    # 启动后 5 秒内退出 = 异常崩溃
                    if time.time() - _start_time < 5:
                        if log_fp is not None:
                            log_name = name.replace(" ", "_").replace("(", "").replace(")", "") + ".log"
                            log_path = os.path.join(LOG_DIR, log_name)
                        else:
                            log_path = None
                        check_crash(name, p, log_fp, log_path)
                        print("可能是启动错误，请检查上面的日志。")
                        print("按 Enter 关闭其余程序...")
                        input()
                        stop_all()
                        return
                    else:
                        # 正常运行后退出 = 用户说「退出」
                        print(f"\n{name} 已退出，正在关闭其余程序...")
                        stop_all()
                        return
            time.sleep(0.5)
    except KeyboardInterrupt:
        stop_all()


if __name__ == "__main__":
    main()
