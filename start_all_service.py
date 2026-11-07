import subprocess
import sys
import asyncio
import time
import os

# 配置你的启动列表
# 格式: (说明, 相对路径, 额外参数)
SERVICES = [
    # --- 第一批：MCP 服务 ---
    ("MCP 天气服务", "mcp_server/mcp_weather.py", []),
    ("MCP 票务服务", "mcp_server/mcp_ticket.py", []),
    ("MCP 行程服务", "mcp_server/mcp_trip.py", []),

    # --- 第二批：A2A 服务 ---
    ("A2A 天气服务", "a2a_server/weather_server.py", []),
    ("A2A 票务服务", "a2a_server/ticket_server.py", []),
    ("A2A 行程服务", "a2a_server/trip_server.py", []),
]


async def start_services():
    processes = []

    print(f"当前使用的 Python 解释器: {sys.executable}\n")

    print("=== 开始启动 MCP 服务 ===")
    for name, script_path, args in SERVICES:
        if "mcp_server" in script_path:
            print(f"正在启动: {name} ({script_path})...")
            # 使用当前环境的 python 启动脚本，非阻塞
            p = subprocess.Popen([sys.executable, script_path] + args)
            processes.append(p)
            await asyncio.sleep(2)  # 给 MCP 服务 2 秒的初始化时间

    print("\n=== 开始启动 A2A 服务 ===")
    for name, script_path, args in SERVICES:
        if "a2a_server" in script_path:
            print(f"正在启动: {name} ({script_path})...")
            p = subprocess.Popen([sys.executable, script_path] + args)
            processes.append(p)
            await asyncio.sleep(2)  # 给 A2A 服务一点启动时间

    print("\n所有服务启动完毕！")
    print("提示：按 Ctrl+C 可以停止所有服务。\n")
    print("-" * 40)

    # 监控进程状态
    try:
        while True:
            # 检查是否有进程意外退出
            for i, p in enumerate(processes):
                if p.poll() is not None:
                    print(f"警告：进程 {p.args[1]} 意外退出了 (退出码: {p.returncode})")
                    # 如果退出了，可以考虑是否要从列表中移除，这里保持简单，仅提示
                    processes.pop(i)
                    break  # 列表长度变了，重新进入循环检查

            await asyncio.sleep(1)

    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\n\n接收到停止信号，正在关闭所有服务...")
        for p in processes:
            if p.poll() is None:
                p.terminate()  # 终止进程
                try:
                    p.wait(timeout=3)  # 等待进程正常关闭
                except subprocess.TimeoutExpired:
                    p.kill()  # 强行杀死
        print("所有服务已关闭。")


if __name__ == "__main__":
    try:
        asyncio.run(start_services())
    except KeyboardInterrupt:
        pass