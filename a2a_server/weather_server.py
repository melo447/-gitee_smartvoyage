"""
需求：实现基于A2A的天气查询服务器，处理用户的天气查询请求并返回结果

架构说明：
    本服务器是 SmartVoyage 系统中的一个子代理（Sub-Agent），负责处理天气查询任务。
    它运行在独立的进程中（localhost:5005），通过 A2A（Agent2Agent）协议与主助手通信。

    1. 主助手通过 A2A 协议向本服务器发送任务（Task）
    2. 本服务器收到任务后，提取用户的自然语言查询
    3. 使用 LangChain Agent（基于工具调用的 Agent）处理查询：
       a. LLM 分析用户输入，从自然语言中提取参数（城市、日期等）
       b. 调用 MCP Server 的参数化工具（query_weather）
       c. MCP Server 内部根据参数拼接 SQL 并执行查询
       d. 工具返回结果后，LLM 将结果格式化为友好的中文回复
    4. 将结果返回给主助手
    涉及的关键技术：
    - LangChain Agent: 让 LLM 自主选择和使用工具的框架
    - Tool Calling Agent: LLM 以结构化格式调用工具
    - MCP Tools: MCP Server 提供的参数化工具（city、start_date、end_date）
    - AgentExecutor: 负责运行 Agent 循环（思考→调用工具→处理结果→继续）
"""

# ==================== 导入依赖 ====================
import asyncio#异步库
from mcp import ClientSession #会话类
from mcp.client.streamable_http import streamablehttp_client #streamable-http传输方式的客户端
from python_a2a import A2AServer, run_server, AgentCard, AgentSkill, TaskStatus, TaskState
#run_server -- 启动A2AServer服务
#AgentCard -- 子代理的名片
#AgentSkill --- 子代理的技能
#TaskStatus --- 状态管理类
#TaskState --- 状态枚举
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.tools import load_mcp_tools
#load_mcp_tools --- 获取mcpServer工具列表、把工具转成StructuredTool对象
from langchain.agents import create_agent
from config import Config
from create_logger import logger
from datetime import datetime
import pytz #时区

conf = Config()

# 初始化LLM
"""
智能选择工具
"""
llm = ChatOpenAI(
    model=conf.model_name,
    base_url=conf.base_url,
    api_key=conf.api_key,
    temperature=conf.temperature
)

# MCP 服务器运行在 8002 端口
MCP_URL = "http://127.0.0.1:8002/mcp"
#
async def query_weather(conversation: str) -> dict:
    """通过 LangChain Agent + MCP Tools 执行天气查询"""
    try:
        # 连接 MCP Server（端口 8002），建立通信通道
        #mcp客户端连接mcp服务端，获取读写流和回调
        async with streamablehttp_client(MCP_URL) as (read, write, _):
            async with ClientSession(read, write) as session:
                #会话初始化
                await session.initialize()

                # 从 MCP 会话中加载所有可用工具
                #tools --[StructureTool]
                tools = await load_mcp_tools(session)

                # 获取当前日期，注入到系统 Prompt 中（用户可能说"明天"、"后天"等相对时间）
                current_date = datetime.now(pytz.timezone('Asia/Shanghai')).strftime('%Y-%m-%d')  # 获取北京时间当前日期

                # 定义 Agent 的系统 Prompt，指引 LLM 提取参数并调用天气工具
                # create_agent 接受 system_prompt 字符串参数，使用 f-string 直接嵌入当前日期
                system_prompt = f"""你是一个天气查询助手，能够调用工具来查询天气信息。
                你需要仔细分析用户的问题，从问题中提取工具需要的参数（城市、日期等），然后调用对应的查询工具。
                如果用户提供的信息不足以提取到调用工具的所有必要参数，则向用户追问，以获取该信息。不能自己编撰参数。
                注意：
                - 用户可能使用相对时间，如"明天"、"后天"、"今天"、"未来3天"等，请根据当前日期转换为具体日期（YYYY-MM-DD格式）。
                - 如果用户只给了城市没给日期，需要追问日期。
                - 如果用户只给了日期没给城市，需要追问城市。
                查询到结果后，请用清晰的中文格式化输出天气信息，包括城市、日期、天气状况、温度、湿度、风向、降水量等。
                如果未查到数据，请回复"未找到相关天气数据，请确认或修改查询条件。"
                当前日期是{current_date}。"""
                # 创建基于工具调用的 Agent
                agent = create_agent(
                    #大模型 --- 具有function call功能
                    model=llm,
                    #工具列表
                    tools=tools,
                    #系统提示词
                    system_prompt=system_prompt)
                # 执行 Agent
                response = await agent.ainvoke({  # ainvoke 为异步调用，执行 Agent 循环
            "messages": [{"role": "user", "content": conversation}]  # 用户输入的查询内容
        })

                return {"status": "success", "message": response["messages"][-1].content}

    except ExceptionGroup as eg:

        first_exc = eg.exceptions[0] if eg.exceptions else eg

        logger.error(f"天气 MCP 查询出错：{first_exc}")

        return {"status": "error", "message": f"天气 MCP 查询出错：{first_exc}"}

    except BaseException as e:

        logger.error(f"天气 MCP 查询出错：{str(e)}")

        return {"status": "error", "message": f"天气 MCP 查询出错：{str(e)}"}

#1 构建AgentCard，所有信息用于展示A2A客户端
agent_card = AgentCard(
    #子代理的名字
    name="WeatherQueryAssistant",
    description="基于LangChain提供天气查询服务的助手",
    #子代理的访问地址
    url="http://localhost:5005",
    version="1.0.0",
    capabilities={"streaming": True, "memory": True},
    #子代理的技能
    skills=[
        AgentSkill(
            name="query weather",
            description="查询天气数据，支持自然语言输入",
            examples=["北京 2025-07-30 天气", "上海未来3天天气如何", "明天天气怎么样"]
        )
    ]
)
#2 继承A2AServer,调用父类的构造方法
class WeatherQueryServer(A2AServer):
    """天气查询 A2A 服务器"""

    def __init__(self):
        super().__init__(agent_card=agent_card)
    #3 实现具体的业务逻辑 接收Task 对象，返回的也是Task对象
    def handle_task(self, task):
        # 提取输入用户的多轮对话内容
        content = (task.message or {}).get("content", {})
        conversation = content.get("text", "") if isinstance(content, dict) else ""
        logger.info(f"用户查询: {conversation}")

        try:
            # 调用 MCP 查询，在同步函数中调用异步函数，asyncio.run(异步函数)
            # weather_result-- 工具的执行结果--Json字符串
            weather_result = asyncio.run(query_weather(conversation))
            logger.info(f"MCP 查询返回: {weather_result}")

            # 根据结果设置任务状态
            if weather_result.get("status") == "success":
                result_text = weather_result.get("message", "")

                # 检查是否是追问消息（需要用户补充查询条件）
                # 使用更全面的关键词列表来判断是否是追问
                follow_up_keywords = [
                    "请提供", "请确认", "请告诉我", "请补充",
                    "需要您提供", "需要您补充", "缺少", "未提供",
                    "无法查询", "无法确定", "请指定", "请输入",
                    "哪个城市", "什么城市", "哪一天", "哪个日期",
                    "具体日期", "日期范围", "没有提供", "信息不足"
                ]
                is_follow_up = any(keyword in result_text for keyword in follow_up_keywords)

                if is_follow_up:
                    # 需要用户补充信息
                    task.status = TaskStatus(
                        state=TaskState.INPUT_REQUIRED,
                        message={"role": "agent", "content": {"text": result_text}}
                    )
                else:
                    # 查询成功，返回天气数据
                    task.artifacts = [{"parts": [{"type": "text", "text": result_text}]}]
                    task.status = TaskStatus(state=TaskState.COMPLETED)

            elif weather_result.get("status") == "error":
                task.status = TaskStatus(
                    state=TaskState.FAILED,
                    message={"role": "agent", "content": {"text": weather_result.get("message", "查询失败，请重试。")}}
                )
            else:
                task.status = TaskStatus(
                    state=TaskState.FAILED,
                    message={"role": "agent", "content": {"text": "查询失败，请重试或提供更多细节。"}}
                )

            return task

        except Exception as e:
            logger.error(f"查询失败: {str(e)}")
            task.status = TaskStatus(
                state=TaskState.FAILED,
                message={"role": "agent", "content": {"text": f"查询失败: {str(e)} 请重试。"}}
            )
            return task
#
if __name__ == "__main__":
    weather_server = WeatherQueryServer()
    print("\n=== 服务器信息 ===")
    print(f"名称: {weather_server.agent_card.name}")
    print(f"描述: {weather_server.agent_card.description}")
    print("\n技能:")
    for skill in weather_server.agent_card.skills:
        print(f"- {skill.name}: {skill.description}")
    #4 启动服务
    run_server(weather_server, host="127.0.0.1", port=5005)