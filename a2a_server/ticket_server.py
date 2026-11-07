"""
需求：实现基于A2A统一票务服务器，处理用户的火车票、机票和演唱会票查询及预订请求

架构说明：
    本服务器是 SmartVoyage 系统中的另一个子代理（Sub-Agent），负责处理所有票务相关任务。
    它运行在独立的进程中（localhost:5006），通过 A2A（Agent2Agent）协议与主助手通信。

    与 weather_server 的区别：
    - weather_server 使用 LLM 生成 SQL，再通过 MCP 执行（Text-to-SQL 模式）
    - ticket_server 使用 LangChain Agent + MCP Tools 模式（工具调用模式）
      LLM 直接决定调用哪个 MCP 工具（火车票查询/机票查询/演唱会查询/预定），
      然后由 LangChain 的 AgentExecutor 自动完成工具调用和结果处理

    工作流程：
    1. 主助手通过 A2A 协议向本服务器发送任务（Task）
    2. 本服务器收到任务后，提取用户的自然语言查询
    3. 使用 LangChain Agent（基于工具调用的 Agent）处理查询：
       a. LLM 分析用户输入，决定调用哪个 MCP 工具
       b. LangChain 自动调用 MCP Server 的工具（端口 8001）
       c. 工具返回结果后，LLM 将结果格式化为友好的中文回复
    4. 将结果返回给主助手

    涉及的关键技术：
    - LangChain Agent: 一种让 LLM 自主选择和使用工具的框架
    - Tool Calling Agent: 一种 Agent 类型，LLM 以结构化格式调用工具
    - MCP Tools: MCP Server 提供的工具，本系统中包括火车票查询、机票查询等
    - AgentExecutor: LangChain 的执行器，负责运行 Agent 循环（思考→调用工具→处理结果→继续）

    MCP Server（端口 8001）提供的工具包括：
    - 火车票查询工具
    - 机票查询工具
    - 演唱会门票查询工具
    - 火车票预定工具
    - 机票预定工具
    - 演唱会门票预定工具
"""

# ==================== 逻辑流程图 ====================
#
#  ┌─────────────────────────────────────────────────────────────────┐
#  │                        主程序入口 (__main__)                      │
#  │  1. 创建 TicketServer 实例                                        │
#  │    │
#  │  2. run_server(ticket_server, host, port=5006)                   │
#  └──────────────────────────────┬──────────────────────────────────┘
#                                 ▼
#  ┌─────────────────────────────────────────────────────────────────┐
#  │                     模块加载 & 全局初始化                            │
#  │  ┌─────────────────────┐    ┌──────────────────────────────┐    │
#  │  │ 1. 初始化 LLM        │    │ 2. 定义 AgentCard             │    │
#  │  │    ChatOpenAI(...)   │    │    - name: TicketAssistant    │    │
#  │  │    (model/base_url/  │    │    - 6 个 AgentSkill:         │    │
#  │  │     api_key/temp)    │    │      查询: 火车/机票/演唱会      │    │
#  │  └─────────────────────┘    │      预定: 火车/机票/演唱会      │    │
#  │                              └──────────────┬───────────────┘    │
#  └─────────────────────────────────────────────┼───────────────────┘
#                                                ▼
#  ┌─────────────────────────────────────────────────────────────────┐
#  │                    TicketServer.__init__()                        │
#  │    super().__init__(agent_card=agent_card)                       │
#  └──────────────────────────────┬──────────────────────────────────┘
#                                 ▼  (主助手通过 A2A 发送 Task)
#  ┌─────────────────────────────────────────────────────────────────┐
#  │                    TicketServer.handle_task(task)                 │
#  │                                                                 │
#  │  5.1 提取用户查询: query = task.message["content"]                │
#  │        │                                                        │
#  │        ▼                                                        │
#  │  5.2 获取 MCP 工具集: tools = load_mcp_tools    │
#  │        │                                                        │
#  │        ▼                                                        │
#  │  5.3 构建 Prompt: ChatPromptTemplate                              │
#  │        │  system: 角色说明 + 参数校验要求 + 当前日期                  │
#  │        │  human:  {input}                                        │
#  │        │  placeholder: {agent_scratchpad}                        │
#  │        ▼                                                        │
#  │  5.4 构建 Agent:                                                  │
#  │        create_agent(llm, tools, prompt)              │                 │
#  │        │                                                        │
#  │        ▼                                                        │
#  │  5.5 执行 Agent 循环: agent_executor.invoke()                      │
#  │        ┌──────────────────────────────────────┐                  │
#  │        │  LLM 分析输入 → 选择 MCP 工具           │                  │
#  │        │  → 调用工具(端口 8001) → 获取结果        │                  │
#  │        │  → 格式化为中文回复                     │                  │
#  │        └──────────────────────────────────────┘                  │
#  │        │                                                        │
#  │        ▼                                                        │
#  │  5.6 结果分支处理:                                                  │
#  │        │                                                        │
#  │        ├─ "缺少" in output                                        │
#  │        │   → TaskState.INPUT_REQUIRED                             │
#  │        │   → 返回追问信息                                          │
#  │        │                                                        │
#  │        ├─ 正常输出                                                │
#  │        │   → TaskState.COMPLETED                                  │
#  │        │   → artifacts = [格式化结果]                              │
#  │        │                                                        │
#  │        └─ Exception                                             │
#  │            → TaskState.FAILED                                     │
#  │            → 错误信息                                             │
#  │                                                                 │
#  └──────────────────────────────┬──────────────────────────────────┘
#                                 ▼
#                          返回 task 给主助手
#
# ==================== 导入依赖 ====================
import asyncio  # 异步IO库

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from python_a2a import A2AServer, run_server, AgentCard, AgentSkill, TaskStatus, TaskState

from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.tools import load_mcp_tools
#load_mcp_tools --- 获取MCPServer的工具列表 ，转成StructureTool
from langchain.agents import create_agent

from config import Config
from create_logger import logger
from datetime import datetime
import pytz

conf = Config()

# 初始化LLM
llm = ChatOpenAI(
    model=conf.model_name,
    base_url=conf.base_url,
    api_key=conf.api_key,
    temperature=conf.temperature
)

# 票务MCP 服务器运行在 8001 端口
MCP_URL = "http://127.0.0.1:8001/mcp"

async def query_tickets(conversation: str) -> dict:
    """通过 LangChain Agent + MCP Tools 执行票务查询或预定"""
    try:
        #客户端连接服务端获取读写流和回调函数
        async with streamablehttp_client(MCP_URL) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await load_mcp_tools(session)
                # 获取当前日期，注入到系统 Prompt 中（用户可能说"明天"、"后天"等相对时间）
                current_date = datetime.now(pytz.timezone('Asia/Shanghai')).strftime('%Y-%m-%d')  # 获取北京时间当前日期

                system_prompt = f"""你是一个票务查询与预定助手，能够调用工具来查询火车票、机票、演唱会票信息，以及进行票务预定。
                你需要仔细分析用户的问题，从问题中提取工具需要的参数，然后调用对应的查询或预定工具。
                
                可用工具说明：
                - query_train: 查询火车票，需要 departure_city(出发城市)、arrival_city(到达城市)、date(日期YYYY-MM-DD)，seat_type(座位类型，可选)
                - query_flight: 查询机票，需要 departure_city(出发城市)、arrival_city(到达城市)、date(日期YYYY-MM-DD)，cabin_type(舱位类型，可选)
                - query_concert: 查询演唱会票，需要 city(城市)、date(日期YYYY-MM-DD)，artist(艺人姓名，可选，不填则查该城市所有演唱会)，ticket_type(票类型，可选)
                - order_train / order_flight / order_concert: 预定对应的票务
                
                重要规则：
                1. 用户可能使用相对时间，如"明天"、"后天"、"今天"等，请根据当前日期转换为具体日期（YYYY-MM-DD格式）。
                2. 如果用户缺少必要参数（如火车票缺少出发/到达城市、火车票/机票缺少日期等），必须向用户追问，不能自己编撰参数。演唱会查询时，如果用户没有指定艺人，可以不填artist参数，查询该城市所有演唱会。
                3. 如果用户同时查询多种票务（如火车票+机票+演唱会票），请逐一调用对应的工具查询。
                4. 查询到结果后，请用清晰的中文格式化输出票务信息：
                   - 火车票/机票：出发城市、到达城市、出发时间、到达时间、车次/航班号、座位/舱位类型、价格、余票数量
                   - 演唱会票：艺人、城市、场馆、开始时间、票类型、价格、余票数量
                5. 如果未查到数据，请回复"未找到相关票务数据，请确认或修改查询条件。"
                
                当前日期是{current_date}。"""

                agent = create_agent(
                    #大模型--具有Function call
                    model=llm,
                    #工具列表
                    tools=tools,
                    #系统提示词
                    system_prompt=system_prompt)
                #response --- json字符串
                response = await agent.ainvoke({  # ainvoke 为异步调用，执行 Agent 循环
                    "messages": [{"role": "user", "content": conversation}]  # 用户输入的查询内容
                })
                return {"status": "success", "message": response["messages"][-1].content}

    except ExceptionGroup as eg:

        first_exc = eg.exceptions[0] if eg.exceptions else eg

        logger.error(f"票务 MCP 查询出错：{first_exc}")

        return {"status": "error", "message": f"票务 MCP 查询出错：{first_exc}"}

    except BaseException as e:

        logger.error(f"票务 MCP 查询出错：{str(e)}")

        return {"status": "error", "message": f"票务 MCP 查询出错：{str(e)}"}
#1 构建AgentCard
agent_card = AgentCard(
    #票务子代理的名字
    name="TicketAssistant",
    description="基于 LangChain 提供票务查询和预定服务的统一助手",
    url="http://localhost:5006",
    version="1.0.0",
    capabilities={"streaming": True, "memory": True},
    skills=[
        AgentSkill(
            name="query train tickets",
            description="查询火车票信息",
            examples=["北京到上海的高铁 2025-10-28"]
        ),
        AgentSkill(
            name="query flight tickets",
            description="查询航班机票信息",
            examples=["上海到北京的机票 10月28日"]
        ),
        AgentSkill(
            name="query concert tickets",
            description="查询演唱会门票信息",
            examples=["刀郎北京演唱会门票"]
        ),
        AgentSkill(
            name="order train tickets",
            description="预定火车票",
            examples=["帮我订一张G1234二等座"]
        ),
        AgentSkill(
            name="order flight tickets",
            description="预定机票",
            examples=["订一张经济舱机票"]
        ),
        AgentSkill(
            name="order concert tickets",
            description="预定演唱会票",
            examples=["买两张刀郎演唱会VIP票"]
        )
    ]
)
# 2 继承A2AServer，调用父类的构造方法
class TicketQueryServer(A2AServer):
    """票务查询和预定 A2A 服务器"""

    def __init__(self):
        super().__init__(agent_card=agent_card)
    # 3 通过handle_task实现具体业务逻辑
    def handle_task(self, task):
        content = (task.message or {}).get("content", {})
        conversation = content.get("text", "") if isinstance(content, dict) else ""
        logger.info(f"用户查询: {conversation}")

        try:
            #ticket_result --字典
            ticket_result = asyncio.run(query_tickets(conversation))
            logger.info(f"MCP 查询返回: {ticket_result}")

            if ticket_result.get("status") == "success":
                result_text = ticket_result.get("message", "")

                # 检查是否是追问消息（需要用户补充查询条件）
                # 使用更全面的关键词列表来判断是否是追问
                follow_up_keywords = [
                    # 明确的追问关键词
                    "请提供", "请确认", "请告诉我", "请补充",
                    "需要您提供", "需要您补充", "缺少", "未提供",
                    "无法查询", "无法确定", "请指定", "请输入",
                    # 询问缺失信息的模式
                    "哪个城市", "什么城市", "哪一天", "哪个日期",
                    "具体日期", "出发城市", "到达城市", "没有提供",
                    "信息不足", "暂未找到", "未找到",
                    # 询问用户需求的模式（LLM 在追问时常用）
                    "请问您需要", "请问您想", "请问您要",
                    "请告诉我您的", "请说明", "需要以下信息",
                    "必要信息", "基本信息", "具体需求",
                    # 表示无法执行的情况
                    "无法为您", "暂时无法", "需要先"
                ]
                is_follow_up = any(keyword in result_text for keyword in follow_up_keywords)

                if is_follow_up:
                    # 需要用户补充信息
                    task.status = TaskStatus(
                        state=TaskState.INPUT_REQUIRED,
                        message={"role": "agent", "content": {"text": result_text}}
                    )
                else:
                    # 查询成功，返回票务数据
                    task.artifacts = [{"parts": [{"type": "text", "text": result_text}]}]
                    task.status = TaskStatus(state=TaskState.COMPLETED)

            elif ticket_result.get("status") == "error":
                task.status = TaskStatus(
                    state=TaskState.FAILED,
                    message={"role": "agent", "content": {"text": ticket_result.get("message", "查询失败，请重试。")}}
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
    ticket_server = TicketQueryServer()
    print("\n=== 服务器信息 ===")
    print(f"名称: {ticket_server.agent_card.name}")
    print(f"描述: {ticket_server.agent_card.description}")
    print("\n技能:")
    for skill in ticket_server.agent_card.skills:
        print(f"- {skill.name}: {skill.description}")
    #4 启动
    run_server(ticket_server, host="127.0.0.1", port=5006)
