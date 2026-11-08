"""
需求：实现基于A2A的行程管家服务器，处理用户的租车、旅游团、保险查询及预订请求

架构说明：
    本服务器是 SmartVoyage 系统中的又一个子代理（Sub-Agent），负责处理所有行程相关任务。
    它运行在独立的进程中（localhost:5007），通过 A2A（Agent2Agent）协议与主助手通信。

    与 ticket_server.py 和 weather_server.py 保持一致的架构模式：
    - 使用 LangChain Agent + MCP Tools 模式
    - LLM 从自然语言中提取参数（城市、日期、类型等）
    - 调用 MCP Server 的参数化工具（端口 8003）
    - MCP Server 返回结果后，LLM 将结果格式化为友好的中文回复

    工作流程：
    1. 主助手通过 A2A 协议向本服务器发送任务（Task）
    2. 本服务器收到任务后，提取用户的自然语言查询
    3. 使用 LangChain Agent 处理查询：
       a. LLM 分析用户输入，决定调用哪个 MCP 工具
       b. LangChain 自动调用 MCP Server 的工具（端口 8003）
       c. 工具返回结果后，LLM 将结果格式化为友好的中文回复
    4. 将结果返回给主助手

    MCP Server（端口 8003）提供的工具包括：
    - 租车查询工具、旅游团查询工具、保险查询工具
    - 租车预订工具、旅游团报名工具、保险购买工具

架构图：

+─────────────────────────────────────────────────────────────────────────────+
│                          SmartVoyage 主助手 (Main Agent)                     │
│                        通过 A2A 协议分派任务到子代理                           │
+──────────────────────────────────┬──────────────────────────────────────────+
                                   │ A2A 协议 (Task)
                                   ▼
+─────────────────────────────────────────────────────────────────────────────+
│                        TripQueryServer (A2AServer)  :5007                    │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  AgentCard: TripAssistant                                              │  │
│  │  Skills: query/order car rental, tour group, insurance                │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                     │                                       │
│  ┌──────────────────────────────────▼──────────────────────────────────┐   │
│  │  handle_task(task)                                                  │   │
│  │  1. 提取用户自然语言查询                                              │   │
│  │  2. asyncio.run(query_trip(conversation))                           │   │
│  │  3. 设置 TaskStatus: COMPLETED / INPUT_REQUIRED / FAILED             │   │
│  └──────────────────────────────────┬──────────────────────────────────┘   │
│                                     │                                       │
│  ┌──────────────────────────────────▼──────────────────────────────────┐   │
│  │  query_trip(conversation)                                            │   │
│  │  ┌────────────────────────────────────────────────────────────────┐ │   │
│  │  │  ChatOpenAI (LLM)                      │ │   │
│  │  │  create_agent()                    │ │   │
│  │  │  ainvoke({"input": conversation, "current_date": ...})          │ │   │
│  │  └────────────────────────┬───────────────────────────────────────┘ │   │
│  │                           │ 选择工具 & 参数                          │   │
│  │  ┌────────────────────────▼───────────────────────────────────────┐ │   │
│  │  │  load_mcp_tools(MCP_URL)                                     │ │   │
│  │  │  MCP Server → LangChain Tools 桥接                              │ │   │
│  │  └───────────────────────────────────────────────────────────────┘ │   │
│  └───────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────┬────────────────────────────────────────┘
                                   │ MCP 协议
                                   ▼
+─────────────────────────────────────────────────────────────────────────────+
│                      MCP Server  :8003                                      │
│                                                                             │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                     │
│   │ 租车工具      │  │ 旅游团工具    │  │ 保险工具      │                     │
│   │ - query_car  │  │ - query_tour │  │ - query_ins  │                     │
│   │ - order_car  │  │ - order_tour │  │ - order_ins  │                     │
│   └──────────────┘  └──────────────┘  └──────────────┘                     │
│                                                                             │
│   ┌───────────────────────────────────────────────────────────────────┐    │
│   │  数据层 (Database / 向量库)                                           │    │
│   │  租车数据 / 旅游团数据 / 保险数据                                    │    │
│   └───────────────────────────────────────────────────────────────────┘    │
+─────────────────────────────────────────────────────────────────────────────+

请求/响应流向：
    用户 → 主助手 → [A2A] → TripQueryServer.handle_task()
                                  ↓
                          query_trip() → LangChain Agent (LLM)
                                  ↓
                          MCP Tools → MCP Server :8003
                                  ↓
                          MCP Server 返回工具结果
                                  ↓
                          LLM 格式化为中文回复
                                  ↓
                          handle_task() 设置 TaskStatus
                                  ↓
                          主助手 → 用户
"""

# ==================== 导入依赖 ====================
import asyncio

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from python_a2a import A2AServer, run_server, AgentCard, AgentSkill, TaskStatus, TaskState

from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.tools import load_mcp_tools
# load_mcp_tools -- 获取MCPserver端的工具列表，装车StructureTool
from langchain.agents import create_agent

from config import Config
from create_logger import logger
from datetime import datetime
import pytz

conf = Config()

#
llm = ChatOpenAI(
    model=conf.model_name,
    base_url=conf.base_url,
    api_key=conf.api_key,
    temperature=conf.temperature
)

# MCP 服务器运行在 8003 端口
MCP_URL = "http://127.0.0.1:8003/mcp"
#
async def query_trip(conversation: str) -> dict:
    """通过 LangChain Agent + MCP Tools 执行行程查询或预订"""
    try:
        #mcpk客户端连接mcp服务端获取读写流和回调
        async with streamablehttp_client(MCP_URL) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await load_mcp_tools(session)

                # 获取当前日期，注入到系统 Prompt 中（用户可能说"明天"、"后天"等相对时间）
                current_date = datetime.now(pytz.timezone('Asia/Shanghai')).strftime('%Y-%m-%d')

                # 定义行程管家 Agent 的系统 Prompt
                system_prompt = f"""你是一个行程管家助手，负责调用工具帮助用户查询和预订租车、旅游团、保险服务。

                ## 核心原则
                **你的首要任务是：从用户输入中提取参数，立即调用对应工具，然后展示结果。**
                只有在缺少必要参数时才追问用户。
                
                ## 工具与参数
                
                ### 租车服务
                - **query_car_rental(pickup_city, date, return_city=None, car_type=None)**
                  - pickup_city: 取车城市（必填）
                  - date: 租车日期 YYYY-MM-DD（必填）
                  - return_city: 还车城市（可选，默认等于取车城市）
                  - car_type: 车型（可选：经济型/SUV/豪华型/MPV）
                - **order_car_rental(date, car_type, number)**: 预订租车
                
                ### 旅游团服务（支持语义搜索）
                - **query_tour_group(query_text, city=None)**
                  - query_text: 用户描述（必填）- 把用户的需求转化为描述性文本
                  - city: 城市过滤（可选）
                - **order_tour_group(date, tour_name, number)**: 报名旅游团
                
                ### 保险服务
                - **query_insurance(insurance_type=None)**
                  - insurance_type: 保险类型（可选：综合型/意外型/医疗型/境外型）
                  - 不指定则查询所有类型
                - **order_insurance(insurance_type, date, number)**: 购买保险
                
                ## 执行规则
                
                1. **立即执行**：用户提供了可提取的参数时，立即调用工具，不要再次确认
                2. **时间转换**：将"明天""后天""下周"等转换为具体日期（当前日期：{current_date}）
                3. **默认值**：
                   - 租车数量默认为1
                   - 还车城市默认等于取车城市
                   - 保险份数默认为1
                4. **追问场景**（仅在缺少必要参数时）：
                   - 租车：缺少取车城市或日期
                   - 旅游团：用户没有描述想要的旅行体验
                   - 保险：一般不需要追问，可直接查询所有类型
                5. **结果展示**：查询到结果后，用清晰的中文格式展示给用户
                
                ## 示例
                
                用户说"帮我查询11月8日北京的租车"
                → 提取：pickup_city=北京, date=2026-11-08
                → 立即调用 query_car_rental(pickup_city="北京", date="2026-11-08")
                
                用户说"我想去有雪山的地方旅游"
                → 提取：query_text=想看雪山的地方
                → 立即调用 query_tour_group(query_text="想看雪山的地方")
                
                用户说"有没有旅行保险"
                → 立即调用 query_insurance() 查询所有类型
                
                **记住：能调用工具时就调用工具，不要只回复文字！**"""

                agent = create_agent(
                    model=llm,
                    tools=tools,
                    system_prompt=system_prompt)
                #response --json字符串
                response = await agent.ainvoke({  # ainvoke 为异步调用，执行 Agent 循环
            "messages": [{"role": "user", "content": conversation}]  # 用户输入的查询内容
        })

                return {"status": "success", "message": response["messages"][-1].content}

    except ExceptionGroup as eg:
        first_exc = eg.exceptions[0] if eg.exceptions else eg
        logger.error(f"行程 MCP 查询出错：{first_exc}")
        return {"status": "error", "message": f"行程 MCP 查询出错：{first_exc}"}
    except BaseException as e:
        logger.error(f"行程 MCP 查询出错：{str(e)}")
        return {"status": "error", "message": f"行程 MCP 查询出错：{str(e)}"}
# 1 创建AgentCard
agent_card = AgentCard(
    #子代理的名字
    name="TripAssistant",
    description="基于 LangChain 提供行程管家服务的统一助手，支持租车、旅游团、保险的查询与预订",
    url="http://localhost:5007",
    version="1.0.0",
    capabilities={"streaming": True, "memory": True},
    skills=[
        AgentSkill(name="query car rental", description="查询租车信息", examples=["租车 北京 上海 2025-08-01"]),
        AgentSkill(name="query tour group", description="通过语义搜索查询旅游团信息", examples=["想看雪山的地方"]),
        AgentSkill(name="query insurance", description="查询旅行保险产品", examples=["旅行保险"]),
        AgentSkill(name="order car rental", description="预订租车服务", examples=["预订租车 2025-08-01 SUV 1辆"]),
        AgentSkill(name="order tour group", description="报名旅游团", examples=["报名丽江三日游 2人"]),
        AgentSkill(name="order insurance", description="购买旅行保险", examples=["购买综合型保险"])
    ]
)
#2 继承A2AServer
class TripQueryServer(A2AServer):
    """行程管家 A2A 服务器"""

    def __init__(self):
        super().__init__(agent_card=agent_card)
#3 通过handle_task 来实现具体的业务逻辑
    def handle_task(self, task):
        # 从 task.message 中提取用户查询文本
        # 支持两种格式：content.text 和 parts[0].text
        message = task.message or {}
        conversation = ""

        # 尝试从 content.text 提取
        content = message.get("content", {})
        if isinstance(content, dict) and content.get("text"):
            conversation = content.get("text", "")
        # 尝试从 parts 数组提取（Google A2A 格式）
        elif "parts" in message:
            parts = message.get("parts", [])
            for part in parts:
                if isinstance(part, dict) and part.get("type") == "text":
                    conversation = part.get("text", "")
                    break

        logger.info(f"用户查询: {conversation}")

        try:
            #trip_result工具的执行结果 -- 字典
            trip_result = asyncio.run(query_trip(conversation))
            logger.info(f"MCP 查询返回: {trip_result}")

            if trip_result.get("status") == "success":
                result_text = trip_result.get("message", "")

                # 检查是否是追问消息（需要用户补充查询条件）
                # 使用更全面的关键词列表来判断是否是追问
                follow_up_keywords = [
                    # 明确的追问关键词
                    "请提供", "请确认", "请告诉我", "请补充",
                    "需要您提供", "需要您补充", "缺少", "未提供",
                    "无法查询", "无法确定", "请指定", "请输入",
                    # 询问缺失信息的模式
                    "哪个城市", "什么城市", "哪一天", "哪个日期",
                    "具体日期", "取车城市", "没有提供",
                    # 注意："未找到/暂未找到"属于查询无结果(no_data)，不是缺少参数，
                    # 由下方 is_no_data 分支单独处理
                    "信息不足",
                    # 询问用户需求的模式
                    "请问您需要", "请问您想", "请问您要",
                    "请告诉我您的", "请说明", "需要以下信息",
                    "必要信息", "基本信息", "具体需求",
                    # 表示无法执行的情况
                    "无法为您", "暂时无法", "需要先"
                ]
                is_follow_up = any(keyword in result_text for keyword in follow_up_keywords)
                # 先判定"查无数据"：查询已经执行但没有结果，属于 no_data；
                # 不能因为回复里带"请确认"等追问字样就误判成缺少参数
                no_data_keywords = ["未找到", "暂未找到", "没有找到", "暂无相关", "无相关"]
                is_no_data = any(keyword in result_text for keyword in no_data_keywords)

                if is_no_data:
                    # 查询完成但无数据，交给主服务如实告知并引导调整条件
                    task.artifacts = [{"parts": [{"type": "text", "text": result_text}]}]
                    task.status = TaskStatus(state=TaskState.COMPLETED)
                elif is_follow_up:
                    # 需要用户补充信息
                    task.status = TaskStatus(
                        state=TaskState.INPUT_REQUIRED,
                        message={"role": "agent", "content": {"text": result_text}}
                    )
                else:
                    # 查询成功，返回结果
                    task.artifacts = [{"parts": [{"type": "text", "text": result_text}]}]
                    task.status = TaskStatus(state=TaskState.COMPLETED)

            elif trip_result.get("status") == "error":
                task.status = TaskStatus(
                    state=TaskState.FAILED,
                    message={"role": "agent", "content": {"text": trip_result.get("message", "查询失败，请重试。")}}
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
    trip_server = TripQueryServer()
    print("\n=== 服务器信息 ===")
    print(f"名称: {trip_server.agent_card.name}")
    print(f"描述: {trip_server.agent_card.description}")
    print("\n技能:")
    for skill in trip_server.agent_card.skills:
        print(f"- {skill.name}: {skill.description}")
    #4 启动服务
    run_server(trip_server, host="127.0.0.1", port=5007)
