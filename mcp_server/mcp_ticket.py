# ============================================================
# MCP 票务服务模块
# 功能：提供火车票、机票、演唱会票的查询与预定的 MCP 工具接口
# 数据源：MySQL 数据库（train_tickets / flight_tickets / concert_tickets 表）
# 启动命令：python mcp_server/mcp_ticket_server.py
# 访问地址：http://127.0.0.1:8001/mcp
#
# 调用流程图：
#
#   AI 智能体（如 Claude）
#        │
#        ▼
#   ┌─────────────────────────────┐
#   │  MCP Server (端口 8001)      │
#   │  create_ticket_mcp_server() │
#   │  FastMCP("TicketTools")     │
#   └──────────────┬──────────────┘
#                  │
#        ┌─────────┴─────────┐
#        │  根据工具名称路由   │
#        └─────────┬─────────┘
#                  │
#   ┌──────────────┼──────────────┬──────────────┐
#   ▼              ▼              ▼              ▼
# query_train  query_flight  query_concert   order_*（预定）
#   │              │              │              │
#   └──────────────┴──────────────┘              │
#                  │                             │
#                  ▼                             ▼
#         TicketService 方法              直接返回模拟结果
#                  │                      （演示模式）
#                  ▼
#         _execute_query(sql, params)
#                  │
#                  ▼
#         ┌──────────────────┐
#         │  MySQL 数据库     │
#         │  参数化查询       │
#         │  (防 SQL 注入)    │
#         └────────┬─────────┘
#                  │
#        ┌─────────┴─────────┐
#        │   查询结果处理     │
#        │  转换 date/Decimal │
#        │  序列化为 JSON     │
#        └─────────┬─────────┘
#                  ▼
#         JSON 字符串返回给 AI 智能体
#
#   返回格式示例：
#   {
#     "status": "success",
#     "data": [
#       {
#         "id": 1,
#         "departure_city": "北京",
#         "arrival_city": "上海",
#         "departure_time": "2026-05-01 08:00:00",
#         "arrival_time": "2026-05-01 12:30:00",
#         "train_number": "G101",
#         "seat_type": "二等座",
#         "price": 553.0,
#         "remaining_seats": 120
#       }
#     ]
#   }
#
# 数据库表说明：
#   train_tickets    - 火车票表（车次、座位类型、余票等）
#   flight_tickets   - 机票表（航班号、舱位类型、余票等）
#   concert_tickets  - 演唱会票表（艺人、场馆、票类型等）
# ============================================================

import mysql.connector
import json
from datetime import date, datetime, timedelta
from decimal import Decimal
from mcp.server.fastmcp import FastMCP

from config import Config
from create_logger import logger
from utils.format import DateEncoder, default_encoder

conf = Config()


# 票务服务类
class TicketService:
    def __init__(self):
        self.conn = mysql.connector.connect(
            host=conf.host,
            port = conf.port,
            user=conf.user,
            password=conf.password,
            database=conf.database
        )

    # 执行参数化查询，返回JSON字符串
    #功能就是执行传入的sql语句
    def _execute_query(self, sql: str, params: list = None) -> str:
        try:
            cursor = self.conn.cursor(dictionary=True)
            #执行sql
            cursor.execute(sql, params or [])
            #results列表 --[{},{}]
            results = cursor.fetchall()
            cursor.close()
            for result in results:
                for key, value in result.items():
                    if isinstance(value, (date, datetime, timedelta, Decimal)):
                        result[key] = default_encoder(value)
            #Json字符串
            return json.dumps(
                {"status": "success", "data": results} if results
                else {"status": "no_data", "message": "未找到票务数据，请确认查询条件。"},
                cls=DateEncoder, ensure_ascii=False
            )
        except Exception as e:
            logger.error(f"票务查询错误: {str(e)}")
            return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)

    # 查询火车票（参数化）
    """
    departure_city:出发城市
    arrival_city：到达城市
    date：出发日期
    seat_type：座位类型(可选参数)
    """
    def query_train(self, departure_city: str, arrival_city: str, date: str, seat_type: str = None) -> str:
        sql = ("SELECT id, departure_city, arrival_city, departure_time, arrival_time, "
               "train_number, seat_type, price, remaining_seats FROM train_tickets "
               "WHERE departure_city = %s AND arrival_city = %s AND DATE(departure_time) = %s")
        params = [departure_city, arrival_city, date]
        if seat_type:
            sql += " AND seat_type = %s"
            params.append(seat_type)
        return self._execute_query(sql, params)

    # 查询机票（参数化）
    """
        departure_city:出发城市
        arrival_city：到达城市
        date：出发日期
        cabin_type：座位类型(可选参数)
        """
    def query_flight(self, departure_city: str, arrival_city: str, date: str, cabin_type: str = None) -> str:
        sql = ("SELECT id, departure_city, arrival_city, departure_time, arrival_time, "
               "flight_number, cabin_type, price, remaining_seats FROM flight_tickets "
               "WHERE departure_city = %s AND arrival_city = %s AND DATE(departure_time) = %s")
        params = [departure_city, arrival_city, date]
        if cabin_type:
            sql += " AND cabin_type = %s"
            params.append(cabin_type)
        return self._execute_query(sql, params)

    # 查询演唱会票（参数化）
    """
            city:举办城市
            date：举办日期
            artist：艺人名字(可选参数)
            ticket_type：座位类型(可选参数)
            """
    def query_concert(self, city: str, date: str, artist: str = None, ticket_type: str = None) -> str:
        sql = ("SELECT id, artist, city, venue, start_time, end_time, "
               "ticket_type, price, remaining_seats FROM concert_tickets "
               "WHERE city = %s AND DATE(start_time) = %s")
        params = [city, date]
        if artist:
            sql += " AND artist = %s"
            params.append(artist)
        if ticket_type:
            sql += " AND ticket_type = %s"
            params.append(ticket_type)
        return self._execute_query(sql, params)

    # 预定方法（模拟返回）
    def order_train(self, departure_date: str, train_number: str, seat_type: str, number: int) -> str:
        return f"恭喜，火车票预定成功！日期：{departure_date}，车次：{train_number}，座位：{seat_type}，数量：{number}张。"

    def order_flight(self, departure_date: str, flight_number: str, cabin_type: str, number: int) -> str:
        return f"恭喜，机票预定成功！日期：{departure_date}，航班号：{flight_number}，舱位：{cabin_type}，数量：{number}张。"

    def order_concert(self, start_date: str, artist: str, venue: str, ticket_type: str, number: int) -> str:
        return f"恭喜，演出票预定成功！日期：{start_date}，艺人：{artist}，场馆：{venue}，票类型：{ticket_type}，数量：{number}张。"

#
def create_ticket_mcp_server():
    ticket_mcp = FastMCP(
        name="TicketTools",
        instructions="票务工具，支持火车票、机票、演唱会票的查询与预定。",
        log_level="ERROR",
        host="127.0.0.1", port=8001
    )

    service = TicketService()

    # 注册查询工具
    @ticket_mcp.tool(
        #工具名字
        name="query_train",
        #工具的描述
        description="查询火车票信息，参数：departure_city(出发城市), arrival_city(到达城市), date(日期), seat_type(座位类型，可选)"
    )
    def query_train(departure_city: str, arrival_city: str, date: str, seat_type: str = None) -> str:
        return service.query_train(departure_city, arrival_city, date, seat_type)

    @ticket_mcp.tool(
        name="query_flight",
        description="查询机票信息，参数：departure_city(出发城市), arrival_city(到达城市), date(日期), cabin_type(舱位类型，可选)"
    )
    def query_flight(departure_city: str, arrival_city: str, date: str, cabin_type: str = None) -> str:
        return service.query_flight(departure_city, arrival_city, date, cabin_type)

    @ticket_mcp.tool(
        name="query_concert",
        description="查询演唱会票信息，参数：city(城市), date(日期YYYY-MM-DD), artist(艺人姓名，可选，不填则查询该城市所有演唱会), ticket_type(票类型，可选)"
    )
    def query_concert(city: str, date: str, artist: str = None, ticket_type: str = None) -> str:
        return service.query_concert(city, date, artist, ticket_type)

    # 注册预定工具
    @ticket_mcp.tool(
        name="order_train",
        description="根据日期、车次、座位类型、数量预定火车票"
    )
    def order_train(departure_date: str, train_number: str, seat_type: str, number: int) -> str:
        return service.order_train(departure_date, train_number, seat_type, number)

    @ticket_mcp.tool(
        name="order_flight",
        description="根据日期、航班号、舱位类型、数量预定机票"
    )
    def order_flight(departure_date: str, flight_number: str, cabin_type: str, number: int) -> str:
        return service.order_flight(departure_date, flight_number, cabin_type, number)

    @ticket_mcp.tool(
        name="order_concert",
        description="根据日期、艺人、场馆、票类型、数量预定演出票"
    )
    def order_concert(start_date: str, artist: str, venue: str, ticket_type: str, number: int) -> str:
        return service.order_concert(start_date, artist, venue, ticket_type, number)

    logger.info("=== 票务MCP服务器信息 ===")
    logger.info(f"名称: {ticket_mcp.name}")
    logger.info(f"描述: {ticket_mcp.instructions}")
    # 步骤5：启动服务器
    try:
        print("服务器已启动，请访问 http://127.0.0.1:8001/mcp")
        ticket_mcp.run(transport="streamable-http")
    except Exception as e:
        print(f"服务器启动失败: {e}")

if __name__ == "__main__":
    create_ticket_mcp_server()