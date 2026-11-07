# ============================================================
# MCP 行程管家服务模块
# 功能：提供租车查询/预订、旅游团语义检索、保险查询/购买的 MCP 工具接口
# 启动命令：python mcp_server/mcp_trip_server.py
# 访问地址：http://127.0.0.1:8003/mcp
#
# 混合数据源架构：
#   ┌─────────────────────────────────────────────────────────────┐
#   │                     数据源分布                               │
#   ├─────────────────────────────────────────────────────────────┤
#   │  MySQL 数据库          │  Milvus 向量数据库                  │
#   │  - car_rentals (租车)  │  - tour_groups (旅游团语义检索)     │
#   │  - insurances (保险)   │    使用 RAG + Qwen Embedding        │
#   └─────────────────────────────────────────────────────────────┘
#
# 调用流程图：
#
#   AI 智能体（如 Claude）
#        │
#        ▼
#   ┌─────────────────────────────┐
#   │  MCP Server (端口 8003)      │
#   │  create_trip_mcp_server()   │
#   │  FastMCP("TripTools")       │
#   └──────────────┬──────────────┘
#                  │
#        ┌─────────┴─────────┐
#        │  根据工具名称路由   │
#        └─────────┬─────────┘
#                  │
#   ┌──────────────┼──────────────┬──────────────┐
#   ▼              ▼              ▼              ▼
# query_car    query_tour     query_        order_*
# _rental      _group         insurance    （预订）
#   │              │              │              │
#   │              │              │              │
#   ▼              ▼              ▼              ▼
# MySQL        Milvus+API     MySQL        直接返回
# 查询         语义检索       查询         模拟结果
#   │              │              │              │
#   │        ┌─────┴─────┐       │              │
#   │        ▼           ▼       │              │
#   │   Qwen API    Milvus       │              │
#   │   Embedding   向量搜索     │              │
#   │        │           │       │              │
#   │        └─────┬─────┘       │              │
#   │              ▼             │              │
#   │        返回匹配旅游团       │              │
#   │              │             │              │
#   └──────────────┴─────────────┘              │
#                  │                            │
#                  ▼                            ▼
#         结果处理 + JSON 序列化          返回预订成功消息
#                  │                            │
#                  └────────────┬───────────────┘
#                               ▼
#                      JSON 字符串返回给 AI 智能体
#
# 旅游团语义检索（RAG）详细流程：
#
#   用户输入: "想看雪山的地方"
#        │
#        ▼
#   ┌─────────────────────────┐
#   │ Qwen Embedding API      │
#   │ text-embedding-v3       │
#   │ 文本 → 向量 (1024维)     │
#   └───────────┬─────────────┘
#               │
#               ▼
#   ┌─────────────────────────┐
#   │ Milvus 向量数据库        │
#   │ COSINE 相似度搜索        │
#   │ 可选: city 过滤条件      │
#   └───────────┬─────────────┘
#               │
#               ▼
#   返回 Top-K 匹配结果
#   (tour_id, tour_name, city, days, price, similarity)
#
# 返回格式示例：
#   {
#     "status": "success",
#     "data": [
#       {
#         "tour_id": "T001",
#         "tour_name": "丽江雪山探秘5日游",
#         "city": "丽江",
#         "days": 5,
#         "price": 3999.0,
#         "similarity": 0.8956
#       }
#     ]
#   }
#
# 数据库/集合说明：
#   MySQL:
#     car_rentals   - 租车表（公司、车型、价格、座位数等）
#     insurances    - 保险表（类型、保额、保障范围等）
#   Milvus:
#     tour_groups   - 旅游团向量集合（嵌入向量 + 元数据）
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
#
# MILVUS_HOST = "localhost"
# MILVUS_PORT = 19530
# COLLECTION_NAME = "tour_groups"
# EMBEDDING_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"
# EMBEDDING_DIM = 1024

#RAG 的在线部分，用户query转向量
def get_embedding(text: str) -> list:
    """调用 Qwen Embedding API 生成文本向量"""
    import requests
    headers = {
        "Authorization": f"Bearer {conf.api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "text-embedding-v3",
        "input": [text],
        "dimensions": conf.embedding_dim
    }
    response = requests.post(conf.embedding_url, headers=headers, json=payload, timeout=30)
    return response.json()["data"][0]["embedding"]


def search_tour_groups_in_milvus(query_text: str, city: str = None, limit: int = 5) -> list:
    """在 Milvus 中进行语义搜索，找到最匹配的旅游团"""
    from pymilvus import connections, Collection
    connections.connect(alias="default", host=conf.milvus_host, port=conf.milvus_port)
    #获取数据集
    collection = Collection(conf.tour_group_collection)
    collection.load()
    #
    results = collection.search(
        #query_text -- 用户原始query
        data=[get_embedding(query_text)],
        anns_field="embedding", #指定计算列
        #取最相似的10个类
        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        limit=limit,#返回的是5个向量
        output_fields=["tour_id", "tour_name", "city", "days", "price",
                       "total_seats", "remaining_seats", "agency", "rating",
                       "departure_dates", "highlights"],
        expr=f'city == "{city}"' if city else None
    )
    tour_list = []
    for hits in results:
        for hit in hits:
            tour_list.append({
                "tour_id": hit.entity.get("tour_id"),
                "tour_name": hit.entity.get("tour_name"),
                "city": hit.entity.get("city"),
                "days": hit.entity.get("days"),
                "price": hit.entity.get("price"),
                "similarity": round(hit.distance, 4)
            })
    return tour_list
#
class TripService:
    def __init__(self):
        self.conn = mysql.connector.connect(
            host = conf.host,
            port = conf.port,
            user = conf.user,
            password = conf.password,
            database = conf.database
        )
    # 执行参数化查询，返回JSON字符串
    #共用的执行查询函数
    def _execute_query(self, sql: str, params: list = None) -> str:
        try:
            #获取游标
            cursor = self.conn.cursor(dictionary=True)
            cursor.execute(sql, params or [])
            #results --类型[{},{},{}]
            results = cursor.fetchall()
            cursor.close()
            for result in results:
                for key, value in result.items():
                    if isinstance(value, (date, datetime, timedelta, Decimal)):
                        result[key] = default_encoder(value)
            #防御性编程
            return json.dumps(
                {"status": "success", "data": results} if results
                else {"status": "no_data", "message": "未找到行程数据，请确认查询条件。"},
                cls=DateEncoder, ensure_ascii=False
            )
        except Exception as e:
            logger.error(f"票务查询错误: {str(e)}")
            return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)

    # ========== 租车查询（MySQL） ==========
    """
    pickup_city:取车城市
    return_city：还车城市
    date：租车日期
    car_type：租车类型(可选)
    """

    def query_car_rental(self, pickup_city: str, return_city: str, date: str, car_type: str = None) -> str:
        """
        查询租车信息（从 MySQL car_rentals 表查询）
        """
        logger.info(f"查询租车: {pickup_city} -> {return_city}, {date}, {car_type}")
        sql = ("SELECT id, company, pickup_city, return_city, pickup_date, car_type, "
               "car_model, price_per_day, total_available, transmission, seats, deposit "
               "FROM car_rentals "
               "WHERE pickup_city = %s AND return_city = %s AND pickup_date = %s")
        params = [pickup_city, return_city, date]
        if car_type:
            sql += " AND car_type = %s"
            params.append(car_type)
        return self._execute_query(sql, params)

    # ========== 旅游团查询（Milvus RAG） ==========

    def query_tour_group(self, query_text: str, city: str = None) -> str:
        """
        查询旅游团信息（从 Milvus 向量数据库进行语义检索）

        参数：
            query_text (str): 用户查询描述，如"想看雪山的地方"、"适合亲子游的短途旅行"
            city (str, optional): 城市过滤条件

        返回值：
            str: JSON 字符串，包含匹配的旅游团列表
        """
        logger.info(f"查询旅游团: query='{query_text}', city='{city}'")
        try:
            tour_list = search_tour_groups_in_milvus(query_text, city, limit=5)

            if tour_list:
                return json.dumps({"status": "success", "data": tour_list}, ensure_ascii=False)
            else:
                return json.dumps({
                    "status": "no_data",
                    "message": f"未找到匹配的旅游团。{'请确认城市名称，' if city else ''}或尝试其他搜索条件。"
                }, ensure_ascii=False)

        except Exception as e:
            logger.error(f"旅游团 RAG 查询错误: {str(e)}")
            return json.dumps({"status": "error", "message": f"旅游团查询出错：{str(e)}"}, ensure_ascii=False)

    # ========== 保险查询（MySQL） ==========

    def query_insurance(self, insurance_type: str = None) -> str:
        """
        查询旅行保险产品信息（从 MySQL insurances 表查询）
        """
        logger.info(f"查询保险: {insurance_type}")
        sql = ("SELECT id, insurance_type, name, company, coverage, price, "
               "duration_days, max_coverage, medical_coverage, baggage_coverage, flight_delay "
               "FROM insurances")
        params = []
        if insurance_type:
            sql += " WHERE insurance_type = %s"
            params.append(insurance_type)
        return self._execute_query(sql, params)

    # ========== 三个预订方法 ==========

    def order_car_rental(self, date: str, car_type: str, number: int) -> str:
        """预订租车"""
        logger.info(f"预订租车: {date}, {car_type}, {number}辆")
        return f"恭喜，租车预订成功！日期：{date}，车型：{car_type}，数量：{number}辆。"

    def order_tour_group(self, date: str, tour_name: str, number: int) -> str:
        """报名旅游团"""
        logger.info(f"报名旅游团: {date}, {tour_name}, {number}人")
        return f"恭喜，旅游团报名成功！团名：{tour_name}，日期：{date}，人数：{number}人。"

    def order_insurance(self, insurance_type: str, date: str, number: int) -> str:
        """购买旅行保险"""
        logger.info(f"购买保险: {insurance_type}, {date}, {number}份")
        return f"恭喜，旅行保险购买成功！类型：{insurance_type}，日期：{date}，份数：{number}份。"

def create_trip_mcp_server():
    """
    创建并启动行程管家 MCP 服务器
    """
    # 1 创建FastMCP 对象
    trip_mcp = FastMCP(
        name="TripTools",
        instructions="支持预订租车、语义检索旅游团、购买保险。",
        log_level="ERROR",
        host="127.0.0.1", port=8003
    )

    service = TripService()

    @trip_mcp.tool(
        name="query_car_rental",
        description="查询租车信息，参数：pickup_city(取车城市), return_city(还车城市，默认等于取车城市), date(日期，格式YYYY-MM-DD), car_type(车型类型，可选：经济型/SUV/豪华型/MPV)"
    )
    def query_car_rental(pickup_city: str, date: str,return_city: str=None, car_type: str = None) -> str:
        if return_city is None:
            return_city = pickup_city
        return service.query_car_rental(pickup_city, return_city, date, car_type)

    @trip_mcp.tool(
        name="query_tour_group",
        description="查询旅游团信息（语义搜索），参数：query_text(查询描述，如'想看雪山的地方')，city(城市过滤，可选)"
    )
    def query_tour_group(query_text: str, city: str = None) -> str:
        return service.query_tour_group(query_text, city)

    @trip_mcp.tool(
        name="query_insurance",
        description="查询旅行保险产品，参数：insurance_type(保险类型，可选：综合型/意外型/医疗型/境外型)"
    )
    def query_insurance(insurance_type: str = None) -> str:
        return service.query_insurance(insurance_type)

    # ========== 注册预订工具 ==========

    @trip_mcp.tool(
        name="order_car_rental",
        description="根据日期、车型和数量预订租车服务。参数：date(租车日期，格式YYYY-MM-DD), car_type(车型，如经济型/SUV/豪华型/MPV), number(租车数量)"
    )
    def order_car_rental(date: str, car_type: str, number: int) -> str:
        logger.info(f"正在预订租车: {date}, {car_type}, {number}辆")
        return service.order_car_rental(date, car_type, number)

    @trip_mcp.tool(
        name="order_tour_group",
        description="根据出发日期、旅游团名称和人数报名旅游团。参数：date(出发日期，格式YYYY-MM-DD), tour_name(旅游团名称，需与查询结果中的团名一致), number(报名人数)"
    )
    def order_tour_group(date: str, tour_name: str, number: int) -> str:
        logger.info(f"正在报名旅游团: {date}, {tour_name}, {number}人")
        return service.order_tour_group(date, tour_name, number)

    @trip_mcp.tool(
        name="order_insurance",
        description="根据保险类型、生效日期和份数购买旅行保险。参数：insurance_type(保险类型，如综合型/意外型/医疗型/境外型), date(生效日期，格式YYYY-MM-DD), number(购买份数)"
    )
    def order_insurance(insurance_type: str, date: str, number: int) -> str:
        logger.info(f"正在购买保险: {insurance_type}, {date}, {number}份")
        return service.order_insurance(insurance_type, date, number)
    #trip_mcp.run(transport="streamable-http")
    logger.info("=== 票务MCP服务器信息 ===")
    logger.info(f"名称: {trip_mcp.name}")
    logger.info(f"描述: {trip_mcp.instructions}")
    #3 启动
    trip_mcp.run(transport="streamable-http")
    return trip_mcp
if __name__ == '__main__':
    create_trip_mcp_server()