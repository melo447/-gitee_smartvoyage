# ============================================================
# MCP 天气服务模块
# 功能：提供天气查询的 MCP（Model Context Protocol）工具接口
# 支持两种数据源：MySQL 数据库 / 和风天气 API（通过配置切换）
# 启动命令：python mcp_server/mcp_weather_server.py
# 访问地址：http://127.0.0.1:8002/mcp
#
# 调用流程图：
#
#   AI 智能体（如 Claude）
#        │
#        ▼
#   ┌─────────────────────────────┐
#   │  MCP Server (端口 8002)      │
#   │  create_weather_mcp_server() │
#   │  FastMCP("WeatherTools")     │
#   └──────────────┬──────────────┘
#                  │
#                  ▼
#           query_weather(city, start_date, end_date)
#                  │
#        ┌─────────┴─────────┐
#        │ 检查 weather_source│
#        └─────────┬─────────┘
#                  │
#        ┌─────────┴─────────┐
#        │                   │
#     "api"               其他（"db"）
#        │                   │
#        ▼                   ▼
#   ┌──────────────┐  ┌──────────────────┐
#   │ fetch_weather│  │ WeatherService   │
#   │ _from_api()  │  │ .query_weather() │
#   └──────┬───────┘  └────────┬─────────┘
#          │                   │
#          ▼                   ▼
#   ① geo接口:城市→ID     SQL查询 weather_data 表
#          │           SELECT * FROM weather_data
#          ▼            WHERE city = ? AND
#   ② 天气API:用ID获取数据    fx_date BETWEEN ? AND ?
#          │                   │
#          ▼                   ▼
#   ③ 过滤日期范围         遍历结果集（字典格式）
#   ④ 字段名映射           处理 date/Decimal 类型
#      tempMax→temp_max      │
#      textDay→text_day      ▼
#                     json.dumps() 序列化为 JSON 字符串
#          │                   │
#          └─────────┬─────────┘
#                    ▼
#           JSON 字符串返回给 AI 智能体
#
#   返回格式示例：
#   {
#     "status": "success",
#     "data": [
#       {
#         "city": "成都",
#         "fx_date": "2026-04-29",
#         "temp_max": 28,
#         "temp_min": 18,
#         "text_day": "多云",
#         "text_night": "晴",
#         "humidity": 65,
#         "wind_dir_day": "南风",
#         "wind_scale_day": "3",
#         "precip": 0.0
#       }
#     ]
#   }
#
# 数据源切换：修改 config.py 中的 weather_source 配置
#   weather_source = "api"  → 使用和风天气实时 API
#   weather_source = "db"   → 使用本地 MySQL 数据库
# ============================================================

import mysql.connector #连接MySQL数据库
import json
from datetime import date, datetime, timedelta
from decimal import Decimal
from mcp.server.fastmcp import FastMCP #mcp官方FastMCP

from config import Config
from create_logger import logger
from utils.format import DateEncoder, default_encoder
import requests


conf = Config()

# 天气服务类
class WeatherService:
    def __init__(self):
        self.conn = mysql.connector.connect(
            host=conf.host,
            port=conf.port,
            user=conf.user,
            password=conf.password,
            database=conf.database
        )

    # 执行参数化查询，返回JSON字符串
    """
    city: 查询天气的城市
    start_date：查询开始时间
    end_date：查询结束时间
    """
    def query_weather(self, city: str, start_date: str, end_date: str) -> str:
        try:
            #返回类型由元组变成字典
            cursor = self.conn.cursor(dictionary=True)
            #参数化查询
            sql = ("SELECT city, fx_date, temp_max, temp_min, text_day, text_night, "
                   "humidity, wind_dir_day, wind_scale_day, precip FROM weather_data "
                   "WHERE city = %s AND fx_date BETWEEN %s AND %s ORDER BY fx_date")
            cursor.execute(sql, [city, start_date, end_date])
            #results数据类型[{},{},{},{}]
            results = cursor.fetchall()
            cursor.close()
            for result in results:
                for key, value in result.items():
                    if isinstance(value, (date, datetime, timedelta, Decimal)):
                        result[key] = default_encoder(value)
              #返回的类型Json 字符串
            return json.dumps(
                {"status": "success", "data": results} if results
                else {"status": "no_data", "message": "未找到天气数据，请确认城市和日期。"},
                cls=DateEncoder, ensure_ascii=False
            )
        except Exception as e:
            logger.error(f"天气查询错误: {str(e)}")
            return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)

    def fetch_weather_from_api(self, city: str, start_date: str, end_date: str) -> str:
        """从和风天气 API 获取天气数据"""
        logger.info(f"从和风天气API获取数据: {city}, {start_date} ~ {end_date}")
        headers = {
            "X-QW-Api-Key": conf.weather_api_key
        }

        # 城市转地理位置
        geo_url = f"{conf.weather_base_url}/geo/v2/city/lookup?location={city}"
        try:
            geo_resp = requests.get(geo_url, headers=headers, timeout=10)
            geo_resp.raise_for_status()
            geo_data = geo_resp.json()
        except Exception as e:
            logger.error(f"和风API城市查询失败: {e}")
            return json.dumps({"status": "error", "message": f"城市查询失败: {e}"}, ensure_ascii=False)

        if not geo_data.get("location"):
            return json.dumps({"status": "no_data", "message": f"未找到城市：{city}"}, ensure_ascii=False)
        location_id = geo_data["location"][0]["id"]

        # 获取天气预报（3天）
        weather_url = f"{conf.weather_base_url}/v7/weather/3d?location={location_id}"
        try:
            weather_resp = requests.get(weather_url, headers=headers, timeout=10)
            weather_resp.raise_for_status()
            weather_data = weather_resp.json()
        except Exception as e:
            logger.error(f"和风API天气查询失败: {e}")
            return json.dumps({"status": "error", "message": f"天气查询失败: {e}"}, ensure_ascii=False)

        # 检查和风API返回的业务状态码
        if weather_data.get("code") != "200":
            err_msg = weather_data.get("error", {}).get("text", "未知错误")
            logger.error(f"和风API返回错误: code={weather_data.get('code')}, msg={err_msg}")
            return json.dumps({"status": "error", "message": f"天气API错误: {err_msg}"}, ensure_ascii=False)

        # 过滤日期范围并格式化
        results = []
        for day in weather_data.get("daily", []):
            fx_date = day.get("fxDate", "")[:10]  # 取 YYYY-MM-DD 部分
            if start_date <= fx_date <= end_date:
                results.append({
                    "city": city,
                    "fx_date": fx_date,
                    "temp_max": int(day.get("tempMax", 0)),
                    "temp_min": int(day.get("tempMin", 0)),
                    "text_day": day.get("textDay", ""),
                    "text_night": day.get("textNight", ""),
                    "humidity": int(day.get("humidity", 0)),
                    "wind_dir_day": day.get("windDirDay", ""),
                    "wind_scale_day": day.get("windScaleDay", ""),
                    "precip": float(day.get("precip", 0))
                })

        if results:
            return json.dumps({"status": "success", "data": results}, ensure_ascii=False)
        return json.dumps({"status": "no_data", "message": f"未找到{city}在{start_date}~{end_date}的天气数据（API仅支持未来3天预报）。"}, ensure_ascii=False)


def create_weather_mcp_server():
    weather_mcp = FastMCP(
        name="WeatherTools",
        instructions="天气查询工具，支持数据库或和风天气API双数据源。",
        log_level="ERROR",
        host="127.0.0.1", port=8002
    )

    service = WeatherService()

    @weather_mcp.tool(
        #工具名
        name="query_weather",
        #工具描述
        description="查询天气数据，参数：city(城市), start_date(开始日期YYYY-MM-DD), end_date(结束日期YYYY-MM-DD)"
    )
    #工具的具体逻辑
    def query_weather(city: str, start_date: str, end_date: str) -> str:
        logger.info(f"执行天气查询: {city}, {start_date} ~ {end_date}")
        if conf.weather_source == "api":
            return service.fetch_weather_from_api(city, start_date, end_date)
        return service.query_weather(city, start_date, end_date)

    logger.info("=== 天气MCP服务器信息 ===")
    logger.info(f"名称: {weather_mcp.name}")
    logger.info(f"描述: {weather_mcp.instructions}")
    #启动
    weather_mcp.run(transport="streamable-http")

if __name__ == "__main__":
    create_weather_mcp_server()