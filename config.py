import os
from dotenv import load_dotenv
load_dotenv()

# 项目根目录
project_root = os.path.dirname(os.path.abspath(__file__))
print(project_root)
# 生产环境
# env = "prod"
# 测试环境
# env = "test"
# 开发环境
env = "dev"
# 预生产环境
# env = "pre_prod"


#定义配置文件
class Config:

    def __init__(self):
        # 大模型配置
        self.base_url = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
        self.api_key = os.getenv("DASHSCOPE_API_KEY")
        self.model_name = 'qwen3.5-omni-plus'

        # mysql数据库配置
        self.host = 'localhost'
        # 如果mysql还没有这个用户，需要先手动创建这个用户和密码，工作中一般不用root，会root会有安全风险
        #CREATE USER 'smart_voyage'@'%' IDENTIFIED BY '123456';
        #GRANT ALL PRIVILEGES ON travel_rag.* TO 'smart_voyage'@'%';
        #FLUSH PRIVILEGES;
        self.user = 'smart_voyage' # 不是root
        self.password = '123456'
        self.port = 3306
        self.database = 'travel_rag'

        # 日志配置
        self.log_file = os.path.join(project_root, 'logs', 'app.log')
        # 查询天气的城市映射
        self.city_codes = {
            # --- 一线及新一线热门 ---
            "北京": "101010100",
            "上海": "101020100",
            "广州": "101280101",
            "深圳": "101280601",
            "成都": "101270101",
            "杭州": "101210101",  # 注意：之前代码里写错成了广州的代码，已修正
            "西安": "101110101",
            "重庆": "101040100",
            "南京": "101190101",
            "武汉": "101200101",
            "长沙": "101250101",

            # --- 特色旅游/度假城市 ---
            "贵阳": "101260101",  # 贵州
            "厦门": "101230201",  # 福建
            "青岛": "101120201",  # 山东
            "三亚": "101310201",  # 海南
            "昆明": "101290101",  # 云南
            "大理": "101290201",  # 云南大理州
            "丽江": "101291401",  # 云南
            "哈尔滨": "101050101", # 黑龙江
            "苏州": "101190401",  # 江苏
            "桂林": "101300501"   # 广西
        }
        self.weather_api_key = os.getenv("WEATHER_API_KEY")
        self.weather_base_url = "https://ng5u9yu9y2.re.qweatherapi.com"
        self.weather_api_host = "ng5u9yu9y2.re.qweatherapi.com"
        self.weather_timezone = "Asia/Shanghai"
        #Milvus 向量数据库配置
        self.milvus_host = "localhost"
        self.milvus_port = 19530
        self.tour_group_collection = "tour_groups"

        # Qwen Embedding API 配置
        self.embedding_url = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"
        self.embedding_dim = 1024

        #意图预设
        self.intent = {
            "weather": "WeatherQueryAssistant",
            "flight": "TicketAssistant",
            "train": "TicketAssistant",
            "concert": "TicketAssistant",
            "order": "TicketAssistant",
            "car_rental": "TripAssistant",
            "tour_group": "TripAssistant",
            "insurance": "TripAssistant",
            "trip_order": "TripAssistant",
        }

        self.temperature = 0.1 # 大模型的随机性
        # 天气定时更新调度时间（北京时间）
        self.weather_schedule_time = "01:00"

        # 天气数据源配置
        # 可选值："database"（从数据库获取） / "api"（直接从和风API获取）
        self.weather_source = "api"


    def get_mysql_config(self, env:str):
        """
        通过不同的环境获取不同的数据库配置
        :return:
        """
        if env == 'prod':
            self.host = 'localhost'
            self.user = 'root'
            self.password = 'root'
            self.database = 'travel_rag'
        elif env == 'dev':
            self.host = 'localhost'
            self.user = 'smart_voyage'
            self.password = '123456'
            self.database = 'travel_rag'
        elif env == 'test':
            self.host = 'localhost2'
            self.user = 'root2'
            self.password = 'root2'
            self.database = 'travel_rag'
        else:
            self.host = 'localhost3'
            self.user = 'root3'
            self.password = 'root3'
            self.database = 'travel_rag'

        return self.host, self.user, self.password, self.database


if __name__ == '__main__':
    print(Config().log_file)
    print(Config().api_key)
    print(Config().get_mysql_config(env))