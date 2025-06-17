import logging


# ANSI 颜色代码
LOG_COLORS = {
    'DEBUG': '\033[94m',      # 蓝色
    'INFO': '\033[92m',       # 绿色
    'WARNING': '\033[93m',    # 黄色
    'ERROR': '\033[91m',      # 红色
    'CRITICAL': '\033[95m',   # 紫色
    'RESET': '\033[0m',       # 重置
}

class ColoredFormatter(logging.Formatter):
    def format(self, record):
        levelname = record.levelname
        msg = super().format(record)
        return f"{LOG_COLORS.get(levelname, LOG_COLORS['RESET'])}{msg}{LOG_COLORS['RESET']}"



logger = logging.getLogger("FreePlay2Api")
logger.setLevel(logging.DEBUG)

# 创建文件处理器
file_handler = logging.FileHandler("FreePlay2ApiFull.log")
file_handler.setLevel(logging.DEBUG)
# 创建控制台处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# 设置格式
file_formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s | %(filename)s %(lineno)d", datefmt="%Y-%m-%d %H:%M:%S"
)

# 彩色格式（控制台使用）
colored_formatter = ColoredFormatter(
    "%(asctime)s [%(levelname)s] %(message)s | %(filename)s %(lineno)d",
    datefmt="%Y-%m-%d %H:%M:%S"
)


file_handler.setFormatter(file_formatter)
console_handler.setFormatter(colored_formatter)


# 将处理器添加到日志记录器
logger.addHandler(file_handler)
logger.addHandler(console_handler)

logging = logger