def _init_logger():
    import logging
    import colorlog

    logger = colorlog.getLogger("searchmanage")

    log_colors_config = {
        'DEBUG': 'white',  # cyan white
        'INFO': 'white',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red',
    }

    # 输出到控制台
    console_handler = colorlog.StreamHandler()
    # 输出到文件
    file_handler = logging.FileHandler(filename="searchmanage.log", mode='a', encoding='utf8')

    # 日志级别，logger 和 handler以最高级别为准，不同handler之间可以不一样，不相互影响
    logger.setLevel(logging.DEBUG)
    console_handler.setLevel(logging.INFO)
    file_handler.setLevel(logging.DEBUG)

    # 日志输出格式
    console_handler.setFormatter(
        colorlog.ColoredFormatter(
            '%(log_color)s[%(asctime)s] [%(levelname)s] %(message)s', log_colors=log_colors_config
        )
    )
    file_handler.setFormatter(logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s'))

    # 重复日志问题：
    # 1、防止多次addHandler；
    # 2、loggername 保证每次添加的时候不一样；
    # 3、显示完log之后调用removeHandler
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    console_handler.close()
    file_handler.close()

    return logger


logger = _init_logger()

from .models import EntitiesSearch, Entities, RequestAnalysis
from .tools import Tools, AnalysisTools
from .SearchManage import SearchManage, Wikipedia, SparqlQuery, BingQuery, SpellCheck, DbpediaLookUp
