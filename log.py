import logging 
import time

logger = None

def log(message):
    if logger == None:
        init_logs()

    logger.info(message)

def debug(message):
    if logger == None:
        init_logs()

    logger.debug(message)

def info(message):
    if logger == None:
        init_logs()

    logger.info(message)

def warning(message):
    if logger == None:
        init_logs()

    logger.warning(message)

def error(message):
    if logger == None:
        init_logs()

    logger.error(message)

def init_logs(logPath="./log/", label = ""):
    global logger
    logger = logging.getLogger()
    logger.setLevel(logging.NOTSET)

    logName = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    logFile = logPath+logName+label+".log"
    fh = logging.FileHandler(logFile, mode='w') 
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logger.debug("log file path: "+logFile)

    return logName 

