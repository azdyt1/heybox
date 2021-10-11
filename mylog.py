# _*_ coding: utf-8 _*_
import logging
import os
import time


class Logger(object):
    def __init__(self, logger, dirpath):
        """
        指定保存日志的文件路径，日志级别，以及调用文件
            将日志存入到指定的文件中
        :param logger:
        """
        # 创建一个logger
        self.logger = logging.getLogger(logger)
        self.logger.setLevel(logging.DEBUG)

        # 创建日志名称。
        filename = time.strftime('%m_%d-%H-%M-%S', time.localtime(time.time()))
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

        # 创建一个handler，用于写入日志文_
        fh = logging.FileHandler(dirpath+'/'+filename+'.log')
        fh.setLevel(logging.INFO)

        # 创建一个handler，用于输出到控制台
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # 定义handler的输出格式
        formatter = logging.Formatter('%(asctime)s - [%(filename)s:%(lineno)s] - %(levelname)s - %(message)s ')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # 给logger添加handler
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def getlog(self):
        return self.logger