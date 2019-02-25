class Logger:
    def __init__(self):
        import logging
        self.LOG_LEVEL = logging.INFO
        LOGFORMAT = "%(log_color)s[%(levelname)s] [%(log_color)s%(asctime)s] %(log_color)s%(filename)s [line:%(log_color)s%(lineno)d] : %(log_color)s%(message)s%(reset)s"
        import colorlog
        logging.root.setLevel(self.LOG_LEVEL)
        ############
        # 此配置是将日志输出到myapp.log
        colorlog.basicConfig(format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                             filename='mylog.log',
                             filemode='w', datefmt='%a, %d %b %Y %H:%M:%S')
        ##############
        formatter = colorlog.ColoredFormatter(LOGFORMAT)
        self.stream = logging.StreamHandler()
        self.stream.setLevel(self.LOG_LEVEL)
        self.stream.setFormatter(formatter)

    def getLogger(self):
        import logging
        log = logging.getLogger()
        log.setLevel(self.LOG_LEVEL)
        log.addHandler(self.stream)
        return log


# log.debug("A quirky message only developers care about")
# log.info("Curious users might want to know this")
# log.warning("Something is wrong and any user should be informed")
# log.error("Serious stuff, this is red for a reason")
# log.critical("OH NO everything is on fire")
#####################################
if __name__ == "__main__":
    log = Logger().getLogger()
    log.warning('aaa')
    log.info("bbb")
