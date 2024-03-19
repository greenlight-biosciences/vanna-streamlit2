import logging

class VannaLogger:
    def __init__(self):
        self.setup_logger()

    def setup_logger(self):
        # Configure logging
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # Create a logger specific to this class
        self.logger = logging.getLogger(self.__class__.__name__)

    def log(self, message: str, level: str = 'info'):
        """General log method with level flexibility."""
        log_method = {
            'debug': self.logger.debug,
            'info': self.logger.info,
            'warning': self.logger.warning,
            'error': self.logger.error,
        }.get(level.lower(), self.logger.info)
        log_method(message)

    def logDebug(self, message: str):
        self.logger.debug(message)
    def logError(self, message: str):
        self.logger.error(message)
    def logWarning(self, message: str):
        self.logger.warning(message)
    def logInfo(self, message: str):
        self.logger.info(message)
