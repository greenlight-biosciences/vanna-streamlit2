import logging
import os
class VannaLogger:
    def __init__(self, env_var_name='LOG_LEVEL'):
        self.setup_logger(env_var_name)

    def setup_logger(self, env_var_name):
        # Fetch the log level from environment variable, default to INFO
        env_log_level = os.getenv(env_var_name, 'INFO').upper()
        log_level = getattr(logging, env_log_level, logging.INFO)

        # Configure logging with the environment variable's level
        logging.basicConfig(level=log_level,
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
