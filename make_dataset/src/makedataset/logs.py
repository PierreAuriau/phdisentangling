import logging

LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL
}
logger = logging.getLogger()

class ConsoleFormatter(logging.Formatter):

    def __init__(self):
        super(ConsoleFormatter, self).__init__()
        bright_black = "\033[0;90m"
        bright_white = "\033[0;97m"
        yellow = "\033[0;33m"
        red = "\033[0;31m"
        magenta = "\033[0;35m"
        white = "\033[0;37m"

        log_format = "[%(name)s] %(levelname)s %(message)s"

        self.log_formats = {
            logging.DEBUG: bright_black + log_format + bright_white,
            logging.INFO: white + log_format + bright_white,
            logging.WARNING: yellow + log_format + bright_white,
            logging.ERROR: red + log_format + bright_white,
            logging.CRITICAL: magenta + log_format + bright_white
        }

    def format(self, record):
        log_fmt = self.log_formats.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def setup_logging(level="info", logfile=None):
    """ Setup the logging.

    Parameters
    ----------
    level : str, default "info"
        the logger level
    logfile: str, default None
        the log file.
    """

    # Reset handlers
    while len(logging.root.handlers) > 0:
        logging.root.removeHandler(logging.root.handlers[-1])
    while len(logger.handlers) > 0:
        logger.removeHandler(logger.handlers[-1])
    # Set level
    level = LEVELS.get(level, None)
    if level is None:
        raise ValueError("Unknown logging level.")
    logger.setLevel(level)
    # Set console handler
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(ConsoleFormatter())
    logger.addHandler(stream_handler)
    # Set logfile handler
    if logfile is not None:
        logging_format = logging.Formatter(
            "[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - "
            "%(message)s", "%Y-%m-%d %H:%M:%S")
        file_handler = logging.FileHandler(logfile, mode="a")
        file_handler.setLevel(level)
        file_handler.setFormatter(logging_format)
        logger.addHandler(file_handler)
