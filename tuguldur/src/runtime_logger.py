import logging
import yaml

def logger_setup(log_file="runtime.log"):
    default_config = """
    version: 1
    loggers:
      runtime:
        level: DEBUG
        handlers: [console, file]
        propagate: false

    handlers:
      console:
        class : logging.StreamHandler
        formatter: brief
        level   : WARN
        stream  : ext://sys.stdout
      file:
        class : logging.handlers.RotatingFileHandler
        formatter: default
        filename: {log_file}
        backupCount: 1
        maxBytes: 102400

    formatters:
      brief:
        format: '%(message)s'
      default:
        format: '[%(asctime)s] [%(levelname)-8s] [%(module)s:%(funcName)20s] : %(message)s'
        datefmt: '%H:%M:%S'

    disable_existing_loggers: true

    """

    new_config = default_config.format(log_file=log_file)
    # Load config to dict
    new_config = yaml.load(new_config, Loader=yaml.FullLoader)
    # Configure logger with dict
    logging.config.dictConfig(new_config)

    logger = logging.getLogger('runtime')
    
    logger.info("Runtime log started.")
    logger.info("Logging to file {0}".format(log_file))
    
    return logger
