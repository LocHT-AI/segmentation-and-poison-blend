import logging
from logging.handlers import TimedRotatingFileHandler
import os
import yaml
from config.loader import cfg

LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

logger = None

logging.basicConfig(format=LOG_FORMAT)

log_dir = cfg['log']['logdir']
log_file_name = cfg['log']['logfile']
verbose = cfg['log']['verbose']

basedir = os.path.abspath(os.path.dirname(__file__))
log_folder = os.path.join(basedir, log_dir)
if not os.path.exists(log_folder):
    os.makedirs(log_folder)

"""
Initialize logger for logging
"""
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if verbose else logging.INFO)

logname = os.path.join(log_folder, log_file_name)
handler = TimedRotatingFileHandler(logname, when='midnight')
handler.setLevel(logging.DEBUG if verbose else logging.INFO)
type_format = logging.Formatter(LOG_FORMAT)
handler.setFormatter(type_format)

logger.addHandler(handler)
