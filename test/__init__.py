import os

os.chdir("..")
from impl import config

config.init_config()

CONFIG = config.CONFIG
