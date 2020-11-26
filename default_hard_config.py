# hard sample config
from easydict import EasyDict as edict

hard_config = edict()

# workers
hard_config.HARD_SERVER = False                             #
hard_config.PREFIX = '/anxiang/models/r122_2730_xl/hard_server'                 #
hard_config.INTERVAL = 100                                  #
hard_config.PROPORTION = 0.8                                #
hard_config.SAVE_INTERVAL = 1000                            #
hard_config.TOP_K = 200


# server
hard_config.HEAD_NAME_LIST = ['sfml', 'celeb', 'badoo', 'scv5', 'xj']
hard_config.BATCH_SIZE = 400
hard_config.FILE_NUM = 64