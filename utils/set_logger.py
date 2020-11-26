import logging
import os
import sys


def set_logger(logger, rank, models_root):
    formatter = logging.Formatter("rank-id:" + str(rank) + ":%(asctime)s-%(message)s")
    file_handler = logging.FileHandler(os.path.join(models_root, "%d_hist.log" % rank))
    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.info('rank_id: %d' % rank)
