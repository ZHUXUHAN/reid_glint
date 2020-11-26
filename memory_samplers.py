import numpy as np
import os
from default_hard_config import hard_config


class CenterPositiveClassGet(object):
    """ Get the corresponding center of the positive class
    """

    def __init__(self, num_sample, num_local, rank):
        self.num_sample = num_sample
        self.num_local = num_local
        self.rank = rank
        self.rank_class_start = self.rank * num_local
        self.rank_class_end = self.rank_class_start + num_local
        pass

    def __call__(self, global_label):
        """
        Return:
        -------
        positive_center_label: list of int
        """
        greater_than = global_label >= self.rank_class_start
        smaller_than = global_label < self.rank_class_end

        positive_index = greater_than * smaller_than
        positive_center_label = global_label[positive_index]

        return positive_center_label


class CenterNegetiveClassSample(object):
    """ Sample negative class center
    """

    def __init__(self, num_sample, num_local, rank):
        self.num_sample = num_sample
        self.num_local = num_local
        self.rank = rank
        self.negative_class_pool = np.arange(num_local)
        pass

    def __call__(self, positive_center_index):
        """
        Return:
        -------
        negative_center_index: list of int
        """
        negative_class_pool = np.setdiff1d(self.negative_class_pool, positive_center_index)
        negative_sample_size = self.num_sample - len(positive_center_index)
        if negative_sample_size <= 0:
            # print('negative_sample_size', negative_sample_size, 'num_sample', self.num_sample,
            #       'len(positive_center_index)', \
            #       len(positive_center_index), 'num_local', self.num_local)
            negative_center_index = np.random.choice(negative_class_pool,
                                                     0, replace=False)
        else:
            negative_center_index = np.random.choice(negative_class_pool,
                                                     negative_sample_size, replace=False)
        return negative_center_index


class HardNegetiveClassSample(object):
    """ Sample hard negative class center
        add by yang xiao
    """

    def __init__(self, num_sample, num_local, rank, name):
        self.num_sample = num_sample
        self.num_local = num_local
        self.rank = rank
        self.similar_categories = None
        self.count = 0
        self.interval = hard_config.INTERVAL
        self.proportion = hard_config.PROPORTION
        self.last_mtime = -1
        self.name = name

    def update(self):
        self.count += 1
        if self.count % self.interval != 0:
            return None
        # FIXME
        # file_name = '/anxiang/tmp/xj600w_largeFC.param.npy'
        file_name = os.path.join(hard_config.PREFIX, '%s_largeFC.param.npy' % self.name)

        while os.path.exists(file_name) and os.path.getmtime(file_name) != self.last_mtime:
            try:
                self.last_mtime = os.path.getmtime(file_name)
                self.similar_categories = np.load(file_name)
                if self.rank == 0:
                    # TODO convert to logging
                    print('update success', self.similar_categories.shape)
                break
            except BaseException as arg:
                # TODO convert to logging
                print("Rank %d Error!" % self.rank)
                print(arg)
                self.last_mtime = -1
                self.similar_categories = None

    def __call__(self, global_label, positive_center_index):
        """
        Parameters:
        ----------
        global_label: np.ndarray
            the unique label of the global_label.
        positive_center_index: np.ndarray
            xxx
        Returns:
        -------
        important_center_index: np.ndarray
            xxx
        """
        assert isinstance(global_label, np.ndarray)
        assert isinstance(positive_center_index, np.ndarray)
        global_label = global_label.astype(np.int32)
        self.update()
        if self.similar_categories is None:
            return positive_center_index
        assert isinstance(self.similar_categories, np.ndarray)

        hard_center_size = (self.num_sample - positive_center_index.shape[0]) * self.proportion
        hard_center_size = int(hard_center_size)

        if hard_center_size < 0:
            return positive_center_index

        tmp = self.similar_categories[global_label].T.reshape(-1) - self.rank * self.num_local
        tmp = tmp[(0 <= tmp) & (tmp < self.num_local)]
        tmp = np.concatenate([positive_center_index, tmp])
        _, index = np.unique(tmp, return_index=True)
        index.sort()
        if len(index) >= positive_center_index.shape[0] + hard_center_size:
            index = index[:positive_center_index.shape[0] + hard_center_size]
        return tmp[index]


class WeightIndexSampler(object):
    """
    """

    def __init__(self, num_sample, num_local, rank, name):
        self.num_sample = num_sample
        self.num_local = num_local
        self.rank = rank
        self.rank_class_start = self.rank * num_local
        self.name = name

        if hard_config.HARD_SERVER:
            self.hard = HardNegetiveClassSample(num_sample, num_local, rank, name)

        self.positive = CenterPositiveClassGet(num_sample, num_local, rank)
        self.negative = CenterNegetiveClassSample(num_sample, num_local, rank)

    def sample_index(self, global_label):
        positive_center_label = self.positive(global_label)
        positive_center_index = positive_center_label - self.positive.rank_class_start
        negative_center_index = self.negative(positive_center_index)
        #
        final_center_index = np.concatenate((positive_center_index, negative_center_index))
        return positive_center_index, final_center_index

    def __call__(self, global_label):
        positive_center_label = self.positive(global_label)
        positive_center_index = positive_center_label - self.positive.rank_class_start
        #
        if hard_config.HARD_SERVER:
            positive_center_index = self.hard(global_label, positive_center_index)
        negative_center_index = self.negative(positive_center_index)
        #
        final_center_index = np.concatenate((positive_center_index, negative_center_index))
        if len(np.unique(final_center_index)) != self.num_sample:
            final_center_index = final_center_index[:self.num_sample]
        assert len(final_center_index) == len(np.unique(final_center_index)) == self.num_sample
        assert len(final_center_index) == self.num_sample
        return final_center_index
