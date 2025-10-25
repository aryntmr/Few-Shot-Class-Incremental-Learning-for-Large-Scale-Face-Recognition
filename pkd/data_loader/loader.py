import sys
sys.path.append('..')
#import nvidia.dali.fn as fn
import torch.utils.data as data
import random
import torch
import time
from collections import defaultdict
from torch.utils.data.distributed import DistributedSampler, Dataset
import math
from typing import TypeVar, Optional, Iterator

__all__ = ["DistributedCustomSampler"]

T_co = TypeVar('T_co', covariant=True)

class ClassUniformlySampler(data.sampler.Sampler):
    '''
    random sample according to class label
    Arguments:
        data_source (Dataset): data_loader to sample from
        class_position (int): which one is used as class
        k (int): sample k images of each class
    '''

    def __init__(self, data_source, class_position, k):

        self.data_source = data_source
        self.class_position = class_position
        self.k = k

        self.samples = self.data_source.samples
        self.class_dict = self._tuple2dict(self.samples)


    def __iter__(self):
        self.sample_list = self._generate_list(self.class_dict)
        return iter(self.sample_list)

    def __len__(self):
        return len(self.sample_list)

    def _tuple2dict(self, inputs):
        '''

        :param inputs: list with tuple elemnts, [(image_path1, class_index_1), (imagespath_2, class_index_2), ...]
        :return: dict, {class_index_i: [samples_index1, samples_index2, ...]}
        '''
        dict = {}
        for index, each_input in enumerate(inputs):
            class_index = each_input[self.class_position]
            if class_index not in list(dict.keys()):
                dict[class_index] = [index]
            else:
                dict[class_index].append(index)
        return dict


    def _generate_list(self, dict):
        '''
        :param dict: dict, whose values are list
        :return:
        '''

        sample_list = []

        dict_copy = dict.copy()
        keys = list(dict_copy.keys())
        random.shuffle(keys)
        for key in keys:
            value = dict_copy[key]
            if len(value) >= self.k:
                random.shuffle(value)
                sample_list.extend(value[0: self.k])
            else:
                value = value * self.k
                random.shuffle(value)
                sample_list.extend(value[0: self.k])

        return sample_list



class ClassUniformlySampler4continual(data.sampler.Sampler):
    '''
    random sample according to class label
    Arguments:
        data_source (Dataset): data_loader to sample from
        class_position (int): which one is used as class
        k (int): sample k images of each class
    '''

    def __init__(self, data_source, class_position, k, pid_list):

        self.data_source = data_source
        self.class_position = class_position
        self.k = k

        self.samples = self.data_source.samples
        class_dict = self._tuple2dict(self.samples)
        self.class_dict = self.filter_current_step_pids(class_dict, pid_list)



    def __iter__(self):
        self.sample_list = self._generate_list(self.class_dict)
        return iter(self.sample_list)

    def __len__(self):
        return len(self.sample_list)

    def _tuple2dict(self, inputs):
        '''

        :param inputs: list with tuple elemnts, [(image_path1, class_index_1), (imagespath_2, class_index_2), ...]
        :return: dict, {class_index_i: [samples_index1, samples_index2, ...]}
        '''
        dict = {}
        for index, each_input in enumerate(inputs):
            class_index = each_input[self.class_position]
            if class_index not in list(dict.keys()):
                dict[class_index] = [index]
            else:
                dict[class_index].append(index)
        return dict


    def _generate_list(self, dict):
        '''
        :param dict: dict, whose values are list
        :return:
        '''

        sample_list = []

        dict_copy = dict.copy()
        keys = list(dict_copy.keys())
        random.shuffle(keys)
        for key in keys:
            value = dict_copy[key]
            if len(value) >= self.k:
                random.shuffle(value)
                sample_list.extend(value[0: self.k])
            else:
                value = value * self.k
                random.shuffle(value)
                sample_list.extend(value[0: self.k])

        return sample_list

    def filter_current_step_pids(self, class_dict, pid_list):
        update_dict = {}
        for pid in pid_list:
            assert pid in class_dict.keys()
            update_dict[pid] = class_dict[pid]

        return update_dict



class ClassUniformlySampler4Incremental(data.sampler.Sampler):
    '''
    random sample according to class label
    Arguments:
        data_source (Dataset): data_loader to sample from
        class_position (int): which one is used as class
        k (int): sample k images of each class
    '''

    def __init__(self, data_source, class_position, k, pid_list):

        self.data_source = data_source
        self.class_position = class_position
        self.k = k

        self.samples = self.data_source.samples
        class_dict = self._tuple2dict(self.samples)
        self.class_dict = self.filter_current_step_pids(class_dict, pid_list)

    def __iter__(self):
        self.sample_list = self._generate_list(self.class_dict)
        return iter(self.sample_list)        

    def __len__(self):
        length = len(self.sample_list)
        return length

    def _tuple2dict(self, inputs):
        '''

        :param inputs: list with tuple elemnts, [(image_path1, class_index_1), (imagespath_2, class_index_2), ...]
        :return: dict, {class_index_i: [samples_index1, samples_index2, ...]}
        '''
        dict = defaultdict(list)
        for index, each_input in enumerate(inputs):
            class_index = each_input[self.class_position]
            dict[class_index].append(index)
        return dict

    def _generate_list(self, dict):
        '''
        :param dict: dict, whose values are list
        :return:
        '''

        sample_list = []

        dict_copy = dict.copy()
        keys = list(dict_copy.keys())
        random.shuffle(keys)
        for key in keys:
            value = dict_copy[key]
            if len(value) >= self.k:
                random.shuffle(value)
                sample_list.extend(value[0: self.k])
                #sampled_samples = random.sample(value, min(len(value), self.k))
                #sample_list.extend(sampled_samples)
            else:
                value = value * self.k
                random.shuffle(value)
                sample_list.extend(value[0: self.k])
                #while len(value) < self.k:
                 #   value.append(random.choice(value))
                #sampled_samples = random.sample(value, self.k)
                #sample_list.extend(sampled_samples)

        return sample_list

    def filter_current_step_pids(self, class_dict, pid_list):
        update_dict = {}
        for pid in pid_list:
            assert pid in class_dict.keys()
            update_dict[pid] = class_dict[pid]
        return update_dict

class DistributedCustomSampler(DistributedSampler):
    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False,
                 class_position: int = 1, k: int = 4, pid_list: list = None) -> None:
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.class_position = class_position
        self.k = k
        self.pid_list = pid_list

    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        if not self.drop_last:
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            indices = indices[:self.total_size]

        assert len(indices) == self.total_size

        indices = indices[self.rank:self.total_size:self.num_replicas]

        # Use the custom sampler logic here
        custom_sampler = ClassUniformlySampler4Incremental(self.dataset, self.class_position, self.k, self.pid_list)
        custom_indices = list(custom_sampler)

        # Distribute the custom indices across processes
        custom_indices = custom_indices[self.rank:len(custom_indices):self.num_replicas]

        return iter(custom_indices)

    def __len__(self) -> int:
        return len(self.sample_list)

    def set_epoch(self, epoch: int) -> None:
        super().set_epoch(epoch)

class IterLoader:

    def __init__(self, loader):
        self.loader = loader
        self.iter = iter(self.loader)

    def next_one(self):
        try:
            return next(self.iter)
        except:
            self.iter = iter(self.loader)
            return next(self.iter)
