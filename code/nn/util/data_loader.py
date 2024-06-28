import numpy as np
from itertools import chain
import torch
from torch.utils import data
from fuxictr.pytorch.torch_utils import seed_everything
from fuxictr.features import FeatureMap
from torch.utils.data.dataloader import default_collate
from torch.utils.data import IterDataPipe, DataLoader, get_worker_info
import glob
import polars as pl
import pandas as pd
from tqdm import tqdm
import os
import logging
import random

    
class CusBlockDataPipe(data.IterDataPipe):
    def __init__(self, block_datapipe, feature_map):
        self.feature_map = feature_map
        self.block_datapipe = block_datapipe
        
    def load_data(self, data_path):
        data_dict = np.load(data_path)
        data_arrays = []
        all_cols = list(self.feature_map.features.keys()) + self.feature_map.labels
        for col in all_cols:
            array = data_dict[col]
            if array.ndim == 1:
                data_arrays.append(array.reshape(-1, 1))
            else:
                data_arrays.append(array)
        data_tensor = torch.from_numpy(np.hstack(data_arrays))
        return data_tensor

    def read_block(self, data_block):
        darray = self.load_data(data_block)
        for idx in range(darray.shape[0]):
            yield darray[idx, :]

    def __iter__(self):
        worker_info = data.get_worker_info()
        if worker_info is None: # single-process data loading
            block_list = self.block_datapipe
        else: # in a worker process
            block_list = [
                block
                for idx, block in enumerate(self.block_datapipe)
                if idx % worker_info.num_workers == worker_info.id
            ]
        return chain.from_iterable(map(self.read_block, block_list))


class CusNpzBlockDataLoader(data.DataLoader):
    def __init__(self, feature_map, data_path, batch_size=32, shuffle=False,
                 num_workers=1, buffer_size=500000, ratio=None, path_schema='/*.npz', **kwargs):
        data_blocks = glob.glob(data_path + path_schema)
        assert len(data_blocks) > 0, f"invalid data_path: {data_path}"
        
        if len(data_blocks) > 1:
            data_blocks.sort() # sort by part name
        
        if path_schema == '/*/*.npz':
            data_blocks = sorted(data_blocks, key=lambda x: (int(x.split('/')[-2]), x.split('/')[-1]))
        
        self.data_blocks = data_blocks
        self.num_blocks = len(self.data_blocks)
        self.seed = kwargs['seed']
        
        if ratio is not None:
            print(f'sample ratio={ratio}')
            sample_num_block = int(self.num_blocks * ratio)
            # 随机采样
            seed_everything(self.seed)
            random.shuffle(self.data_blocks)
            # 更新
            self.data_blocks = self.data_blocks[0: sample_num_block]
            self.data_blocks.sort() # sort没有按照日期来排序
            
            if path_schema == '/*/*.npz':
                self.data_blocks = sorted(self.data_blocks, key=lambda x: (int(x.split('/')[-2]), x.split('/')[-1]))
        
            self.num_blocks = len(self.data_blocks)
            
        print('preview', self.data_blocks[0:5])
        
        self.feature_map = feature_map
        self.batch_size = batch_size
        self.num_batches, self.num_samples = self.count_batches_and_samples()
        
        datapipe = CusBlockDataPipe(self.data_blocks, feature_map)
        if shuffle:
            print(f'shuffle, buffer_size={buffer_size}, num_worker={num_workers}....')
            datapipe = datapipe.shuffle(buffer_size=buffer_size)
        else:
            print('not shuffle....')
            num_workers = 1 # multiple workers cannot keep the order of data reading 
        
        print('seed', self.seed)
        
        seed_everything(seed=self.seed)
        super(CusNpzBlockDataLoader, self).__init__(dataset=datapipe, batch_size=batch_size,
                                                 num_workers=1)

    def __len__(self):
        return self.num_batches

    def count_batches_and_samples(self):
        num_samples = 0
        for block_path in tqdm(self.data_blocks):
            block_size = np.load(block_path)[self.feature_map.labels[0]].shape[0]
            num_samples += block_size
        num_batches = int(np.ceil(num_samples / self.batch_size))
        return num_batches, num_samples


class CusRankDataLoader(object):
    def __init__(self, feature_map, stage="both", train_data=None, valid_data=None, test_data=None,
                 batch_size=32, shuffle=True, streaming=False, train_ratio=None, valid_ratio=None, path_schema='/*.npz', **kwargs):
        logging.info("Loading datasets...")
        train_gen = None
        valid_gen = None
        test_gen = None
        DataLoader = CusNpzBlockDataLoader if streaming else CusNpzDataLoader
        self.stage = stage
        if stage in ["both", "train"]:
            train_gen = DataLoader(feature_map, train_data, batch_size=batch_size, shuffle=shuffle, ratio=train_ratio, path_schema=path_schema, **kwargs)
            logging.info("Train samples: total/{:d}, blocks/{:d}".format(train_gen.num_samples, train_gen.num_blocks))     
            if valid_data:
                valid_gen = DataLoader(feature_map, valid_data, batch_size=batch_size, shuffle=False, ratio=valid_ratio, path_schema=path_schema, **kwargs)
                logging.info("Validation samples: total/{:d}, blocks/{:d}".format(valid_gen.num_samples, valid_gen.num_blocks))

        if stage in ["both", "test"]:
            if test_data:
                test_gen = DataLoader(feature_map, test_data, batch_size=batch_size, shuffle=False, **kwargs)
                logging.info("Test samples: total/{:d}, blocks/{:d}".format(test_gen.num_samples, test_gen.num_blocks))
        self.train_gen, self.valid_gen, self.test_gen = train_gen, valid_gen, test_gen

    def make_iterator(self):
        if self.stage == "train":
            logging.info("Loading train and validation data done.")
            return self.train_gen, self.valid_gen
        elif self.stage == "test":
            logging.info("Loading test data done.")
            return self.test_gen
        else:
            logging.info("Loading data done.")
            return self.train_gen, self.valid_gen, self.test_gen
        
        
        
#交叉验证定制
def cv_kfold_split(seed, data_path, n_fold=5):
    if isinstance(data_path, str):
        data_path = [data_path]
        
    data_blocks = []
    for path in data_path:
        files = glob.glob(path + "/*.npz")
        data_blocks.extend(files)
        
    assert len(data_blocks) > 0, f"invalid data_path: {data_path}"
    if len(data_blocks) > 1:
        data_blocks.sort() # sort by part name
        
    print(f'to split total data_block_num={len(data_blocks)}')
    
    from sklearn.model_selection import KFold
    seed_everything(seed)
    num_data_blocks = len(data_blocks)

    X = np.array(range(num_data_blocks)).reshape(-1, 1)
    base = pd.DataFrame(X, columns=['idx'])
    base['filename'] = data_blocks
    
    base['fold'] = -1
    cv = KFold(n_splits=n_fold, shuffle=True)
    for fold_i, (idx_train, idx_valid) in enumerate(cv.split(base)):
        base.loc[idx_valid, 'fold'] = fold_i
    
    return base
    
    
def cv_kfold_split_by_day(seed, train_path, valid_path=None, n_fold=5, path_schema='/*.npz'):

    data_block_dict = {}
    train_files = glob.glob(train_path + path_schema)
    
    for day in range(7):
        data_blocks = glob.glob(train_path + '/' + str(day) + path_schema)
        data_blocks.sort()
        data_block_dict[day] = data_blocks
    
    if valid_path is not None:
        for day in range(7, 14):
            data_blocks = glob.glob(valid_path + '/' + str(day) + path_schema)
            data_blocks.sort()
            data_block_dict[day] = data_blocks
            
    assert len(data_block_dict) > 0, f"invalid data_path: {train_path}"
        
    print(f'to split total num days={len(data_block_dict)}')
    
    from sklearn.model_selection import KFold
    seed_everything(seed)
    num_data_blocks = len(data_blocks)
    
    # X = np.array(range(14)).reshape(-1, 1)
    # base = pd.DataFrame(X, columns=['day'])
    # base['fold'] = -1
    # cv = KFold(n_splits=n_fold, shuffle=True)
    # for fold_i, (idx_train, idx_valid) in enumerate(cv.split(base)):
    #     base.loc[idx_valid, 'fold'] = fold_i
    
    X = np.array(range(14)).reshape(-1, 1)
    base = pd.DataFrame(X, columns=['day'])
    base['fold'] = -1
    base.loc[base.day.isin([0,1,2]), 'fold'] = 0
    base.loc[base.day.isin([3,4,5]), 'fold'] = 1
    base.loc[base.day.isin([6,7,8]), 'fold'] = 2
    base.loc[base.day.isin([9,10,11]), 'fold'] = 3
    base.loc[base.day.isin([12,13]), 'fold'] = 4
    
    return base, data_block_dict


class CusCvNpzBlockDataLoader(data.DataLoader):
    def __init__(self, feature_map, data_blocks, batch_size=32, shuffle=False,
                 num_workers=1, buffer_size=500000, ratio=None, **kwargs):
        
        assert len(data_blocks) > 0
        print(f'dataloader, data_blocks_num={len(data_blocks)}')
    
        if len(data_blocks) > 1:
            data_blocks.sort() # sort by part name
        self.data_blocks = data_blocks
        self.num_blocks = len(self.data_blocks)
        self.seed = kwargs['seed']
        if ratio is not None:
            print(f'sample ratio={ratio}')
            sample_num_block = int(self.num_blocks * ratio)
            # 随机采样
            seed_everything(self.seed)
            random.shuffle(self.data_blocks)
            # 更新
            self.data_blocks = self.data_blocks[0: sample_num_block]
            self.data_blocks.sort()
            self.num_blocks = len(self.data_blocks)
            
        self.feature_map = feature_map
        self.batch_size = batch_size
        self.num_batches, self.num_samples = self.count_batches_and_samples()
        datapipe = CusBlockDataPipe(self.data_blocks, feature_map)
        if shuffle:
            print(f'shuffle, buffer_size={buffer_size}')
            datapipe = datapipe.shuffle(buffer_size=buffer_size)
        else:
            print('not shuffle....')
            num_workers = 1 # multiple workers cannot keep the order of data reading 
        
        print('seed={}'.format(self.seed))
        seed_everything(self.seed)

        super(CusCvNpzBlockDataLoader, self).__init__(dataset=datapipe, batch_size=batch_size,
                                                 num_workers=1)

    def __len__(self):
        return self.num_batches

    def count_batches_and_samples(self):
        num_samples = 0
        for block_path in tqdm(self.data_blocks):
            block_size = np.load(block_path)[self.feature_map.labels[0]].shape[0]
            num_samples += block_size
        num_batches = int(np.ceil(num_samples / self.batch_size))
        return num_batches, num_samples
    
    
class CusCvRankDataLoader(object):
    def __init__(self, feature_map, stage="cv", train_data=None, valid_data=None, test_data=None,
                 batch_size=32, shuffle=True, streaming=False, n_fold=5, train_ratio=None, valid_ratio=None, **kwargs):
        logging.info("Loading datasets...")
        train_gen = None
        valid_gen = None
        test_gen = None
        self.stage = stage
        
        # 合并
        data_path = [train_data]
        if valid_data is not None:
            data_path.append(valid_data)
        print(f'data_path={data_path}')

        train_valid_gens = []

        if stage == 'cv':
            fold_pd, data_block_dict = cv_kfold_split_by_day(kwargs['seed'], train_data, valid_path=valid_data, n_fold=n_fold)
                
            for fold_i in range(n_fold):
                
                valid_data_days = fold_pd[fold_pd.fold == fold_i]['day'].tolist() # 该fold验证集ID
                valid_data_days.sort()
                train_data_days = fold_pd[fold_pd.fold != fold_i]['day'].tolist() # 该fold训练集ID
                train_data_days.sort()
                print(f'valid days={valid_data_days}, train days={train_data_days}')
                
                train_data_blocks, valid_data_blocks = [], []
                
                for day in train_data_days:
                    train_data_blocks.extend(data_block_dict[day])
                for day in valid_data_days:
                    valid_data_blocks.extend(data_block_dict[day])
                
                # train_data_blocks = ['./data/ebnerd_large_x2_3c350928/train/part_00000.npz']
                # valid_data_blocks = ['./data/ebnerd_large_x2_3c350928/train/part_00000.npz']
                
                print(f"Fold {fold_i}, train={len(train_data_blocks)}, valid={len(valid_data_blocks)}")

                train_gen = CusCvNpzBlockDataLoader(feature_map, train_data_blocks, batch_size=batch_size, 
                                                    shuffle=shuffle, ratio=train_ratio, **kwargs)
                valid_gen = CusCvNpzBlockDataLoader(feature_map, valid_data_blocks, 
                                                    batch_size=batch_size, shuffle=False, ratio=valid_ratio, **kwargs)
                
                train_valid_gens.append((train_gen, valid_gen))
                logging.info("Fold {:d}, Train samples: total/{:d}, blocks/{:d}".format(fold_i, train_gen.num_samples, train_gen.num_blocks))  
                logging.info("Fold {:d}, Validation samples: total/{:d}, blocks/{:d}".format(fold_i, valid_gen.num_samples, valid_gen.num_blocks))
                print("----------------------------------")
            self.train_valid_gens = train_valid_gens
            self.fold_pd = fold_pd
            
        elif stage == 'hold_out':
            train_data_blocks, valid_data_blocks = train_val_split(kwargs['seed'], train_data, valid_data, 
                                                                   valid_ratio=valid_ratio, shuffle=True)
            self.train_gen = CusCvNpzBlockDataLoader(feature_map, train_data_blocks, batch_size=batch_size, shuffle=shuffle, **kwargs)
            self.valid_gen = CusCvNpzBlockDataLoader(feature_map, valid_data_blocks, batch_size=batch_size, shuffle=False, **kwargs)
            

    def make_iterator(self):
        if self.stage == 'cv':
            return self.train_valid_gens
        elif self.stage == 'hold_out':
            return self.train_gen, self.valid_gen
