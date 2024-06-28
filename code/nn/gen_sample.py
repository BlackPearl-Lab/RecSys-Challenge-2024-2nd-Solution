from pathlib import Path
import polars as pl
import numpy as np
import os
import gc
import time
import warnings
import argparse
from fuxictr.utils import load_config, print_to_json, print_to_list
import sys
sys.path.append('.')
from util import *
warnings.filterwarnings("ignore")


os.chdir('../../')

base_dataset = 'nn_data'
root = 'inputs'
data_path = f"{root}/large"
train_path = f"{data_path}/train"
dev_path = f"{data_path}/validation"
test_path = f"{data_path}/test"
vector_path = f"{root}/vectors"



parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='code/nn/config/V1.yaml', help='The config directory.')
parser.add_argument('--re_load', action="store_true", help='load the trained feature processor.')
parser.add_argument('--mode', type=str, default='all', help='mode: all/train/validation/test, only use the argument when re_load is true')
args = vars(parser.parse_args())


mode = args['mode']
re_load = args['re_load']

config_file = args['config']
config_dir = enumerate_params(config_file)
experiment_id_list = load_experiment_ids(config_dir)
exp_id = experiment_id_list[0]
params = load_config(config_dir, exp_id)

data_dir = os.path.join(params['data_root'], params['dataset_id'])
feature_map_json = os.path.join(data_dir, "feature_map.json")


timestamp = str(int(time.time()))
set_logger(params, model_id_suffix=timestamp)


feature_encoder = CusFeatureProcessor(**params)

if re_load:
    feature_encoder = feature_encoder.load_pickle()
    print('load the trained feature processor done...')
    
else:
    print('preparing training feature processor....')
    article_ddf = pl.scan_parquet(params['article_data'])
    
    train_ddf = pl.scan_parquet(params['train_data'])
    train_ddf = train_ddf.join(article_ddf, on='article_id', how='left')
    train_ddf = feature_encoder.preprocess(train_ddf)
    
    valid_ddf = pl.scan_parquet(params['valid_data'])
    valid_ddf = valid_ddf.join(article_ddf, on='article_id', how='left')
    valid_ddf = feature_encoder.preprocess(valid_ddf)
    
    test_ddf = pl.scan_parquet(params['test_data'])
    test_ddf = test_ddf.filter(pl.col('impression_id') > 0)
    test_ddf = test_ddf.join(article_ddf, on='article_id', how='left')
    test_ddf = feature_encoder.preprocess(test_ddf)
    
    train_history_ddf = pl.scan_parquet(params['train_history'])
    train_history_ddf = feature_encoder.preprocess(train_history_ddf)
    
    valid_history_ddf = pl.scan_parquet(params['valid_history'])
    valid_history_ddf = feature_encoder.preprocess(valid_history_ddf)
    
    test_history_ddf = pl.scan_parquet(params['test_history'])
    test_history_ddf = feature_encoder.preprocess(test_history_ddf)
    
    article_ddf = feature_encoder.preprocess(article_ddf)
    
    
    feature_encoder.fit(pl.concat([train_ddf, valid_ddf, test_ddf]), 
                        pl.concat([train_history_ddf, valid_history_ddf, test_history_ddf]), 
                        article_ddf,
                        **params)

    print('preparing training feature processor done')


# read meta info
data_dir = os.path.join(params['data_root'], params['dataset_id'])
feature_map_json = os.path.join(data_dir, "feature_map.json")
feature_map = FeatureMap(params['dataset_id'], data_dir)
feature_map.load(feature_map_json, params) 
all_cols = list(feature_map.features.keys()) + feature_map.labels


# generate training samples
if mode  in  ('all', 'train'):
    print('preparing training, transform....')
    df_train_ans = transform_by_chunk_day_parquet_oom_restart_opt(feature_encoder, 
                                              params['train_data'], params['train_history'], params['article_data'],
                                              'train', params['data_block_size'], num_thread=10,
                                               all_cols=all_cols,
                                               save_npz_compressed=True)
    
    df_train_ans.to_parquet(os.path.join(feature_encoder.data_dir, 'train_ids_by_day.parquet')) 
    print('preparing training, done....')


# generate validation samples
if mode  in  ('all', 'validation'):
    print('preparing validation, transform....')
    df_valid_ans = transform_by_chunk_day_parquet_oom_restart_opt(feature_encoder, 
                                              params['valid_data'], params['valid_history'], params['article_data'],
                                              'valid', params['data_block_size'], num_thread=10,
                                              all_cols=all_cols,
                                              save_npz_compressed=True)
    
    df_valid_ans.to_parquet(os.path.join(feature_encoder.data_dir, 'valid_ids_by_day.parquet'))
    print('preparing validation, done....')


# generate test samples
if mode  in  ('all', 'test'):
    print('preparing test, transform....')
    df_test_ans = transform_by_chunk_parquet_oom_restart_opt(feature_encoder, 
                               params['test_data'], params['test_history'], params['article_data'],
                               'test', params['data_block_size'], num_thread=10,
                               all_cols=all_cols, save_npz_compressed=True)
    
    df_test_ans.to_parquet(os.path.join(feature_encoder.data_dir, 'test_ids.parquet')) 
    
    print('preparing test, done....')
