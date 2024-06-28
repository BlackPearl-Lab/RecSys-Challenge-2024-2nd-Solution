import logging
from datetime import datetime
import gc
import argparse
import itertools
import polars as pl
import subprocess
import yaml
import os
import numpy as np
import time
import glob
import hashlib
from fuxictr.utils import print_to_json, load_model_config, load_dataset_config
import joblib
from tqdm import tqdm
from sklearn.metrics import roc_auc_score



def set_logger(params, model_id_suffix=''):
    dataset_id = params['dataset_id']
    model_id = params.get('model_id', '')
    log_dir = os.path.join(params.get('model_root', './checkpoints'), dataset_id + '_' + model_id_suffix)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, model_id + '.log')

    # logs will not show in the file without the two lines.
    for handler in logging.root.handlers[:]: 
        logging.root.removeHandler(handler)
        
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s P%(process)d %(levelname)s %(message)s',
                        handlers=[logging.FileHandler(log_file, mode='w'),
                                  logging.StreamHandler()])
    
    
def enumerate_params(config_file, exclude_expid=[]):
    with open(config_file, "r") as cfg:
        config_dict = yaml.load(cfg, Loader=yaml.FullLoader)
    # 模型tuning space
    tune_dict = config_dict["tuner_space"]
    for k, v in tune_dict.items():
        if not isinstance(v, list):
            tune_dict[k] = [v]
            
   # 模型：DIN_default
    experiment_id = config_dict["base_expid"]
    
    if "model_config" in config_dict: # 自定义model_config
        model_dict = config_dict["model_config"][experiment_id]
    else:
        base_config_dir = config_dict.get("base_config", os.path.dirname(config_file)) # 优先使用config_file里的base_config
        model_dict = load_model_config(base_config_dir, experiment_id)

    # 数据集，dataset_id标识
    dataset_id = config_dict.get("dataset_id", model_dict["dataset_id"]) # 优先使用config_file里的dataset_id
    if "dataset_config" in config_dict:
        dataset_dict = config_dict["dataset_config"][dataset_id] # 优先使用自定义的数据集
    else:
        dataset_dict = load_dataset_config(base_config_dir, dataset_id)
    
    # 实验ID编码方式：模型名称_数据集ID
    if model_dict["dataset_id"] == "TBD": # rename base expid
        model_dict["dataset_id"] = dataset_id
        experiment_id = model_dict["model"] + "_" + dataset_id

    # key checking
    tuner_keys = set(tune_dict.keys())
    base_keys = set(model_dict.keys()).union(set(dataset_dict.keys()))
    if len(tuner_keys - base_keys) > 0:
        raise RuntimeError("Invalid params in tuner config: {}".format(tuner_keys - base_keys))

    # 创建出来的实验粒度的配置，并输出，文件夹名字命名
    config_dir = config_file.replace(".yaml", "")
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)

    # enumerate dataset para combinations, 
    dataset_dict = {k: tune_dict[k] if k in tune_dict else [v] for k, v in dataset_dict.items()}
    dataset_para_keys = list(dataset_dict.keys())
    # ['data_root',
    #  'data_format',
    #  'train_data',
    #  'valid_data',
    #  'test_data',
    #  'min_categr_count',
    #  'data_block_size',
    #  'streaming',
    #  'feature_cols',
    #  'label_col']

    dataset_para_combs = dict()
    
    # 所有dataset的配置项取值进行排列组合，比如低频过滤可以填[10,20]，用dataset_id+组合hash标识
    for idx, values in enumerate(itertools.product(*map(dataset_dict.get, dataset_para_keys))):
        dataset_params = dict(zip(dataset_para_keys, values))
        if (dataset_params["data_format"] == "npz" or
           (dataset_params["data_format"] == "parquet" and 
            dataset_params["rebuild_dataset"] == False)):
            dataset_para_combs[dataset_id] = dataset_params
        else:
            hash_id = hashlib.md5("".join(sorted(print_to_json(dataset_params))).encode("utf-8")).hexdigest()[0:8]
            dataset_para_combs[dataset_id + "_{}".format(hash_id)] = dataset_params
    # print(dataset_para_combs)

    # dump dataset para combinations to config file
    dataset_config = os.path.join(config_dir, "dataset_config.yaml")
    with open(dataset_config, "w") as fw:
        yaml.dump(dataset_para_combs, fw, default_flow_style=None, indent=4)

    
    # 所有模型配置项排列组合 enumerate model para combinations
    model_dict = {k: tune_dict[k] if k in tune_dict else [v] for k, v in model_dict.items()}
    model_para_keys = list(model_dict.keys())
    model_param_combs = dict()
    for idx, values in enumerate(itertools.product(*map(model_dict.get, model_para_keys))):
        model_param_combs[idx + 1] = dict(zip(model_para_keys, values))
        
    # print(model_param_combs)
    # 数据项，模型项 排列组合，把dataset_id_hash标识写进model_config中
    merged_param_combs = dict()
    for idx, item in enumerate(itertools.product(model_param_combs.values(),
                                                 dataset_para_combs.keys())):
        para_dict = item[0]
        para_dict["dataset_id"] = item[1]
        
        if 'model_id' in para_dict:
            del para_dict["model_id"]
            
        random_str = ""
        if para_dict["debug_mode"]:
            random_str = "{:06d}".format(np.random.randint(1e6)) # add a random number to avoid duplicate during debug
        hash_id = hashlib.md5(("".join(sorted(print_to_json(para_dict))) + random_str).encode("utf-8")).hexdigest()[0:8]
        hash_expid = experiment_id + "_{:03d}_{}".format(idx + 1, hash_id)

        if hash_expid not in exclude_expid:
            merged_param_combs[hash_expid] = para_dict.copy()
    
    print(merged_param_combs.keys())
    
    # dump model para combinations to config file
    model_config = os.path.join(config_dir, "model_config.yaml")
    print(model_config)
    with open(model_config, "w") as fw:
        yaml.dump(merged_param_combs, fw, default_flow_style=None, indent=4)
        
    print("Enumerate all tuner configurations done.")    
    return config_dir


def load_experiment_ids(config_dir):
    model_configs = glob.glob(os.path.join(config_dir, "model_config.yaml"))
    if not model_configs:
        model_configs = glob.glob(os.path.join(config_dir, "model_config/*.yaml"))
    experiment_id_list = []
    for config in model_configs:
        with open(config, "r") as cfg:
            config_dict = yaml.load(cfg, Loader=yaml.FullLoader)
            experiment_id_list += config_dict.keys()
    return sorted(experiment_id_list)


def get_gauc(df, true_col='click', pred_col='score'):
    def calc_auc(x):
        return roc_auc_score(x[true_col], x[pred_col])

    aucs = joblib.Parallel(n_jobs=32, backend="multiprocessing")(
        joblib.delayed(calc_auc)(group) for _, group in tqdm(df.select(['impression_id', pred_col, true_col]).groupby("impression_id"))
    )
    return np.mean(aucs)