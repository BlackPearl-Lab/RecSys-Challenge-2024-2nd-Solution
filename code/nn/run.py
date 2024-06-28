import os
import torch
import torch.nn as nn
from fuxictr.utils import load_config, print_to_json, print_to_list
import logging
import random
import time
import argparse
import pickle
import sys
sys.path.append('.')
import warnings
warnings.filterwarnings("ignore")

from util import *
from util.model import CusMaskNet, CusDIN

os.chdir('../../') # change to root path


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='code/nn/config/V1.yaml', help='The config directory.')
parser.add_argument('--mode', type=str, default='all', help='mode: all/train/cv/submit')
parser.add_argument('--gpus', nargs='+', default=[0], help='list of gpus, -1 for cpu')
parser.add_argument('--batch_size', type=int, default=10240, help='batch size')
parser.add_argument('--epochs', type=int, default=2, help='epoch')
parser.add_argument('--eval_steps', type=int, default=2000, help='eval steps')
parser.add_argument('--model_dir', type=str, default='', help='loading ckpt from pretrained model path, e.g., checkpoints/V1_6e00181c')

args = vars(parser.parse_args())

mode = args['mode']
epochs = args['epochs']
gpus = [int(g) for g in args['gpus']]
batch_size = args['batch_size']
model_dir = args['model_dir']
eval_steps = args['eval_steps']
logging.info(f'mode={mode}, model_dir={model_dir}, gpus={gpus}, batch_size={batch_size}, epochs={epochs}, eval_steps={eval_steps}')

# step1: load config
print('step1: load config begin...')
config_file = args['config']
config_dir = enumerate_params(config_file)
experiment_id_list = load_experiment_ids(config_dir)
exp_id = experiment_id_list[0]
params = load_config(config_dir, exp_id)


timestamp = str(int(time.time()))
set_logger(params, model_id_suffix=timestamp)
logging.info("Params: " + print_to_json(params))
print('step1: load config done...')

# step2: load data
print('step2: load config begin...')
data_dir = os.path.join(params['data_root'], params['dataset_id'])
feature_map_json = os.path.join(data_dir, "feature_map.json")
feature_map = FeatureMap(params['dataset_id'], data_dir)
feature_map.load(feature_map_json, params) 
print('step2: load config end...')

params["train_data"], params["valid_data"], params["test_data"] = os.path.join(data_dir, "train"), \
           os.path.join(data_dir, "valid"), \
           os.path.join(data_dir, "test")

for feature, feature_spec in feature_map.features.items():
    print(feature, feature_spec)

print('step3: setting params begin...')
params['gpus'] = gpus
if len(gpus) == 1:
    params['gpu'] = gpus[0]
else:
    params['gpu'] = -2 # all
gpu = params['gpu']

print(f'gpus={gpus}, gpu={gpu}')


params['batch_size'] = batch_size
params['epochs'] = epochs
params['metrics'] = ['avgAUC']
params['eval_steps'] = eval_steps
params['batch_norm'] = True # importance for nn
    
params['din_target_field']= [('category', 'subcat1', 'sentiment_label', 'article_type'),
      ('article_id_img', 'article_id_text')]

params['din_sequence_field'] = [('hist_cat', 'hist_subcat1', 'hist_sentiment', 'hist_type', 'hist_scroll_percent_bin', 'hist_read_time_bin'), ('hist_id_img', 'hist_id_text')] 
    
not_required_feature_columns=['article_id', 'user_id', 'hist_id']


print('step3: setting params end...')

# ............training..............
if mode  in ('all', 'train'):
    print('step4: 2-fold training begin...')


    logging.info(f"epochs={params['epochs']}")
    logging.info(f"learning_rate={params['learning_rate']}")
    logging.info(f"batch_size={params['batch_size']}")
    logging.info(f"eval_steps={params['eval_steps']}")
    
    
    logging.info(f"train_data={params['train_data']}, valid_data={params['valid_data']}")
    train_gen, valid_gen = CusRankDataLoader(feature_map, stage='train', path_schema='/*/*.npz', 
                                             valid_ratio=0.05, **params).make_iterator()
    
    params['train_data'], params['valid_data'] = params['valid_data'], params['train_data']
    logging.info(f"train_data={params['train_data']}, valid_data={params['valid_data']}")
    train_val_gen, valid_tr_gen = CusRankDataLoader(feature_map, stage='train', 
                                                            valid_ratio=0.05, path_schema='/*/*.npz', **params).make_iterator()
    params['train_data'], params['valid_data'] = params['valid_data'], params['train_data'] # recover
    
    
    train_valid_gens = [(train_gen, valid_gen), (train_val_gen, valid_tr_gen)]
    auc_list = []
    full_auc_list = []
    
    
    for i, (fold_train_gen, fold_valid_gen) in enumerate(train_valid_gens): 
        logging.info(f'************************* fold_{i} begin ****************************')
        seed_everything(seed=params['seed'])
        
        fold_model = CusDIN(feature_map, model_id_suffix=timestamp, 
                            not_required_feature_columns=not_required_feature_columns, **params)
        
        # fold_model = CusMaskNet(feature_map, model_id_suffix=timestamp, 
        #                  not_required_feature_columns=not_required_feature_columns, # session_article_seq
        #                  emb_batchnorm=True,
        #                  **params)
        
        fold_model.checkpoint = os.path.abspath(os.path.join(fold_model.model_dir, fold_model.model_id + f"_fold_{i}.model"))
        logging.info(f'fold_{i} checkpoint={fold_model.checkpoint}')
        
        fold_model.fit(fold_train_gen, validation_data=fold_valid_gen, **params)
        auc = fold_model.val_logs['avgAUC']
        auc_list.append(auc)
        
        del fold_model
        torch.cuda.empty_cache()
        
        logging.info(f'************************* fold_{i} end *****************************')
    
    print('step4: 2-fold training end...')



# ............submitting..............
if mode  in ('all', 'submit'):
    params['batch_size'] = 20480 
    
    print('step4: 2-fold submitting begin...')
    if len(model_dir) == 0:
         model_dir = os.path.join(params["model_root"], feature_map.dataset_id + '_' + timestamp)
    model_id = params['model_id']
    logging.info(f'model_dir={model_dir}, model_id={model_id}....')
        
    ans = pl.read_parquet(os.path.join(data_dir, 'test_ids.parquet'))

    os.makedirs(f'{model_dir}/sub', exist_ok=True)
    
    for i in range(2):
        test_gen = CusRankDataLoader(feature_map, stage='test', **params).make_iterator()
        
        pretrain_ckpt_path = os.path.join(model_dir, f'{model_id}_fold_{i}.model')
        
        logging.info(f'************************* fold_{i} begin ****************************')
        
        seed_everything(seed=params['seed'])
        
        fold_model = CusDIN(feature_map, model_id_suffix=timestamp, 
                            not_required_feature_columns=not_required_feature_columns, **params)
            
        # fold_model = CusMaskNet(feature_map, model_id_suffix=timestamp, 
        #                  not_required_feature_columns=not_required_feature_columns, 
        #                  emb_batchnorm=True,
        #                  **params)
        
        fold_model.load_weights(pretrain_ckpt_path)
        print(f'loading ckpt={pretrain_ckpt_path}')
    
        fold_test_score = fold_model.predict(test_gen)

        ans = ans.with_columns(score=fold_test_score).with_columns(pl.col('score').alias(f'fold_{i}_score'))
        ans.write_parquet(f'{model_dir}/sub/sub_fold_{i}.parquet', use_pyarrow=True)
        
        del fold_model
        torch.cuda.empty_cache()
        
        logging.info(f'************************* fold_{i} end *****************************')

    ans = ans.with_columns(final_score=(pl.col('fold_0_score')+pl.col('fold_1_score'))/2.0)

    # ensure the order is same as the original testing behavior data
    test = pl.read_parquet("inputs/large/test/behaviors.parquet").with_columns(
        [
            pl.col("impression_id").cast(pl.Int64),
            pl.col("user_id").cast(pl.Int64)

        ]
    ).explode(['article_ids_inview']).select(['impression_id','user_id','article_ids_inview'])
    test = test.with_columns(pl.col('article_ids_inview').cast(pl.Int64)).rename({'article_ids_inview': 'article_id'})
    test = test.join(ans, on=['impression_id', 'user_id', 'article_id'], how='left')
    
    os.makedirs(f'code/oof/xtf_v42_2fold_856', exist_ok=True)
    test.write_parquet(f'code/oof/xtf_v42_2fold_856/test_score.parquet', use_pyarrow=True)
    print('step4: 2-fold submitting end...')

        

# ............cv predicting..............
if mode  in ('all', 'cv'):
    params['batch_size'] = 20480 
    params['shuffle'] = False # important
    print('predict validation result begin...')
    model_id = params['model_id']
    
    logging.info(f"full_train_data={params['train_data']}, full_valid_data={params['valid_data']}")
    full_train_gen, full_valid_gen = CusRankDataLoader(feature_map, 
                                                       stage='train', path_schema='/*/*.npz', **params).make_iterator()

    if len(model_dir) == 0:
         model_dir = os.path.join(params["model_root"], feature_map.dataset_id + '_' + timestamp)
    logging.info(f'model_dir={model_dir}....')

    os.makedirs(f'{model_dir}/cv', exist_ok=True)

    pretrain_ckpt_path = os.path.join(model_dir, f'{model_id}_fold_0.model')
    seed_everything(seed=params['seed'])
    
    fold_model = CusDIN(feature_map, model_id_suffix=timestamp, 
                        not_required_feature_columns=not_required_feature_columns, **params)

    # fold_model = CusMaskNet(feature_map, model_id_suffix=timestamp, 
    #                      not_required_feature_columns=not_required_feature_columns, 
    #                      emb_batchnorm=True,
    #                      **params)
    fold_model.load_weights(pretrain_ckpt_path)
    fold_valid_score = fold_model.predict(full_valid_gen)

    os.makedirs(f'code/oof/xtf_v42_2fold_856', exist_ok=True)
    valid_ids_by_day_df = pl.read_parquet(os.path.join(data_dir, 'valid_ids_by_day.parquet'))
    valid_ids_by_day_df = valid_ids_by_day_df.with_columns(score=fold_valid_score)
    valid_ids_by_day_df.drop('impression_day').write_parquet('code/oof/xtf_v42_2fold_856/valid_score.parquet', use_pyarrow=True)
    del fold_model
    torch.cuda.empty_cache()
    print('predict validation result end...')


    print('predict training result begin...')
    pretrain_ckpt_path = os.path.join(model_dir, f'{model_id}_fold_1.model')
    seed_everything(seed=params['seed'])
    
    fold_model = CusDIN(feature_map, model_id_suffix=timestamp, 
                        not_required_feature_columns=not_required_feature_columns, **params)

    # fold_model = CusMaskNet(feature_map, model_id_suffix=timestamp, 
    #                      not_required_feature_columns=not_required_feature_columns, 
    #                      emb_batchnorm=True,
    #                      **params)
    fold_model.load_weights(pretrain_ckpt_path)
    fold_train_score = fold_model.predict(full_train_gen)

    train_ids_by_day_df = pl.read_parquet(os.path.join(data_dir, 'train_ids_by_day.parquet'))
    train_ids_by_day_df = train_ids_by_day_df.with_columns(score=fold_train_score)
    
    os.makedirs(f'code/oof/xtf_v42_2fold_856', exist_ok=True)
    train_ids_by_day_df.drop('impression_day').write_parquet('code/oof/xtf_v42_2fold_856/train_score.parquet', use_pyarrow=True)
    del fold_model
    torch.cuda.empty_cache()
    print('predict training result end...')
                                     
    
    
    
