from pathlib import Path
import polars as pl
import numpy as np
import os
from datetime import datetime
import gc
import pandas as pd
import warnings
import argparse
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='all', help='mode: all/train/validation/test, only use the argument when re_load is true')
parser.add_argument('--base_dataset_path', type=str, default='dataset/', help='path of full features for catboost training...')
args = vars(parser.parse_args())
mode = args['mode']
lgb_feat_path = args['base_dataset_path']


os.chdir('../../')

base_dataset = 'nn_data'

root = 'inputs'
data_path = f"{root}/large"
train_path = f"{data_path}/train"
dev_path = f"{data_path}/validation"
test_path = f"{data_path}/test"
vector_path = f"{root}/vectors"
feature_path = f"features"

os.makedirs(f'{feature_path}/{base_dataset}/imp', exist_ok=True)


lgb_feats_pd = pd.read_csv(f'{feature_path}/imp_catboost_0620_pair_rank_loss_0.8601_filter_feature.csv').rename(columns={'Unnamed: 0': "feat"})
lgb_feats_list = lgb_feats_pd['feat'].tolist()
print(f'feat_len={len(lgb_feats_list)}')

news = pl.read_parquet(f'{feature_path}/{base_dataset}/article/article_feats.parquet')
news = news.select(['article_id', 'category', 'subcat1','article_type','premium', 'sentiment_label'])
news_has_cols = [c for c in pl.scan_parquet(f'./{feature_path}/{base_dataset}/article/article_feats.parquet').columns if c!='article_id']


def preprocess(data_path, mode):
    full_df = pl.scan_parquet(f'{lgb_feat_path}/{mode}_0620.parquet')
    
    full_df = (full_df
                .select(['impression_id', 'user_id', 'click'] + lgb_feats_list)
                .rename({'article_ids_inview': 'article_id'})
               ).drop([c for c in full_df.columns if c in news_has_cols]).collect() # delete article features which will be joined during the sample generation process later
    print(full_df.shape)
    print(full_df.head(5))
    
    behavior_file = os.path.join(data_path, "behaviors.parquet")
    sample_df = pl.scan_parquet(behavior_file).select(['impression_id', 'user_id', 'impression_time', 'session_id']).collect()
    print('begin join feature')
    print(sample_df.head(5))

    
    sample_df = sample_df.join(full_df, how='left', on=['impression_id', 'user_id'])
    print('join feature done...')
    
    del full_df
    gc.collect()
    
    sample_df = sample_df.with_columns(pl.col('impression_time').dt.date().alias('date'))
    sample_df = sample_df.with_row_count('index')
    print('impression indexing...')
    
    # 特征工程
    sample_df = sample_df.sort(by=['user_id', 'impression_time'])
    sample_df = sample_df.with_columns(pl.col('impression_id').count().over("session_id").alias("session_impression_id_num"))  
    sample_df = sample_df.with_columns(pl.col('impression_id').cum_count().over('user_id', 'date').alias('impression_id_in_userdaily_rank'))
    # next_read_time
    sample_df = sample_df.with_columns(pl.col('impression_time').shift(-1).over(['user_id', 'session_id']).alias('next_impression_time'))
    sample_df = sample_df.with_columns(predict_next_read_time=(pl.col("next_impression_time")-pl.col("impression_time")).dt.total_seconds() - pl.col("read_time"))
    sample_df = sample_df.drop('next_impression_time')

    # session duration bias
    sample_df = sample_df.with_columns(session_min_impression_time=pl.col('impression_time').min().over('session_id'), session_max_impression_time=pl.col('impression_time').max().over('session_id'))
    sample_df = sample_df.with_columns(session_duration_time=(pl.col('session_max_impression_time')-pl.col('session_min_impression_time')).dt.total_minutes())
    # user x daily
    sample_df = sample_df.with_columns(pl.col('impression_id').n_unique().over(['user_id', 'date']).alias('user_daily_impression_id_num'))
    sample_df = sample_df.with_columns(pl.col('read_time').sum().over(['user_id', 'date']).alias('user_daily_total_read_time'))
    sample_df = sample_df.drop(['session_min_impression_time', 'session_max_impression_time'])
    
    sample_df = sample_df.sort('index')
    sample_df = sample_df.drop(['index'])
    print('recover index done')
    
    sample_df = sample_df.with_columns(impression_day=(pl.col('impression_time') - pl.lit(datetime(2023, 5, 18, 7, 0, 0))).dt.total_days().cast(pl.Int8))
    sample_df = sample_df.drop('impression_time')
    
    print(sample_df.shape)
    print(sample_df.head(5))
    
    sample_df = sample_df.with_row_count('index')
    print('item index...')

    sample_df = sample_df.join(news, on=['article_id'], how='left')
    # user_daily_cate_count, hour_cate_count
    sample_df = sample_df.with_columns(pl.col('article_id').count().over(['user_id', 'date', 'category']).alias('user_daily_category_count'))
    sample_df = sample_df.with_columns(pl.col('article_id').count().over(['user_id', 'date', 'subcat1']).alias('user_daily_subcat1_count'))
    sample_df = sample_df.with_columns(pl.col('article_id').count().over(['user_id', 'date', 'article_type']).alias('user_daily_articletype_count'))
    sample_df = sample_df.with_columns(pl.col('article_id').count().over(['user_id', 'date', 'premium']).alias('user_daily_premium_count'))
    sample_df = sample_df.with_columns(pl.col('article_id').count().over(['user_id', 'date', 'sentiment_label']).alias('user_daily_sentiment_count'))
    gc.collect()
    sample_df = sample_df.with_columns(pl.col('article_id').count().over(['user_id', 'impression_hour', 'category']).alias('user_hour_category_count'))
    sample_df = sample_df.with_columns(pl.col('article_id').count().over(['user_id', 'impression_hour', 'subcat1']).alias('user_hour_daily_subcat1_count'))
    sample_df = sample_df.with_columns(pl.col('article_id').count().over(['user_id', 'impression_hour', 'article_type']).alias('user_hour_articletype_count'))
    sample_df = sample_df.with_columns(pl.col('article_id').count().over(['user_id', 'impression_hour', 'premium']).alias('user_hour_premium_count'))
    sample_df = sample_df.with_columns(pl.col('article_id').count().over(['user_id', 'impression_hour', 'sentiment_label']).alias('user_hour_sentiment_count'))
    gc.collect()
    print('extract user x side info DONE')
    
    sample_df = sample_df.drop(['category', 'subcat1','article_type','premium', 'sentiment_label'])
    gc.collect()
    
    sample_df = sample_df.sort('index')
    sample_df = sample_df.drop(['index'])
    print('recover impression-item item index done')
    
    sample_df = sample_df.drop(['date', 'impression_time', 'session_id'])
    print(sample_df.columns)
    print(sample_df.head(5))
    
    return sample_df



# You can launch multiple processes to run train/validation/test in parallel
if mode in ('all', 'train'):
    print('begin extract training imp features....')
    df_train_imp = preprocess(train_path, 'train')
    df_train_imp.write_parquet(f'{feature_path}/{base_dataset}/imp/train_imp.parquet', use_pyarrow=True)
    del df_train_imp
    gc.collect()
    print('extract training imp features done....')

if mode in ('all', 'validation'):
    print('begin extract validation imp features....')
    df_val_imp = preprocess(dev_path, 'validation')
    df_val_imp.write_parquet(f'{feature_path}/{base_dataset}/imp/valid_imp.parquet', use_pyarrow=True)
    del df_val_imp
    gc.collect()
    print('extract validation imp features done....')

if mode in ('all', 'test'):
    print('extract test imp features done....')
    df_test_imp = preprocess(test_path, 'test')
    df_test_imp.write_parquet(f'{feature_path}/{base_dataset}/imp/test_imp.parquet', use_pyarrow=True)
    del df_test_imp
    gc.collect()
    print('extract test imp features done....')
    
