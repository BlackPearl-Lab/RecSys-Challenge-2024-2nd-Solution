#####获取article的特征
####统计article的历史点击数据
import pandas as pd
import polars as pl
import numpy as np
import os

def reduce_memory_usage_pl(df, verbose=1):
    """Reduce memory usage by polars dataframe {df} with name {name} by changing its data types.
    Original pandas version of this function: https://www.kaggle.com/code/arjanso/reducing-dataframe-memory-size-by-65
    """

    mem1 = round(df.estimated_size("gb"), 2)
    Numeric_Int_types = [pl.Int8, pl.Int16, pl.Int32, pl.Int64]
    Numeric_Float_types = [pl.Float32, pl.Float64]
    for col in df.columns:
        try:
            col_type = df[col].dtype
            if col_type == pl.Categorical:
                continue
            c_min = df[col].min()
            c_max = df[col].max()
            if col_type in Numeric_Int_types:
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df = df.with_columns(df[col].cast(pl.Int8))
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df = df.with_columns(df[col].cast(pl.Int16))
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df = df.with_columns(df[col].cast(pl.Int32))
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df = df.with_columns(df[col].cast(pl.Int64))
            elif col_type in Numeric_Float_types:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df = df.with_columns(df[col].cast(pl.Float32))
                else:
                    pass
            else:
                pass
        except:
            pass
    if verbose:
        mem2 = round(df.estimated_size("gb"), 2)
        ratio = round((mem1 - mem2) / mem1 * 100, 2)
        print(f"Memory usage of dataframe {mem1} GB ---> {mem2} GB, less {ratio}%")
    return df

def gen_his_article_feature(data):
    data = data.explode(['article_id_fixed','read_time_fixed'])
    article_feature = data.groupby(['article_id_fixed']).agg({"user_id":['count',"nunique"],"read_time_fixed":['sum','std',"skew"]})
    article_feature.columns = ['_'.join(col).strip() for col in article_feature.columns.values]
    article_feature['article_user_avg_clk'] = article_feature['user_id_count'] / article_feature['user_id_nunique'] 
    return article_feature

def gen_article_clk_feature(phase):
    train_his = pd.read_parquet(f"../../inputs/large/{phase}/history.parquet")
    train_his = train_his[['user_id','article_id_fixed','read_time_fixed']]
    train_article_feature = gen_his_article_feature(train_his)
    train_article_feature = train_article_feature.reset_index()
    train = pl.read_parquet(f"../../caches/{phase}.parquet").to_pandas()
    train = train.merge(train_article_feature,left_on='article_ids_inview',right_on='article_id_fixed',how="left")
    return train



def add_feature(data):
    ###ctr 特征
    data.fillna({'total_pageviews': 0, 'total_inviews': 0}, inplace=True)
    data['ctr'] = np.where(data['total_inviews'] == 0, 0, data['total_pageviews'] / data['total_inviews'])
    ####user count
    # data['user_freq'] = data.groupby('user_id')['user_id'].transform('count')
    data['user_impression_freq'] = data.groupby(['impression_id','user_id'])['user_id'].transform('count')
    # data['user_impression_cnt'] = data.groupby(['impression_id','user_id'])['user_id'].transform('nunique')
    data['article_freq'] = data.groupby('article_ids_inview')['article_ids_inview'].transform('count')
    
    data['user_article_freq'] = data.groupby('user_id')['article_ids_inview'].transform('nunique')
    data['article_user_freq'] = data.groupby('article_ids_inview')['user_id'].transform('nunique')
    
    day_min = data['impression_time'].dt.day.min()
    data['imp_day'] = data['impression_time'].dt.day-day_min
    
    return data
    

def sim_feature(phase):
    d1 = pl.scan_parquet(f"../../features/{phase}_sim_0.parquet")
    d2 = pl.scan_parquet(f"../../features/{phase}_sim_1.parquet")
    d3 = pl.scan_parquet(f"../../features/{phase}_sim_2.parquet")
    d = pl.concat([d1,d2,d3]).drop(['impression_id', 'user_id'])
    list_name = [i for i in d.columns if "_list" in i]
    d = d.explode(list_name).collect()
    d = reduce_memory_usage_pl(d)
    d = d.to_pandas()
    return d 

def pageindex_fea(df):
    df_unique = df[['user_id', 'session_id', 'impression_id', 'impression_time']].drop_duplicates()
    sess_df = df_unique.sort_values(['session_id', 'impression_time'])
    sess_df['session_impression_rank'] = sess_df.groupby('session_id').cumcount() + 1
    
    df_unique['impression_date'] = df_unique['impression_time'].dt.date

    date_df = df_unique.sort_values(['impression_date', 'user_id', 'impression_time'])
    date_df['date_impression_rank'] = date_df.groupby(['impression_date', 'user_id']).cumcount() + 1

    df_unique['session_impression_rank'] = sess_df['session_impression_rank']
    df_unique['date_impression_rank'] = date_df['date_impression_rank']
    
    df_unique = df_unique[['impression_id', 'user_id','session_impression_rank', 'date_impression_rank']]

    df = df.merge(df_unique, on=["impression_id","user_id"], how="left")
    
    ##### 
#     df_unique = df[['user_id', 'impression_id', 'impression_time','read_time','user_impression_freq']].drop_duplicates()
#     df_unique = df_unique.sort_values(['user_id', 'impression_time'])
#     df_unique['user_impression_freq_last'] = df_unique.groupby(['user_id'])['user_impression_freq'].shift(-1)
#     df_unique['user_impression_freq_next'] = df_unique.groupby(['user_id'])['user_impression_freq'].shift(1)
    
#     df_unique['read_time_last'] = df_unique.groupby(['user_id',])['read_time'].shift(-1)
#     df_unique['read_time_next'] = df_unique.groupby(['user_id'])['read_time'].shift(1)
#     df_unique = df_unique[['user_id', 'impression_id','user_impression_freq_last','user_impression_freq_next','read_time_last','read_time_next']]
    
#     df = df.merge(df_unique, on=["impression_id","user_id"], how="left")
    
    
    
    return df



def gen_cross_feature(data):
    for col in ['ctr','bpr','user_impression_freq','distance_publish_hours','w2v_each_cosine','total_inviews']:
        for stats in ['mean','std','skew']:
            data[f'user_{col}_{stats}'] = data.groupby("user_id")[col].transform(stats)
    return data


import  gc

train = gen_article_clk_feature("train")
train = add_feature(train)
train_sim = sim_feature("train")
train = train.join(train_sim)

del train_sim
gc.collect()
train_bpr= np.load("../../features/train-bpr.npy")
train['bpr'] = train_bpr[:len(train)]

train = gen_cross_feature(train)
train = pageindex_fea(train)
train = pl.from_pandas(train)
train = reduce_memory_usage_pl(train).to_pandas()
train.to_parquet("../../caches/lgb_train_new.parquet")

import gc 
del train,train_bpr
gc.collect()


valid = gen_article_clk_feature("validation")
valid = add_feature(valid)
val_sim = sim_feature("val")
valid = valid.join(val_sim)
del val_sim
gc.collect()

valid_bpr= np.load("../../features/validation-bpr.npy")
valid['bpr'] = valid_bpr[:len(valid)]

valid = gen_cross_feature(valid)
valid = pageindex_fea(valid)
valid = pl.from_pandas(valid)
valid = reduce_memory_usage_pl(valid).to_pandas()
valid.to_parquet("../../caches/lgb_valid_new.parquet")

del valid,valid_bpr
gc.collect()

test = gen_article_clk_feature("test")
test = add_feature(test)
test_sim = sim_feature("test")
assert len(test) == len(test_sim)
test = pl.from_pandas(test)
test = reduce_memory_usage_pl(test)

test_sim = pl.from_pandas(test_sim)
test_sim = reduce_memory_usage_pl(test_sim)

from tqdm import tqdm 
cols = list(test_sim.columns)
for col in tqdm(cols):
    try:
        test[col] = test_sim.with_columns(test_sim[col].alias(col))
        test_sim = test_sim.drop(col)
    except:
        pass

del test_sim
gc.collect()



test_bpr = np.load("../../features/test-bpr.npy")
test['bpr'] = test_bpr

test = gen_cross_feature(test)
test = pageindex_fea(test)

test.to_parquet("../../caches/lgb_test_new.parquet")

del test,test_bpr
gc.collect()



