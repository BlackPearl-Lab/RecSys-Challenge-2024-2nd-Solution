##### 新的特征 
import numpy as np
import pandas as pd
import polars as pl
import os  
from tqdm import tqdm  
import gc 


def pageindex_fea(df):
    
    
    ##### 
    df_unique = df[['user_id', 'impression_id', 'impression_time','read_time','user_impression_freq']].drop_duplicates()
    df_unique = df_unique.sort_values(['user_id', 'impression_time'])
    df_unique['user_impression_freq_last_fix'] = df_unique.groupby(['user_id'])['user_impression_freq'].shift(-1)
    df_unique['user_impression_freq_next_fix'] = df_unique.groupby(['user_id'])['user_impression_freq'].shift(1)
    
    df_unique['read_time_last_fix'] = df_unique.groupby(['user_id',])['read_time'].shift(-1)
    df_unique['read_time_next_fix'] = df_unique.groupby(['user_id'])['read_time'].shift(1)
    df_unique = df_unique[['user_id', 'impression_id','user_impression_freq_last_fix','user_impression_freq_next_fix','read_time_last_fix','read_time_next_fix']]
    
    df = df.merge(df_unique, on=["impression_id","user_id"], how="left")
    
    
    
    return df

def gen_new_feature(df,phase):
#     df = df.sort_values(['session_id', 'impression_time'])
#     df['is_first_impression'] = df.groupby(['session_id', 'impression_id']).cumcount() == 0
#     df.loc[df['is_first_impression'], 'session_id_impression_rank'] = df.loc[df['is_first_impression']].groupby('session_id').cumcount() + 1
#     df = df.drop('is_first_impression',axis=1)
#     df['session_id_impression_rank'] = df['session_id_impression_rank'].fillna(method='ffill')
    
    
    
    ###session 
    df['user_impression_cnt'] = df.groupby(['user_id'])['impression_id'].transform('nunique')
    df['user_session_cnt'] = df.groupby(['user_id'])['session_id'].transform('nunique')
    #####
    df['article_id_day_count'] = df.groupby(['article_ids_inview','imp_day'],as_index=False)['user_id'].transform("count")
    ######
    for stats in ['mean',"std"]:
        df['article_inviews_len_'+stats] = df.groupby(['article_ids_inview'],as_index=False)['user_impression_freq'].transform(stats)
    
    #######
    df_unique = df[['user_id', 'session_id','impression_id', 'impression_time',]].drop_duplicates()
    df_unique = df_unique.sort_values(['user_id', 'session_id','impression_time'])
    df_unique['next_pred_readtime'] = df_unique.groupby(['user_id','session_id'],as_index=False)['impression_time'].diff().dt.total_seconds()
    df = df.merge(df_unique[['user_id','session_id','impression_id','next_pred_readtime']],on=['user_id','session_id','impression_id'],how="left")
    # df = df.sort_values(by=['user_id','session_id',"impression_time"])
    # df['next_pred_readtime'] = df.groupby(['user_id','session_id'],as_index=False)['impression_time'].diff().dt.total_seconds()
    
    df = pageindex_fea(df)
    his_cnt = pd.read_parquet(f"../../features/{phase}_his_cate_feature.parquet")
    df = df.merge(his_cnt,on = ['impression_id','article_ids_inview'],how="left")
    
    
    
    
    return df

def part_gen_data(phase):
    # phase = "train"
    train_uids = pl.scan_parquet(f"../../dataset/{phase}.parquet").select(['user_id']).collect().to_pandas()['user_id'].unique()
    train = pl.scan_parquet(f"./dataset/{phase}.parquet")
    non_train = train.filter(pl.col("impression_id")==0).collect().to_pandas()
    print(non_train.shape)
    train = train.filter(pl.col("impression_id")!=0)
    # print(train.shape)
    
    batch_size = 200000
    grouped_ids = [train_uids[i:i + batch_size] for i in range(0, len(train_uids), batch_size)]
    article_stats = pl.scan_parquet("../../features/article_num_stats.parquet")
    if phase == "test":
        trains = [non_train]
    else:
        trains = []
    for ids in tqdm(grouped_ids):
    
        train_part = train.filter(pl.col("user_id").is_in(ids))
        train_part = train_part.join(article_stats,left_on ="article_ids_inview" ,right_on="article_id",how="left")
        train_part = train_part.collect().to_pandas()
        train_part = gen_new_feature(train_part,phase)
        trains.append(train_part)
    
    trains = pd.concat(trains,axis=0,ignore_index=True)
    return trains

train = part_gen_data("train")
train.to_parquet("../../dataset/train_0529.parquet")

del train 
gc.collect()

valid = part_gen_data("validation")
valid.to_parquet("../../dataset/validation_0529.parquet")

del valid 
gc.collect()

test = part_gen_data("test")
test.to_parquet("../../dataset/test_0529.parquet")
del test 
gc.collect()

        
        
        
    