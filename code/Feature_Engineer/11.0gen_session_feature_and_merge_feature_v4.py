import polars as pl 
import pandas as pd 
import numpy as np 
from tqdm import tqdm 


#####获取鹏哥v16 v17 的特征
def gen_session_feature(phase):
    
    tr_be = pl.scan_parquet(f"../../inputs/large/{phase}/behaviors.parquet")
    non_train = tr_be.filter(pl.col("impression_id")==0).select(['session_id','user_id',"impression_id"]).collect().to_pandas()
    print(non_train.shape)
    tr_be = tr_be.filter(pl.col("impression_id")!=0).select(['session_id','user_id',"impression_id",'impression_time',"article_ids_inview","read_time"])
    tr_be = tr_be.collect().to_pandas()
    # print(train.shape)
    if phase == "test":
        trains = [non_train]
    else:
        trains = []
        
    tr_be['articles_len'] = tr_be['article_ids_inview'].map(len)
    tr_be = tr_be.sort_values(by=['user_id','session_id', 'impression_time'])
    

    tr_be['session_id_articles_len_last'] = tr_be.groupby(['user_id',"session_id"])['articles_len'].diff(1)
    tr_be['session_id_articles_len_next'] = tr_be.groupby(['user_id',"session_id"])['articles_len'].diff(-1)


    tr_be['session_id_read_time_last'] = tr_be.groupby(['user_id',"session_id"])['read_time'].diff(1)
    tr_be['session_id_read_time_next'] = tr_be.groupby(['user_id',"session_id"])['read_time'].diff(-1)


    tr_be['session_id_impression_time_last'] = tr_be.groupby(['user_id',"session_id"])['impression_time'].diff(1).dt.total_seconds()
    tr_be['session_id_impression_time_next'] = tr_be.groupby(['user_id',"session_id"])['impression_time'].diff(-1).dt.total_seconds()

    tr_be['mod_read_time'] = tr_be['read_time']+tr_be['session_id_impression_time_next']
    
    tr_be = tr_be.drop(['impression_time','article_ids_inview','read_time','articles_len'],axis=1)
    trains.append(tr_be)
    
    trains = pd.concat(trains,axis=0,ignore_index=True)
    # print(trains.head(10))
    return trains

# ####train
train = gen_session_feature("train")
train.to_parquet("../../features/train_yp_v1617_session_feature.parquet")

# ####valid
valid = gen_session_feature("validation")
valid.to_parquet("../../features/validation_yp_v1617_session_feature.parquet")

###test
test = gen_session_feature("test")
test.to_parquet("../../features/test_yp_v1617_session_feature.parquet")


# 同一个session下，article的特征
def extract_session_article_feat(df):
    cols = ["ctr","total_inviews","total_pageviews",'distance_publish_seconds','distance_published_hour_cos']
    df_feat = df[cols+['user_id','session_id','impression_id','article_ids_inview']].drop_duplicates()
    for c in cols:
        df_feat[c+'_mean'] = df_feat.groupby(['user_id','session_id'])[c].transform('mean')
    df_feat.drop(columns=cols,inplace=True)
    df = df.merge(df_feat,on=['user_id','session_id','impression_id','article_ids_inview'],how='left')
    for c in cols:
        df[f'session_{c}_mean_diff'] = df[c] - df[c+'_mean']
        
    return df

# 同一个session下，article的特征
def extract_impr_article_feat(df):
    cols = ["ctr","total_inviews","total_pageviews",'distance_publish_seconds','distance_published_hour_cos']
    df_feat = df[cols+['user_id','impression_id','article_ids_inview']].drop_duplicates()
    for c in cols:
        df_feat[c+'_mean_impr'] = df_feat.groupby(['user_id','impression_id'])[c].transform('mean')
    df_feat.drop(columns=cols,inplace=True)
    df = df.merge(df_feat,on=['user_id','impression_id','article_ids_inview'],how='left')
    for c in cols:
        df[f'impression_{c}_mean_impr_diff'] = df[c] - df[c+'_mean_impr']
        
    return df

# 同一个session下，article的出现次数
def extract_article_imprs(df):
    
    # 按照时间排序，计算每个article在同一个session中的出现顺序
    print('session_article_impr_time_rank')
    df = df.sort_values(by=['user_id','session_id', 'article_ids_inview','impression_time'])
    df['session_article_impr_time_rank'] = df.groupby(['session_id','article_ids_inview']).cumcount()+1
    print('session_article_impr_time_rank_reverse')
    df = df.sort_values(by=['user_id','session_id', 'article_ids_inview','impression_time'],ascending=False)
    df['session_article_impr_time_rank_reverse'] = df.groupby(['session_id','article_ids_inview']).cumcount()+1

    # 按照时间排序，计算每个article与上一次出现的时间差
    print('session_article_impr_time_diff')
    df = df.sort_values(by=['user_id','session_id', 'article_ids_inview','impression_time'])
    df['session_article_impr_time_diff'] = df.groupby(['session_id','article_ids_inview'])['impression_time'].diff().dt.total_seconds()
    
    # print(df.head(5))
    
    return df

#####获取session_article的特征
def gen_session_article_feature(phase):
    
    
    train = pl.scan_parquet(f"../../dataset/{phase}_0529.parquet").select(['impression_id',
                "ctr","total_inviews","total_pageviews",'distance_publish_seconds','distance_published_hour_cos',                                      
                'session_id','user_id','impression_time','article_ids_inview'])
    non_train = train.filter(pl.col("impression_id")==0).select(['impression_id','session_id','user_id']).collect().to_pandas()
    train = train.filter(pl.col("impression_id")!=0)
    
    train = train.collect().to_pandas()
    
    train = extract_article_imprs(train)
    train = extract_session_article_feat(train) 
    train = extract_impr_article_feat(train)
    train = train.drop(['impression_time',"ctr","total_inviews","total_pageviews",'distance_publish_seconds','distance_published_hour_cos'],axis=1)
    print(train.head())
    
    if phase == "test":
        trains = [non_train]
    else:
        trains = []
        
    trains.append(train)
    trains = pd.concat(trains,axis=0,ignore_index=True)
    # print(trains.head(10))
    return trains

train_session_article = gen_session_article_feature("train")
train_session_article.to_parquet("../../features/train_session_article_feature.parquet")
import gc
del train_session_article
gc.collect()

train_session_article = gen_session_article_feature("validation")
train_session_article.to_parquet("../../features/validation_session_article_feature.parquet")

import gc
del train_session_article
gc.collect()

train_session_article = gen_session_article_feature("test")
train_session_article.to_parquet("../../features/test_session_article_feature.parquet")

import gc
del train_session_article
gc.collect()

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

def part_merge_data(phase):
    # phase = "train"
    train_uids = pl.scan_parquet(f"../../dataset/{phase}_0529.parquet").select(['user_id']).collect().to_pandas()['user_id'].unique()
    train = pl.scan_parquet(f"../../dataset/{phase}_0529.parquet")
    non_train = train.filter(pl.col("impression_id")==0).collect()
    non_train = non_train.with_columns(pl.col("article_ids_inview").cast(pl.Int32)).to_pandas()
    print(non_train.shape)
    train = train.filter(pl.col("impression_id")!=0)
    train = train.with_columns(pl.col("article_ids_inview").cast(pl.Int32))
    # print(train.shape)
    
    batch_size = 200000
    grouped_ids = [train_uids[i:i + batch_size] for i in range(0, len(train_uids), batch_size)]
    if phase == "test":
        trains = [non_train]
    else:
        trains = []
        
    session_tr = pl.scan_parquet(f"../../features/{phase}_yp_v1617_session_feature.parquet")
    # session_tr = session_tr.with_columns(pl.col("article_ids_inview").cast(pl.Int32))
    # print(list(session_tr.columns))
    session_article_tr = pl.scan_parquet(f"../../features/{phase}_session_article_feature.parquet")
    session_article_tr = session_article_tr.with_columns(pl.col("article_ids_inview").cast(pl.Int32))
    # print(list(session_article_tr.columns))
    
    for ids in tqdm(grouped_ids):
    
        train_part = train.filter(pl.col("user_id").is_in(ids))
        train_part = train_part.join(session_tr,on=['session_id','user_id',"impression_id"],how="left")

        train_part = train_part.join(session_article_tr,on=['impression_id', 'session_id', 'user_id', 'article_ids_inview'],how="left")
        train_part = train_part.collect()
        train_part = reduce_memory_usage_pl(train_part)
        train_part = train_part.to_pandas()
        trains.append(train_part)
        
    
    trains = pd.concat(trains,axis=0,ignore_index=True)
    print(trains.shape)
    return trains
        

    
train = part_merge_data("train")
train.to_parquet("../../dataset/train_0605.parquet")

del train 
gc.collect()

valid = part_merge_data("validation")
valid.to_parquet("../../dataset/validation_0605.parquet")

del valid 
gc.collect()

test = part_merge_data("test")
test.to_parquet("../../dataset/test_0605.parquet")

del test 
gc.collect()
