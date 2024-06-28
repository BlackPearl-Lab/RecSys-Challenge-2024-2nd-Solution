import polars as pl 
import pandas as pd 
import numpy as np 
from tqdm import tqdm 
import gc
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

# v26
def extract_session_article_other_feat(df):
    cols = ['user_impression_freq',"bpr","ctr","total_inviews","total_pageviews",'distance_publish_seconds','distance_published_hour_cos','impr_hour_article_cnt']
    for c in cols:
        df = df.sort_values(by=['impression_id',c])
        print(c+'_rank')
        df['impr_article_'+c+'_rank'] = df.groupby(['user_id','impression_id']).cumcount()+1
        print(c+'_rank_reverse')
        df = df.sort_values(by=['impression_id',c],ascending=False)
        df['impr_article_'+c+'_rank_reverse'] = df.groupby(['user_id','impression_id']).cumcount()+1
    return df



# v27
def extract_article_imprs(df):
    
    df['article_imprs_user'] = df.groupby(['user_id','article_ids_inview'])['impression_time'].transform("count")
    df['category_imprs_user'] = df.groupby(['user_id','category'])['impression_time'].transform("count")
    
    # 按照时间排序，计算每个article在同一个session中的出现顺序
    print('user_article_impr_time_rank')
    df = df.sort_values(by=['user_id', 'article_ids_inview','impression_time'])
    df['user_article_impr_time_rank'] = df.groupby(['user_id','article_ids_inview']).cumcount()+1
    print('user_article_impr_time_rank_reverse')
    df = df.sort_values(by=['user_id', 'article_ids_inview','impression_time'],ascending=False)
    df['user_article_impr_time_rank_reverse'] = df.groupby(['user_id','article_ids_inview']).cumcount()+1

    # 按照时间排序，计算每个article与上一次出现的时间差
    print('user_article_impr_time_diff')
    df = df.sort_values(by=['user_id', 'article_ids_inview','impression_time'])
    df['user_article_impr_time_diff'] = df.groupby(['user_id','article_ids_inview'])['impression_time'].diff().dt.total_seconds()
    
    # df.drop(columns=['user_id'],inplace=True)
    return df

def part_merge_data(phase):
    # phase = "train"
    train_uids = pl.scan_parquet(f"../../dataset/{phase}_0608.parquet").select(['user_id']).collect().to_pandas()['user_id'].unique()
    train = pl.scan_parquet(f"../../dataset/{phase}_0608.parquet")
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
        
    
    for ids in tqdm(grouped_ids):
        
        
        
    
        train_part = train.filter(pl.col("user_id").is_in(ids)).collect().to_pandas()
        
        
        train_part['category_freq_fix'] = train_part.groupby('category')['user_id'].transform('count')
        
        ###### session_article_cnt
        train_part['session_category_cnt'] = train_part.groupby(['user_id','session_id', 'category'])['impression_time'].transform("count")
        train_part['impr_category_cnt'] = train_part.groupby(['user_id','impression_id', 'category'])['impression_time'].transform("count")
        train_part['impr_hour_category_cnt'] = train_part.groupby(['category','imp_day','impression_hour'],as_index=False)['user_id'].transform("count")
        
        train_part = extract_session_article_other_feat(train_part)
        train_part = extract_article_imprs(train_part)
       
        

        train_part = pl.from_pandas(train_part)
        train_part = reduce_memory_usage_pl(train_part)
        train_part = train_part.to_pandas()
        trains.append(train_part)
        
    
    trains = pd.concat(trains,axis=0,ignore_index=True)
    print(trains.shape)
    return trains

train = part_merge_data("train")
train.to_parquet("../../dataset/train_0614.parquet")


del train 
gc.collect()

valid = part_merge_data("validation")
valid.to_parquet("../../dataset/validation_0614.parquet")

del valid 
gc.collect()

test = part_merge_data("test")
test.to_parquet("../../dataset/test_0614.parquet")

del test 
gc.collect()