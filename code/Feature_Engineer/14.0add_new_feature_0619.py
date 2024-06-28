import polars as pl 
import pandas as pd 
import numpy as np 
from tqdm import tqdm 


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


def extract_article_listwise_feat(df, feat_list, bywhat):
    # 遍历特征列表
    for c in feat_list:
        print(c)
        # 为每个特征创建排序和反向排序的排名
        df = df.sort(by=bywhat+[c],descending=False)
        df = df.with_columns(
            pl.col(c).cum_count().over(bywhat).alias("_".join(bywhat) + '_article_' + c + '_rank')
        )

        df = df.sort(by=bywhat+[c],descending=True)
        df = df.with_columns(
            pl.col(c).cum_count().over(bywhat).alias("_".join(bywhat) + '_article_' + c + '_rank_reverse')
        )

        # 计算每个分组的特征均值，并合并回原始 DataFrame
        df_mean = df.group_by(bywhat).agg(
            pl.col(c).mean().alias("_".join(bywhat)  + '_' + c + '_mean')
        )
        df = df.join(df_mean, on=bywhat, how='left')

        # 计算特征与其均值的差
        df = df.with_columns(
            (pl.col(c) - pl.col("_".join(bywhat) + '_' + c + '_mean')).alias("_".join(bywhat) + '_' + c + '_mean_diff')
        )

    return df

def extract_article_impr_len_feature(df):
    df = df.with_columns(pl.col('user_impression_freq').mean().over('article_ids_inview').alias('article_inview_num_mean'))
    df = df.with_columns(pl.col('user_impression_freq').std().over('article_ids_inview').alias('article_inview_num_std'))
    
    df = df.with_columns(pl.col('user_impression_freq').mean().over('category').alias('category_inview_num_mean'))
    df = df.with_columns(pl.col('user_impression_freq').std().over('category').alias('category_inview_num_std'))
    
    
    df = df.with_columns(pl.col('user_impression_freq').mean().over(['impression_hour', 'article_ids_inview']).alias('impression_hour_article_inview_num_mean'))
    df = df.with_columns(pl.col('user_impression_freq').std().over(['impression_hour', 'article_ids_inview']).alias('impression_hour_article_inview_num_std'))
    
    df = df.with_columns(pl.col('user_impression_freq').mean().over(['imp_day','impression_hour', 'article_ids_inview']).alias('impression_day_hour_article_inview_num_mean'))
    df = df.with_columns(pl.col('user_impression_freq').std().over(['imp_day','impression_hour', 'article_ids_inview']).alias('impression_day_hour_article_inview_num_std'))
    
#     df = df.with_columns(
#         distance_publish_hours = df['distance_publish_seconds']//3600
#     )
    
    df = df.with_columns(
        df['distance_publish_hours'].clip(None, 48).alias('publish_48hour')
    )
    
    # df = df.with_columns(
    #     df['distance_publish_hours'].clip(None, 24).alias('publish_24hour')
    # )
    
    df = df.with_columns(pl.col('user_impression_freq').mean().over('publish_48hour', 'article_ids_inview').alias('publish_48hour_article_inview_num_mean'))
    df = df.with_columns(pl.col('user_impression_freq').std().over('publish_48hour', 'article_ids_inview').alias('publish_48hour_article_inview_num_std'))
   
    df = df.with_columns(pl.col('user_impression_freq').mean().over('device_type', 'article_ids_inview').alias('device_type_article_inview_num_mean'))
    df = df.with_columns(pl.col('user_impression_freq').std().over('device_type', 'article_ids_inview').alias('device_type_article_inview_num_std'))
    
    ####
    
    
    # df = df.with_columns(pl.col('impr_article_impr_hour_article_cnt_rank').mean().over(['article_ids_inview']).alias('article_app_position_by_hour_imp_rank'))
    
#     df = df.sort(['imp_day', 'impression_hour', 'impr_hour_article_cnt'])
#     df  = df.with_columns(pl.col('article_ids_inview').cum_count().over(['imp_day', 'impression_hour']).alias('article_hour_imp_count_rank'))
#     df = df.sort(['imp_day', 'impression_hour', 'impr_hour_article_cnt'], descending=True)
#     df  = df.with_columns(pl.col('article_ids_inview').cum_count().over(['imp_day', 'impression_hour']).alias('article_hour_imp_count_rank_reverse'))
    
#     # 全局天级别曝光的rank特征
#     df = df.sort(['imp_day', 'article_id_day_count'])
#     df = df.with_columns(pl.col('article_ids_inview').cum_count().over('imp_day').alias('article_daily_imp_count_rank'))
#     df = df.sort(['imp_day', 'article_id_day_count'], descending=True)
#     df = df.with_columns(pl.col('article_ids_inview').cum_count().over('imp_day').alias('article_daily_imp_count_rank_reverse'))


    return df



def gen_feature(df):
    # 处理阅读时间相关的特征
    
    df = df.with_columns([
    pl.col("read_time").fill_null(-1),
    pl.col("scroll_percentage").fill_null(-1)
        ])

    # 假设 df 已经是一个 Polars DataFrame
    df = df.with_columns(pl.col('read_time').mean().over('article_ids_inview').alias('article_read_time_mean'))
    df = df.with_columns(pl.col('read_time').std().over('article_ids_inview').alias('article_read_time_std'))
    
    df = df.with_columns(pl.col('read_time').mean().over(['imp_day', 'impression_hour', 'article_ids_inview']).alias('day_hour_article_read_time_mean'))
    df = df.with_columns(pl.col('read_time').std().over(['imp_day', 'impression_hour', 'article_ids_inview']).alias('day_hour_article_read_time_std'))
    
    df = df.with_columns(pl.col('read_time').mean().over(['publish_48hour', 'article_ids_inview']).alias('publish_48hour_article_read_time_mean'))
    df = df.with_columns( pl.col('read_time').std().over(['publish_48hour', 'article_ids_inview']).alias('publish_48hour_article_read_time_std'))
       
        
    df = df.with_columns(pl.col('scroll_percentage').mean().over('article_ids_inview').alias('article_scroll_percentage_mean'))
    df = df.with_columns(pl.col('scroll_percentage').std().over('article_ids_inview').alias('article_scroll_percentage_std'))
    
    df = df.with_columns(pl.col('scroll_percentage').mean().over(['imp_day', 'impression_hour', 'article_ids_inview']).alias('day_hour_article_scroll_percentage_mean'))
    df = df.with_columns(pl.col('scroll_percentage').std().over(['imp_day', 'impression_hour', 'article_ids_inview']).alias('day_hour_article_scroll_percentage_std'))
    
    df = df.with_columns(pl.col('scroll_percentage').mean().over(['publish_48hour', 'article_ids_inview']).alias('publish_48hour_article_scroll_percentage_mean'))
    df = df.with_columns( pl.col('scroll_percentage').std().over(['publish_48hour', 'article_ids_inview']).alias('publish_48hour_article_scroll_percentage_std'))
        

    print("agg count")
    # 使用 groupby 和 count 来计算每组的数量
    grouped_count = df.group_by(['user_id', 'impression_id', 'first_sub_category']).agg(
        pl.count('impression_time').alias('impr_first_sub_category_cnt')
    )

    # 将计算结果添加回原始 DataFrame
    df = df.join(
        grouped_count,
        on=['user_id', 'impression_id', 'first_sub_category'],
        how='left'
    )

    print("deal ....")
    feats = [
             "total_read_time",
             "w2v_each_cosine",
             "publish_48hour_article_inview_num_mean",
        "impression_day_hour_article_inview_num_mean",
             "his_subcate_cnt",
            "cl_each_cosine",
            # "article_pv_median",
            # "sentiment_score"
            ]
    df = extract_article_listwise_feat(df,feats,['user_id','impression_id'])
    
    

    return df

def part_merge_data(phase):
    # phase = "train"
    train_uids = pl.scan_parquet(f"../../dataset/{phase}_0614.parquet").select(['user_id']).collect().to_pandas()['user_id'].unique()
    train = pl.scan_parquet(f"../../dataset/{phase}_0614.parquet")
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
        
    ####his 
    # his = pl.read_parquet(f"./dataset/history_add_sentiment/{phase}/history.parquet").drop(['impression_time_fixed','scroll_percentage_fixed','article_id_fixed','read_time_fixed'])
        
    
    for ids in tqdm(grouped_ids):
        
        
        
    
        train_part = train.filter(pl.col("user_id").is_in(ids)).collect()
        print("part 1")
        train_part = extract_article_impr_len_feature(train_part)
        print("part 2...")
        train_part = gen_feature(train_part)
        
        train_part = reduce_memory_usage_pl(train_part)
        train_part = train_part.to_pandas()
        trains.append(train_part)
        
    
    trains = pd.concat(trains,axis=0,ignore_index=True)
    print(trains.shape)
    return trains

train = part_merge_data("train")
train.to_parquet("../../dataset/train_0619.parquet")
        
import gc
del train 
gc.collect()

valid = part_merge_data("validation")
valid.to_parquet("../../dataset/validation_0619.parquet")

del valid 
gc.collect()

test = part_merge_data("test")
test.to_parquet("../../dataset/test_0619.parquet")

del test 
gc.collect()

