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


def gen_feature(df):
    df = df.with_columns(pl.col('user_impression_freq').median().over('imp_day','impression_hour', 'article_ids_inview').alias('impression_day_hour_article_inview_num_median'))
    df = df.with_columns(pl.col('user_impression_freq').skew().over('imp_day','impression_hour', 'article_ids_inview').alias('impression_day_hour_article_inview_num_skew'))
    df = df.with_columns(pl.col('user_impression_freq').max().over('imp_day','impression_hour', 'article_ids_inview').alias('impression_day_hour_article_inview_num_max'))
    df = df.with_columns(pl.col('user_impression_freq').min().over('imp_day','impression_hour', 'article_ids_inview').alias('impression_day_hour_article_inview_num_min'))
    df = df.with_columns(impression_day_hour_article_inview_num_max_min = pl.col('impression_day_hour_article_inview_num_max') - pl.col("impression_day_hour_article_inview_num_min"))
    
    
    df = df.with_columns(pl.col('user_impression_freq').mean().over('imp_day','impression_hour','device_type', 'article_ids_inview').alias('impression_day_hour_device_type_article_inview_num_mean'))
    df = df.with_columns(pl.col('user_impression_freq').std().over('imp_day','impression_hour','device_type', 'article_ids_inview').alias('impression_day_hour_device_type_article_inview_num_std'))
    df = df.with_columns(pl.col('user_impression_freq').median().over('imp_day','impression_hour','device_type', 'article_ids_inview').alias('impression_day_hour_device_type_article_inview_num_median'))
    df = df.with_columns(pl.col('user_impression_freq').skew().over('imp_day','impression_hour','device_type', 'article_ids_inview').alias('impression_day_hour_device_type_article_inview_num_skew'))
    df = df.with_columns(pl.col('user_impression_freq').max().over('imp_day','impression_hour','device_type', 'article_ids_inview').alias('impression_day_hour_device_type_article_inview_num_max'))
    df = df.with_columns(pl.col('user_impression_freq').min().over('imp_day','impression_hour','device_type', 'article_ids_inview').alias('impression_day_hour_device_type_article_inview_num_min'))
    
    df = df.with_columns(pl.col('user_impression_freq').skew().over('publish_48hour', 'article_ids_inview').alias('publish_48hour_article_inview_num_skew'))
    df = df.with_columns(pl.col('user_impression_freq').max().over('publish_48hour', 'article_ids_inview').alias('publish_48hour_article_inview_num_max'))
    df = df.with_columns(pl.col('user_impression_freq').min().over('publish_48hour', 'article_ids_inview').alias('publish_48hour_article_inview_num_min'))
    df = df.with_columns(pl.col('user_impression_freq').std().over('publish_48hour',"device_type", 'article_ids_inview').alias('publish_48hour_device_type_article_inview_num_std'))
    
    
    df = df.with_columns(pl.col('user_impression_freq').mean().over('imp_day','impression_hour',"publish_48hour", 'article_ids_inview').alias('impression_day_hour_publish_48hour_article_inview_num_mean'))
    df = df.with_columns(pl.col('user_impression_freq').std().over('imp_day','impression_hour', "publish_48hour",'article_ids_inview').alias('impression_day_hour_publish_48hour_article_inview_num_std'))
    
    df = df.with_columns(pl.col('user_impression_freq').mean().over('imp_day', 'article_ids_inview').alias('impression_day_article_inview_num_mean'))
    df = df.with_columns(pl.col('user_impression_freq').std().over('imp_day', 'article_ids_inview').alias('impression_day_article_inview_num_std'))
    
    df = df.with_columns(article_hour_cnt_with_ctr = pl.col("impr_hour_article_cnt")*pl.col("ctr"))
    # df = df.with_columns(target_fit = pl.col("impr_hour_article_cnt")*pl.col("ctr")*pl.col("article_pv_median")*pl.col("bpr")*pl.col("w2v_each_cosine"))
    
    print("deal ....")
    feats = [
             'article_hour_cnt_with_ctr',
            # "target_fit"
            # "article_pv_median",
            # "sentiment_score"
            ]
    df = extract_article_listwise_feat(df,feats,['user_id','impression_id'])
    
    return df
      
        

def part_merge_data(phase):
    # phase = "train"
    train_uids = pl.scan_parquet(f"../../dataset/{phase}_0619.parquet").select(['user_id']).collect().to_pandas()['user_id'].unique()
    train = pl.scan_parquet(f"../../dataset/{phase}_0619.parquet")
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
    use_cols = [
    'impression_id',
     'article_id',
     'user_id',

     'article_impr_hour_inview_mean',
     'impr_pub_hour_imprs_mean_impr',
     'total_inviews',
     'total_inviews_mean_impr_diff',
     'impr_article_impr_hour_article_cnt_rank_reverse',
     # 'total_ctr',
     'impr_pub_hour_imprs_mean_impr_diff',
     'impr_article_impr_pub_hour_imprs_rank',
     'impr_hour_article_cnt_mean_impr_diff',
     'impr_article_impr_pub_interval_rank',
     'total_avg_time_mean_impr_diff',
     'impr_pub_hour_imprs',
     'impr_article_impr_pub_hour_imprs_rank_reverse',
     'impr_pub_hour_imprs_diff',
     'impr_article_total_inviews_rank',
     'impr_article_impr_pub_interval_rank_reverse',
     'subcate_str',
     # 'article_id_right',
     'category_hist_click_num',
     'hist_category_length',
     'category_hist_click_num_ratio',
     'category_hist_click_read_time_sum_ratio',
     'category_hist_click_scroll_percentage_mean_ratio',
     'impr_category_cnt_new',
     'impr_inview_cnt',
     'impr_category_ratio',
     'user_impr_category_num_std',
     'user_impr_category_ratio_std'
    ]
    # his = pl.read_parquet(f"./dataset/history_add_sentiment/{phase}/history.parquet").drop(['impression_time_fixed','scroll_percentage_fixed','article_id_fixed','read_time_fixed'])
    path = "../../features/all_extend_feature_zzj01/"
    train_ex = pl.scan_parquet(path+f"{phase}.parquet").select(use_cols)
    
    for ids in tqdm(grouped_ids):
        
        
        
    
        train_part = train.filter(pl.col("user_id").is_in(ids))
        train_ex_part = train_ex.filter(pl.col('user_id').is_in(ids))
        train_part = train_part.join(train_ex_part,left_on=['impression_id','user_id','article_ids_inview'],right_on=['impression_id','user_id','article_id'],how="left")
        train_part = train_part.collect()
        
        
        train_part = gen_feature(train_part)
        
        train_part = reduce_memory_usage_pl(train_part)
        train_part = train_part.to_pandas()
        trains.append(train_part)
        
    
    trains = pd.concat(trains,axis=0,ignore_index=True)
    print(trains.shape)
    return trains

train = part_merge_data("train")
train.to_parquet("../../dataset/train_0620.parquet")

import gc
del train 
gc.collect()

valid = part_merge_data("validation")
valid.to_parquet("../../dataset/validation_0620.parquet")

del valid 
gc.collect()

test = part_merge_data("test")
test.to_parquet("../../dataset/test_0620.parquet")

del test 
gc.collect()