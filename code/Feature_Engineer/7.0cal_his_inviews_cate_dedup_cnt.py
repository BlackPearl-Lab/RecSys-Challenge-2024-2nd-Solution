import pandas as pd
import numpy as np
import os  
from tqdm import tqdm 
from scipy.stats import skew
import polars as pl 

article = pd.read_parquet("../../inputs/large/articles.parquet")

def gen_subcategory(cate_list):
    try:
        return cate_list[0]
    except:
        return "no category"
    
article['subcategory1'] = article['subcategory'].map(gen_subcategory)
article['subcategory'] = article['subcategory'].map(lambda x:"-".join([str(i) for i in x]))

for col in ['subcategory1','subcategory','category']:
    col_dict =  dict(zip(list(article[col].unique()), range(len(article[col].unique()))))
    article[col] = article[col].map(col_dict)

news2cat = dict(zip(list(article["article_id"]), list(article["category"])))
news2subcat = dict(zip(list(article["article_id"]), list(article["subcategory1"])))

article = pl.from_pandas(article)
article = article.select(['article_id','subcategory','published_time'])

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

import gc
def gen_feature(phase):
    train_his = pl.read_parquet(f"../../inputs/large/{phase}/history.parquet")
    train_his = train_his.with_columns(pl.col("article_id_fixed").alias("his_article_ids"))
    
    def convert_to_cate_ids(article_ids):
        return [news2cat[i] for i in article_ids]
    
    train_his = train_his.with_columns(train_his['his_article_ids'].apply(convert_to_cate_ids, return_dtype=pl.List(pl.Int32)).alias('his_cate_ids'))
    
    def convert_to_subcate_ids(article_ids):
        return [news2subcat[i] for i in article_ids]
    
    train_his = train_his.with_columns(train_his['his_article_ids'].apply(convert_to_subcate_ids, return_dtype=pl.List(pl.Int32)).alias('his_subcate_ids'))

    
    train_his = train_his.select(['user_id','his_article_ids', 'his_cate_ids','his_subcate_ids'])
    
    train = pl.read_parquet(f"../../inputs/large/{phase}/behaviors.parquet")
    train = train.select(['impression_id','impression_time','article_ids_inview','user_id'])
    
    def convert_to_cate_ids(article_ids):
        return [news2cat[i] for i in article_ids]

    # 添加新的列或者覆盖已有的列
    train = train.with_columns(train['article_ids_inview'].apply(convert_to_cate_ids, return_dtype=pl.List(pl.Int32)).alias('cate_ids_inview'))

    def convert_to_subcate_ids(article_ids):
        return [news2subcat[i] for i in article_ids]

    # 添加新的列或者覆盖已有的列
    train = train.with_columns(train['article_ids_inview'].apply(convert_to_subcate_ids, return_dtype=pl.List(pl.Int32)).alias('subcate_ids_inview'))
    train = train.select(['impression_id','impression_time','user_id','article_ids_inview','cate_ids_inview','subcate_ids_inview'])
    train = train.explode(['article_ids_inview','cate_ids_inview','subcate_ids_inview'])
    
    train_his = train_his.select(['user_id','his_cate_ids','his_subcate_ids'])
    
    train = reduce_memory_usage_pl(train)
    # train_his = reduce_memory_usage_pl(train_his)
    
    train = train.join(train_his,on="user_id",how="left")
    
    del train_his
    gc.collect()
   
    train = train.join(article,left_on="article_ids_inview",right_on = "article_id",how="left")
    train = train.with_columns(distance_publish_seconds=(pl.col("impression_time") - pl.col("published_time"))
                .dt.total_seconds()
                .cast(pl.Int32),
                distance_publish_minutes=(pl.col("impression_time") - pl.col("published_time"))
                .dt.total_minutes()
                .cast(pl.Int32))
    
    train = train.drop(['impression_time','user_id','published_time'])
    train = train.to_pandas()
    
    train['his_cate_cnt']  = [sum(1 for id in ids if id == cat) for ids, cat in tqdm(zip(train['his_cate_ids'], train['cate_ids_inview']))]
    train = train.drop(['his_cate_ids','cate_ids_inview'],axis=1)
    train['his_subcate_cnt']  = [sum(1 for id in ids if id == cat) for ids, cat in tqdm(zip(train['his_subcate_ids'], train['subcate_ids_inview']))]
    train = train.drop(['his_subcate_ids','subcate_ids_inview'],axis=1)
    
    # train = reduce_memory_usage_pl(train)
    return train

train = gen_feature("train")
train.to_parquet("../../features/train_his_cate_feature.parquet")
del train
gc.collect()

val = gen_feature("validation")
val.to_parquet("../../features/validation_his_cate_feature.parquet")
del val
gc.collect()

test = gen_feature("test")
test.to_parquet("../../features/test_his_cate_feature.parquet")
del test
gc.collect()

