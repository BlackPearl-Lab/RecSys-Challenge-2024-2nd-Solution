import polars as pl 
import pandas as pd
import numpy as np 
from tqdm import tqdm 

cat_pair_860 = pl.read_parquet("./oof/test_oof_0620_pair_rank_loss_0.8601_all.parquet")
cat_pair_860 = cat_pair_860.with_columns(pl.col('pred').alias("cat_pair_860")).drop("pred")

cat_query_866 = pl.read_parquet("./oof/test_oof_0620_query_loss_0.866_all.parquet")
cat_query_866 = cat_query_866.with_columns(pl.col('pred').alias("cat_query_866")).drop("pred")

path = "./oof/"

xtf_test = pl.read_parquet(path+"xtf_v42_2fold_856/test_score.parquet")
xtf_test = xtf_test.with_columns(pl.col('final_score').alias("xtf_856")).drop(['fold_0_score','fold_1_score','final_score'])


test = xtf_test.clone()
test = test.with_columns(pl.col('impression_id').cast(pl.UInt32))
test = test.with_columns(pl.col('user_id').cast(pl.UInt32))
test = test.with_columns(pl.col('article_id').cast(pl.Int32))

test = test.join(cat_pair_860,left_on=['impression_id','user_id','article_id'],right_on = ['impression_id','user_id','article_ids_inview'],how="left")
test = test.join(cat_query_866,left_on=['impression_id','user_id','article_id'],right_on = ['impression_id','user_id','article_ids_inview'],how="left")

print(test.head())

test_pandas = test.to_pandas()


print(test_pandas.columns)
select_models = [ "xtf_856" ,'cat_pair_860','cat_query_866']

print(test_pandas[select_models].corr())




def normalize_pred_by_group(df, group_col=['impression_id','user_id'], target_col='pred'):
    """
    对 DataFrame 中的目标列按指定的分组列进行最大最小归一化。
    
    参数:
    - df: Polars DataFrame。
    - group_col: 用于分组的列名。
    - target_col: 需要进行归一化的目标列名。
    
    返回:
    - 更新后的 DataFrame，包含归一化后的目标列。
    """
    # 使用 groupby 和 agg 获取每个组的目标列的最小值和最大值
    min_max = df.groupby(group_col).agg([
        pl.col(target_col).min().alias("min_" + target_col),
        pl.col(target_col).max().alias("max_" + target_col)
    ])
    
    # 将最小值和最大值合并回原始 DataFrame
    df = df.join(min_max, on=group_col)
    
    # 计算归一化的目标列值
    normalized_col = ((pl.col(target_col) - pl.col("min_" + target_col)) / 
                      (pl.col("max_" + target_col) - pl.col("min_" + target_col))).alias(target_col)
    
    # 添加归一化列到 DataFrame
    df = df.with_columns(normalized_col)
    
    # 如果不需要保留 min_ 和 max_ 列，可以选择删除它们
    df = df.drop(["min_" + target_col, "max_" + target_col])
    
    return df

columns_to_normalize = select_models
for col in columns_to_normalize:
    test = normalize_pred_by_group(test, ['impression_id',"user_id"], col)




test = test.with_columns((test['xtf_856']*0.3+test['cat_pair_860']*0.2+test['cat_query_866']*0.5).alias("blend"))

####
infer = pl.read_parquet("../inputs/large/test/behaviors.parquet").with_columns(
    [
        pl.col("impression_id").cast(pl.UInt32),
        pl.col("user_id").cast(pl.UInt32)
        
    ]
).explode(['article_ids_inview']).select(['impression_id','user_id','article_ids_inview'])
infer = infer.with_columns(pl.col('article_ids_inview').cast(pl.Int32))
test = infer.join(test,on=['impression_id','user_id','article_ids_inview'],how="left")
#####


to_infer = test.group_by(["impression_id", "user_id"]).agg([pl.col("blend").rank(descending=True).cast(pl.UInt32)])
ori_test = pl.read_parquet("../inputs/large/test/behaviors.parquet").with_columns(
    [
        pl.col("impression_id").cast(pl.UInt32),
        pl.col("user_id").cast(pl.UInt32),
    ]
)

submit = (
    ori_test[["impression_id", "user_id"]]
    .join(to_infer, how="left", on=["impression_id", "user_id"])
    .drop("user_id")
    .to_pandas()
)

print(submit.head())

with open("../submit/predictions.txt", "w") as f:
    for value in tqdm(submit.values):
        v = ",".join([str(i) for i in value[1]])
        v = f"{value[0]} [{v}]"
        f.write(v)
        f.write("\n")

import shutil

shutil.make_archive(f"../submit/blend_8678", "zip", "../submit", "predictions.txt")
