import polars as pl 
article = pl.read_parquet("../../inputs/large/articles.parquet")
article = article.select(['article_id','published_time'])

##### 
tr_be = pl.read_parquet("../../inputs/large/train/behaviors.parquet")
val_be = pl.read_parquet("../../inputs/large/validation/behaviors.parquet")
test_be = pl.read_parquet("../../inputs/large/test/behaviors.parquet")

tr_be = tr_be.select(['impression_id','impression_time','article_ids_inview']).rename({"article_ids_inview":"article_id"}).explode("article_id")
val_be = val_be.select(['impression_id','impression_time','article_ids_inview']).rename({"article_ids_inview":"article_id"}).explode("article_id")
test_be = test_be.select(['impression_id','impression_time','article_ids_inview']).rename({"article_ids_inview":"article_id"}).explode("article_id")

tr_be = tr_be.join(article,on='article_id',how="left")
val_be = val_be.join(article,on='article_id',how="left")
test_be = test_be.join(article,on='article_id',how="left")

data = pl.concat([tr_be,val_be,test_be])

import gc
del tr_be,val_be,test_be
gc.collect()

data  = data.with_columns(distance_publish_hours=(pl.col("impression_time") - pl.col("published_time"))
                .dt.total_hours()
                .cast(pl.Int32),
                distance_publish_seconds=(pl.col("impression_time") - pl.col("published_time"))
                .dt.total_seconds()
                .cast(pl.Int32),
            )

data = data.filter((pl.col("distance_publish_seconds") >= 0))
data = data.filter((pl.col("distance_publish_hours") >= 0) & (pl.col("distance_publish_hours") < 48))
data = data.to_pandas()
data_group = data.groupby(["article_id",'distance_publish_hours'],as_index=False)['distance_publish_seconds'].count()
data_group.columns = ['article_id','hour',"count"]
df_pivot = data_group.pivot(index='article_id', columns='hour', values='count')
df_pivot.columns = ["num_publish_hour_"+str(i) for i in range(48)]
df_pivot = df_pivot.reset_index()

data_group['article_num_count_diff'] = data_group.groupby("article_id",as_index=False)['count'].diff()

#####article针对曝光数的统计特征
for stats in ['mean',"max","sum","std","skew",'min','median']:
    data_group['article_pv_'+stats] = data_group.groupby("article_id",as_index=False)['count'].transform(stats)
    data_group['article_pv_diff_'+stats] = data_group.groupby("article_id",as_index=False)['article_num_count_diff'].transform(stats)
    
data_group['article_hour_nunique'] = data_group.groupby("article_id",as_index=False)['hour'].transform('nunique')

data_group = data_group.drop(["hour",'count'],axis=1).drop_duplicates()
data_group = data_group.drop(["article_num_count_diff"],axis=1).drop_duplicates()
df_pivot.to_parquet("../../features/article_publish_48hour_num.parquet")


def cumsum_hour(data,hour):
    init = data['num_publish_hour_0'].copy()
    for i in range(1,hour):
        init+=data[f'num_publish_hour_{i}'].copy().fillna(0.0)
    return init

df_pivot['cumsum_publish_hour_3'] = cumsum_hour(df_pivot,3)
df_pivot['cumsum_publish_hour_5'] = cumsum_hour(df_pivot,5)
df_pivot['cumsum_publish_hour_6'] = cumsum_hour(df_pivot,6)
df_pivot['cumsum_publish_hour_8'] = cumsum_hour(df_pivot,8)
df_pivot['cumsum_publish_hour_9'] = cumsum_hour(df_pivot,9)
df_pivot['cumsum_publish_hour_10'] = cumsum_hour(df_pivot,10)
df_pivot['cumsum_publish_hour_12'] = cumsum_hour(df_pivot,12)
df_pivot['cumsum_publish_hour_24'] = cumsum_hour(df_pivot,24)

df_pivot = df_pivot[['article_id','cumsum_publish_hour_3','cumsum_publish_hour_5',
                     'cumsum_publish_hour_6','cumsum_publish_hour_8','cumsum_publish_hour_9','cumsum_publish_hour_10','cumsum_publish_hour_12','cumsum_publish_hour_24']]

data_group = data_group.merge(df_pivot,on='article_id',how="left")

data_group.to_parquet("../../features/article_num_stats.parquet")