from pathlib import Path
import polars as pl
import numpy as np
import os
from datetime import datetime
import gc
import warnings
import sklearn.preprocessing as sklearn_preprocess
warnings.filterwarnings("ignore")


os.chdir('../../')

base_dataset = 'nn_data'

root = 'inputs'
data_path = f"{root}/large"
train_path = f"{data_path}/train"
dev_path = f"{data_path}/validation"
test_path = f"{data_path}/test"
vector_path = f"{root}/vectors"
feature_path = f"features"

os.makedirs(f'{feature_path}/{base_dataset}/history', exist_ok=True)

news = pl.read_parquet(f'{feature_path}/{base_dataset}/article/article_feats.parquet')
news2cat = dict(zip(news["article_id"].cast(str), news["category"].cast(str)))
news2subcat = dict(zip(news["article_id"].cast(str), news["subcat1"].cast(str)))
news2sentiment = dict(zip(news["article_id"].cast(str), news["sentiment_label"]))
news2type = dict(zip(news["article_id"].cast(str), news["article_type"]))


# step1: validation history enhanced 
print('step1: validation history enhanced begin...')
history_file = os.path.join(train_path, "history.parquet")
train_history_df = pl.read_parquet(history_file)
train_history_df_explode = train_history_df.explode(['impression_time_fixed', 'scroll_percentage_fixed', 
                                                     'article_id_fixed', 'read_time_fixed'])
train_behaviors_df = pl.read_parquet(os.path.join(train_path, "behaviors.parquet"))
train_behaviors_clk_df = train_behaviors_df.explode(['article_ids_clicked']).rename({'article_ids_clicked': 'article_id_fixed'})

train_behaviors_clk_df = train_behaviors_clk_df.select(['user_id', 'impression_time', 'scroll_percentage', 'article_id_fixed', 'read_time']).rename(
    {'impression_time': 'impression_time_fixed',
    'scroll_percentage': 'scroll_percentage_fixed',
    'read_time': "read_time_fixed"})

train_history_df_explode_full = pl.concat([train_history_df_explode, train_behaviors_clk_df])

history_file = os.path.join(dev_path, "history.parquet")
valid_history_df = pl.read_parquet(history_file)
valid_history_df_explode = valid_history_df.explode(['impression_time_fixed', 'scroll_percentage_fixed', 
                                                     'article_id_fixed', 'read_time_fixed'])

valid_full_history_df_explode = pl.concat([train_history_df_explode_full, valid_history_df_explode])
valid_full_history_df_explode = valid_full_history_df_explode.unique(['user_id', 'impression_time_fixed', 'article_id_fixed'])


valid_full_history_df_explode = valid_full_history_df_explode.sort(['user_id', 'impression_time_fixed'])
dev_history_df_new = valid_full_history_df_explode.group_by('user_id').agg([pl.col('impression_time_fixed'), 
                                            pl.col('scroll_percentage_fixed'),
                                            pl.col('article_id_fixed'),
                                            pl.col('read_time_fixed'),
                                           ])

dev_history_df_new.write_parquet(f'{feature_path}/{base_dataset}/history/enhanced_valid_history_new.parquet', use_pyarrow=True)
print('step1: validation history enhanced end...')


# step2: test history enhanced 
print('step2: test history enhanced begin...')
history_file = os.path.join(test_path, "history.parquet")
test_history_df = pl.read_parquet(history_file)
test_history_df_explode = test_history_df.explode(['impression_time_fixed', 'scroll_percentage_fixed', 
                                                     'article_id_fixed', 'read_time_fixed'])
dev_history_df_new = pl.read_parquet(f'{feature_path}/{base_dataset}/history/enhanced_valid_history_new.parquet')
dev_history_df_new_explode = dev_history_df_new.explode(['impression_time_fixed', 'scroll_percentage_fixed', 
                                                     'article_id_fixed', 'read_time_fixed'])
valid_behaviors_df = pl.read_parquet(os.path.join(dev_path, "behaviors.parquet"))
valid_behaviors_clk_df = valid_behaviors_df.explode(['article_ids_clicked']).rename({'article_ids_clicked': 'article_id_fixed'})
valid_behaviors_clk_df = valid_behaviors_clk_df.select(['user_id', 'impression_time', 'scroll_percentage', 'article_id_fixed', 'read_time']).rename(
    {'impression_time': 'impression_time_fixed',
    'scroll_percentage': 'scroll_percentage_fixed',
    'read_time': "read_time_fixed"})

test_full_history_df_explode = pl.concat([dev_history_df_new_explode, valid_behaviors_clk_df, test_history_df_explode])
test_full_history_df_explode = test_full_history_df_explode.unique(['user_id', 'impression_time_fixed', 'article_id_fixed']) 
test_full_history_df_explode = test_full_history_df_explode.sort(['user_id', 'impression_time_fixed'])
test_history_df_new = test_full_history_df_explode.group_by('user_id').agg([pl.col('impression_time_fixed'), 
                                            pl.col('scroll_percentage_fixed'),
                                            pl.col('article_id_fixed'),
                                            pl.col('read_time_fixed'),
                                           ])
test_history_df_new.write_parquet(f'{feature_path}/{base_dataset}/history/enhanced_test_history_new.parquet', use_pyarrow=True)
print('step2: test history enhanced end...')


# step3: train bucketizer by frequency
print('step3: train bucketizer by frequency begin...')
history_file = os.path.join(train_path, "history.parquet")
history_df = pl.read_parquet(history_file)
history_df_explode = history_df.explode(['impression_time_fixed', 'scroll_percentage_fixed', 'article_id_fixed', 'read_time_fixed'])
# read_time
num_buckets = 50
qtf1 = sklearn_preprocess.KBinsDiscretizer(n_bins=num_buckets, encode='ordinal', strategy='quantile')
qtf1.fit(history_df_explode['read_time_fixed'].to_pandas().values.reshape(-1, 1))

# scroll_percentage
qtf2 = sklearn_preprocess.KBinsDiscretizer(n_bins=num_buckets, encode='ordinal', strategy='quantile')
qtf2.fit(history_df_explode['scroll_percentage_fixed'].to_pandas().fillna(0.0).values.reshape(-1, 1))
print('step3: train bucketizer by frequency end...')

# step4: get bucket no. for training
print('step4: get bucket no. for training begin...')
transform_data = qtf1.transform(history_df_explode['read_time_fixed'].to_pandas().values.reshape(-1,1)).squeeze()
history_df_explode = history_df_explode.with_columns(read_time_bin_fixed=transform_data)
transform_data = qtf2.transform(history_df_explode['scroll_percentage_fixed'].to_pandas().fillna(0.0).values.reshape(-1,1)).squeeze()
history_df_explode = history_df_explode.with_columns(scroll_percentage_bin_fixed=transform_data)
history_df_explode = history_df_explode.sort(['user_id', 'impression_time_fixed'])
history_df_new = history_df_explode.group_by('user_id').agg([pl.col('impression_time_fixed'), 
                                            pl.col('scroll_percentage_fixed'),
                                            pl.col('article_id_fixed'),
                                            pl.col('read_time_fixed'),
                                            pl.col('read_time_bin_fixed').cast(pl.Int8),
                                            pl.col('scroll_percentage_bin_fixed').cast(pl.Int8),
                                           ])
history_df_new.write_parquet(f'{feature_path}/{base_dataset}/history/train_history_enhanced_new_bin.parquet', use_pyarrow=True)
print('step4: get bucket no. for training end...')

# step5: get bucket no. for validation
print('step5: get bucket no. for validation begin...')
history_file = os.path.join(feature_path, base_dataset, 'history', "enhanced_valid_history_new.parquet")
dev_history_df = pl.read_parquet(history_file)
dev_history_df_explode = dev_history_df.explode(['impression_time_fixed', 'scroll_percentage_fixed', 'article_id_fixed', 'read_time_fixed'])

transform_data = qtf1.transform(dev_history_df_explode['read_time_fixed'].to_pandas().values.reshape(-1,1)).squeeze()
dev_history_df_explode = dev_history_df_explode.with_columns(read_time_bin_fixed=transform_data)
transform_data = qtf2.transform(dev_history_df_explode['scroll_percentage_fixed'].to_pandas().fillna(0.0).values.reshape(-1,1)).squeeze()
dev_history_df_explode = dev_history_df_explode.with_columns(scroll_percentage_bin_fixed=transform_data)
dev_history_df_explode = dev_history_df_explode.sort(['user_id', 'impression_time_fixed'])
dev_history_df_new = dev_history_df_explode.group_by('user_id').agg([pl.col('impression_time_fixed'), 
                                            pl.col('scroll_percentage_fixed'),
                                            pl.col('article_id_fixed'),
                                            pl.col('read_time_fixed'),
                                            pl.col('read_time_bin_fixed').cast(pl.Int8),
                                            pl.col('scroll_percentage_bin_fixed').cast(pl.Int8),
                                           ])
dev_history_df_new.write_parquet(f'{feature_path}/{base_dataset}/history/valid_history_enhanced_new_bin.parquet', use_pyarrow=True)
print('step5: get bucket no. for validation end...')

# step6: get bucket no. for test
print('step6: get bucket no. for test begin...')
history_file = os.path.join(feature_path, base_dataset, 'history', "enhanced_test_history_new.parquet")
test_history_df = pl.read_parquet(history_file)
test_history_df_explode = test_history_df.explode(['impression_time_fixed', 'scroll_percentage_fixed', 'article_id_fixed', 'read_time_fixed'])
transform_data = qtf1.transform(test_history_df_explode['read_time_fixed'].to_pandas().values.reshape(-1,1)).squeeze()
test_history_df_explode = test_history_df_explode.with_columns(read_time_bin_fixed=transform_data)
transform_data = qtf2.transform(test_history_df_explode['scroll_percentage_fixed'].to_pandas().fillna(0.0).values.reshape(-1,1)).squeeze()
test_history_df_explode = test_history_df_explode.with_columns(scroll_percentage_bin_fixed=transform_data)
test_history_df_explode = test_history_df_explode.sort(['user_id', 'impression_time_fixed'])
test_history_df_new = test_history_df_explode.group_by('user_id').agg([pl.col('impression_time_fixed'), 
                                            pl.col('scroll_percentage_fixed'),
                                            pl.col('article_id_fixed'),
                                            pl.col('read_time_fixed'),
                                            pl.col('read_time_bin_fixed').cast(pl.Int8),
                                            pl.col('scroll_percentage_bin_fixed').cast(pl.Int8),
                                           ])
test_history_df_new.write_parquet(f'{feature_path}/{base_dataset}/history/test_history_enhanced_new_bin.parquet', use_pyarrow=True)
print('step6: get bucket no. for test end...')


# step7: sequence processing
print('step7: sequence processing begin...')
def tokenize_seq(df, column, max_seq_length=5, sep="^"):
    df = df.with_columns(pl.col(column).apply(lambda x: x[-max_seq_length:]))
    df = df.with_columns(pl.col(column).apply(lambda x: f"{sep}".join(str(i) for i in x)))
    return df
def history_preprocess(history_df, id_col='hist_id', suffix='', max_seq_length=None):
    history_df = tokenize_seq(history_df, id_col, max_seq_length=max_seq_length)
    
    history_df = history_df.with_columns(
        pl.col(id_col).apply(lambda x: "^".join([news2cat.get(i, "") for i in x.split("^")])).alias("hist_cat"+suffix),
        pl.col(id_col).apply(lambda x: "^".join([news2subcat.get(i, "") for i in x.split("^")])).alias("hist_subcat1"+suffix),
        pl.col(id_col).apply(lambda x: "^".join([news2sentiment.get(i, "") for i in x.split("^")])).alias("hist_sentiment"+suffix),
        pl.col(id_col).apply(lambda x: "^".join([news2type.get(i, "") for i in x.split("^")])).alias("hist_type"+suffix)
    )
    history_df = history_df.collect()
    return history_df

def preprocess(mode, max_seq_length):
    history_file = os.path.join(feature_path, base_dataset, 'history', f"{mode}_history_enhanced_new_bin.parquet")
    history_df = pl.scan_parquet(history_file)
    
    history_df = history_df.rename({"article_id_fixed": "hist_id", 
                                    "read_time_fixed": "hist_read_time",
                                    "impression_time_fixed": "hist_time",
                                    "scroll_percentage_fixed": "hist_scroll_percent",
                                    "read_time_bin_fixed": "hist_read_time_bin",
                                    "scroll_percentage_bin_fixed": "hist_scroll_percent_bin",
                                   })
    
    history_df = history_df.with_columns(hist_clk_num=pl.col('hist_id').list.len(), 
                                         hist_total_readtime=pl.col('hist_read_time').list.sum())
    
    history_df = tokenize_seq(history_df, 'hist_read_time_bin', max_seq_length=max_seq_length)
    history_df = tokenize_seq(history_df, 'hist_scroll_percent_bin', max_seq_length=max_seq_length)
    
    history_df = history_preprocess(history_df, max_seq_length=max_seq_length)

    
    history_df = history_df.select(['user_id', 'hist_id', 'hist_clk_num', 'hist_total_readtime', 
                                    'hist_cat', 'hist_subcat1', 'hist_sentiment', 'hist_type',
                                    'hist_read_time_bin', 'hist_scroll_percent_bin'])
    return history_df


# processing sequence for nn
MAX_SEQ_LEN = 200
df_train_history = preprocess('train', MAX_SEQ_LEN)
df_train_history.write_parquet(f'{feature_path}/{base_dataset}/history/train_history.parquet', use_pyarrow=True)
df_valid_history = preprocess('valid', MAX_SEQ_LEN)
df_valid_history.write_parquet(f'{feature_path}/{base_dataset}/history/valid_history.parquet', use_pyarrow=True)
df_test_history = preprocess('test', MAX_SEQ_LEN)
df_test_history.write_parquet(f'{feature_path}/{base_dataset}/history/test_history.parquet', use_pyarrow=True)

print('step7: sequence processing end...')
