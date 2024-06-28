import pandas as pd
import numpy as np
import os  
from tqdm import tqdm 
from scipy.stats import skew

train_his = pd.read_parquet("../../inputs/large/train/history.parquet")
val_his = pd.read_parquet("../../inputs/large/validation/history.parquet")
test_his = pd.read_parquet("../../inputs/large/test/history.parquet")

def cal_skew(df):
    nonempty_data = df[~np.isnan(df)]
    return skew(nonempty_data)
def gen_his_feature(data):
    #####scroll_percentage_fixed
    for col in ['scroll_percentage_fixed','read_time_fixed']:
        data["his_"+col+'_mean'] = data[col].map(np.nanmean)
        data["his_"+col+'_std'] = data[col].map(np.nanstd)
        data["his_"+col+'_max'] = data[col].map(np.nanmax)
        data["his_"+col+'_min'] = data[col].map(np.nanmin)
        data["his_"+col+'_sum'] = data[col].map(np.nansum)
        
        
        data["his_"+col+'_skew'] = data[col].map(cal_skew)
    data['his_count'] = data['article_id_fixed'].map(len)
    data['his_count_unique'] = data['article_id_fixed'].map(lambda x:len(set(x)))
    return data

train_his = gen_his_feature(train_his)
val_his = gen_his_feature(val_his)
test_his = gen_his_feature(test_his)

####load article 
article = pd.read_parquet("../../inputs/large/articles.parquet")

def gen_subcategory(cate_list):
    try:
        return cate_list[0]
    except:
        return "no category"
    
article['subcategory1'] = article['subcategory'].map(gen_subcategory)



for col in ['premium','article_type','sentiment_label','subcategory1','category']:
    col_dict =  dict(zip(list(article[col].unique()), range(len(article[col].unique()))))
    article[col] = article[col].map(col_dict)
    
news2cat = dict(zip(list(article["article_id"]), list(article["category"])))
news2subcat = dict(zip(list(article["article_id"]), list(article["subcategory1"])))

train_his['his_article_ids'] = train_his['article_id_fixed']
val_his['his_article_ids'] = val_his['article_id_fixed']
test_his['his_article_ids'] = test_his['article_id_fixed']


train_his['his_cate_ids'] = train_his['his_article_ids'].map(lambda x:[news2cat[i] for i in x])
val_his['his_cate_ids'] = val_his['his_article_ids'].map(lambda x:[news2cat[i] for i in x])
test_his['his_cate_ids'] = test_his['his_article_ids'].map(lambda x:[news2cat[i] for i in x])

train_his['his_subcate_ids'] = train_his['his_article_ids'].map(lambda x:[news2subcat[i] for i in x])
val_his['his_subcate_ids'] = val_his['his_article_ids'].map(lambda x:[news2subcat[i] for i in x])
test_his['his_subcate_ids'] = test_his['his_article_ids'].map(lambda x:[news2subcat[i] for i in x])

import pandas as pd
import numpy as np

def calculate_time_diffs(timestamp_list):
    # 将字符串转换为时间戳
    timestamps = pd.to_datetime(timestamp_list)
    
    # 计算两两之间的时间差
    diffs = np.diff(timestamps)
    
    # 计算出小时，天数，分钟数，秒数
    hours = diffs / np.timedelta64(1, 'h')
    days = diffs / np.timedelta64(1, 'D')
    minutes = diffs / np.timedelta64(1, 'm')
    seconds = diffs / np.timedelta64(1, 's')
    
    return hours, days, minutes, seconds

import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
def flatten_stats(stats):
    # 创建一个新的字典来存储拉平的统计结果
    flat_stats = {}

    for diff_name, diff_stats in stats.items():
        for stat_name, stat_value in diff_stats.items():
            # 创建一个新的键，将时间差值的名称和统计名称组合在一起
            new_key = f'{diff_name}_{stat_name}'

            # 存储统计结果
            flat_stats[new_key] = stat_value

    return flat_stats


def calculate_time_diffs_stats(timestamp_list):
    # 计算时间差值
    hours, days, minutes, seconds = calculate_time_diffs(timestamp_list)
    
    # 创建一个字典来存储统计结果
    stats = {}
    
    for diff, name in zip([hours, days, minutes, seconds], ['his_diff_hours', 'his_diff_days', 'his_diff_minutes', 'his_diff_seconds']):
        # 计算并存储统计结果
        stats[name] = {
            'mean': np.mean(diff),
            'std': np.std(diff),
            'skew': skew(diff),
            'max': np.max(diff),
            'min': np.min(diff),
            'kurt': kurtosis(diff)
        }
    stats = flatten_stats(stats)
    return stats

def gen_his_time_diff_feature(data):
    diff_df = []
    for time_list in list(data['impression_time_fixed']):
        diff_feature = calculate_time_diffs_stats(time_list)
        diff_df.append(diff_feature)
    diff_df = pd.DataFrame(diff_df)
    return diff_df

train_diff = gen_his_time_diff_feature(train_his)
val_diff = gen_his_time_diff_feature(val_his)
test_diff = gen_his_time_diff_feature(test_his)

train_his = train_his.join(train_diff)
val_his = val_his.join(val_diff)
test_his = test_his.join(test_diff)


train_his.to_parquet("./features/train_his.parquet")
val_his.to_parquet("./features/val_his.parquet")
test_his.to_parquet("./features/test_his.parquet")