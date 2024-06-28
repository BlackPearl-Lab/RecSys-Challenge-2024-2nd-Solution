import os
import numpy as np
import pandas as pd
import polars as pl
from joblib import Parallel, delayed
from tqdm import tqdm
import gc

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


drop_features = ['user_article_freq', 'images_all_mean_cosine', 'images_inview_std',
       'images_inview_skew', 'images_inview_kurt', 'last_exp_diff',
       'gender', 'title_inreviws_mean_std', 'images_history_skew',
       'cl_all_mean_cosine', 'history_time_diff_skew', 'cl_inview_skew',
       'impr_article_user_impression_freq_rank_reverse',
       'impr_article_user_impression_freq_rank', 'impr_hour_category_cnt',
       'bert_all_mean_cosine', 'pulish_7day', 'bert_inview_std',
       'bert_inview_kurt', 'bert_history_std', 'w2v_all_mean_cosine',
       'total_pageviews_mean', 'w2v_inview_skew', 'w2v_inview_kurt',
       'distance_last_click_time', 'session_id_impression_time_last',
       'roberta_all_mean_cosine', 'read_time_next_fix',
       'user_impression_freq_next_fix', 'next_pred_readtime',
       'roberta_inview_skew', 'roberta_inview_kurt', 'user_session_cnt',
       'roberta_history_std', 'w2v_history_skew',
       'title_his_item_all_mean_sim', 'body_his_item_all_mean_sim',
       'user_total_inviews_skew', 'user_bpr_skew', 'user_bpr_std',
       'user_bpr_mean', 'user_ctr_skew', 'title_his_mean_skew',
       'subcate_emb_sim_his_mean_kurt', 'subcate_emb_sim_his_mean_std',
       'subcate_emb_sim_inreviws_mean_kurt',
       'subcate_emb_sim_his_item_all_mean_sim', 'title_his_mean_std',
       'user_user_impression_freq_skew', 'body_inreviws_mean_std',
       'cate_emb_sim_his_mean_std', 'user_w2v_each_cosine_std',
       'cate_emb_sim_inview_item_all_mean_sim_list',
       'user_inreviws_mean_skew', 'user_his_mean_skew',
       'topic_his_item_all_mean_sim', 'history_scroll_percentage_mean',
       'body_his_mean_kurt', 'topic_inreviws_mean_kurt',
       'topic_his_mean_std', 'title_his_mean_kurt',
       'user_user_impression_freq_std', 'user_his_mean_kurt',
       'cate_emb_sim_his_item_all_mean_sim', 'roberta_history_skew',
       'read_time_fixed_skew', 'history_article_id_entropy',
       'cate_emb_sim_inreviws_mean_skew', 'imporession_weekday_cos',
       'user_his_item_all_mean_sim', 'bert_inview_skew',
       'body_smog_index', 'imporession_weekday_sin',
       'article_user_avg_clk', 'num_publish_hour_15',
       'category_imprs_user', 'distance_publish_seconds_mean',
       'impr_article_bpr_rank_reverse', 'num_publish_hour_4',
       'session_distance_publish_seconds_mean_diff', 'document_vector5',
       'history_read_time_std', 'num_publish_hour_14',
       'num_publish_hour_18', 'session_impression_rank',
       'user_total_inviews_std', 'article_pv_sum', 'date_impression_rank',
       'impr_article_distance_published_hour_cos_rank_reverse',
       'cl_bert_cosine_similarity', 'user_id_count',
       'subcate_emb_sim_inreviws_mean_skew', 'contrastive_vector2',
       'body_gunning_fog', 'body_inreviws_mean_kurt',
       'cumsum_publish_hour_5', 'title_difficult_words',
       'last_read_time_sort', 'num_publish_hour_37',
       'body_coleman_liau_index', 'document_vector7',
       'published_weekday_sin', 'article_pv_diff_min',
       'roberta_each_cosine', 'body_automated_readability_index',
       'user_id_nunique', 'topic_his_mean_item_euclidean_list',
       'body_inreviws_mean_skew', 'published_hour_cos',
       'contrastive_vector7', 'impr_article_total_inviews_rank_reverse',
       'cl_roberta_cosine_similarity', 'user_his_mean_std',
       'num_publish_hour_41', 'num_publish_hour_12', 'document_vector6',
       'article_pv_diff_max', 'article_pv_diff_median', 'cl_inview_std',
       'subcategory', 'roberta_inview_std', 'body_linsear_write_formula',
       'document_vector10', 'sentiment_label', 'contrastive_vector4',
       'num_publish_hour_5', 'contrastive_vector6', 'impression_weekday',
       'body_flesch_reading_ease', 'num_publish_hour_27',
       'impr_article_total_inviews_rank', 'num_publish_hour_6',
       'user_his_mean_item_sim_list', 'roberta_each_euclidean',
       'session_article_cnt', 'title_his_mean_item_sim_list',
       'article_pv_diff_mean', 'article_pv_diff_std',
       'session_total_inviews_mean_diff', 'session_id_read_time_next',
       'user_his_mean_item_euclidean_list', 'published_hour_sin',
       'contrastive_vector5', 'user_his_item_dot_list',
       'session_distance_published_hour_cos_mean_diff',
       'subcate_emb_sim_his_item_dot_list', 'num_publish_hour_8',
       'topic_his_mean_item_sim_list', 'impr_article_ctr_rank',
       'body_his_mean_item_sim_list',
       'title_his_mean_item_euclidean_list', 'contrastive_vector8',
       'topic_his_item_dot_list', 'cate_emb_sim_his_item_dot_list',
       'title_his_item_dot_list',
       'impr_article_impr_hour_article_cnt_rank_reverse',
       'user_user_impression_freq_mean',
       'impr_article_distance_publish_seconds_rank',
       'impr_article_distance_publish_seconds_rank_reverse',
       'body_his_mean_item_euclidean_list', 'num_publish_hour_7',
       'impression_distance_published_hour_cos_mean_impr_diff',
       'body_his_item_dot_list', 'document_vector2', 'document_vector1',
       'impression_total_inviews_mean_impr_diff', 'total_inviews',
       'article_pv_skew', 'distance_publish_hours'
]

train = pl.scan_parquet("../dataset/train_0620.parquet")
valid = pl.scan_parquet("../dataset/validation_0620.parquet")


#####drop features
add_cols = [
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
drop_features = list(set(drop_features)- set(add_cols))
train = train.drop(drop_features)
valid = valid.drop(drop_features)


# article_stats = pl.scan_parquet("../features/article_num_stats.parquet")
# train = train.join(article_stats,left_on ="article_ids_inview" ,right_on="article_id",how="left")
# valid = valid.join(article_stats,left_on ="article_ids_inview" ,right_on="article_id",how="left")
# valid = pl.concat([train,valid])

valid = valid.collect()
valid = valid.to_pandas()
sample_ids = valid['user_id'].unique()[:10000]
valid = valid[valid['user_id'].isin(sample_ids)]

train = train.collect()
train = train.to_pandas()

train = train.sort_values(by=['user_id','impression_id'])
valid = valid.sort_values(by=['user_id','impression_id'])

print(train.shape, valid.shape)

not_use_columns = [
    "impression_id",
    "impression_time",
    "session_id",
    "phase",
    "click",
    "next_read_time",
    "next_scroll_percentage",
    "user_id",
    "bert_history_skew",
    "history_read_time_skew",
    "postcode",
    "age",
    "article_id",
    # "article_ids_inview",
    "impression_position",
    "trigger_id",
    'user_impression_freq_last',
    'user_impression_freq_next',
    'read_time_last',
    'read_time_next',
    '__index_level_0__',
    "article_freq",
    "article_id_fixed"
    
    
]

category_name = [
    "article_ids_inview",
    "device_type",
    "is_sso_user",
    "gender",
    "is_subscriber",
]
# dense_name = [i for i in valid.columns if i not in not_use_columns + category_name]
# feature_name = category_name + dense_name
feature_name = [i for i in valid.columns if i not in not_use_columns]

print(len(feature_name))

import catboost as cat
from sklearn.model_selection import GroupKFold
from catboost import CatBoostRanker, CatBoostClassifier, Pool
print('CatBoost version',cat.__version__)

train_set = Pool(
        data = train[feature_name],
        label =train['click'],
        group_id = train['impression_id']
    )
    
valid_set = Pool(
        data = valid[feature_name],
       label =valid['click'],
        group_id = valid['impression_id']
    )
    
del train 
gc.collect()

model = CatBoostClassifier(iterations=5000,
                              random_state=2024,
                              max_depth=8,
                           # num_leaves=64,   #max_leaves=32,
                              learning_rate=0.05,
                              # subsample=0.8,
                              task_type='GPU',
                        # bootstrap_type='Bernoulli',
                              #custom_metric = 'MAP:top=50',
                              #eval_metric = 'MAP',
                             # sampling_frequency=5,
                              # colsample_bylevel=0.4,
                            # custom_metric=['AUC', 'NDCG:top=10'],
                              #thread_count=16,
                              #scale_pos_weight=10
                              loss_function="Logloss"#'PairLogitPairwise'
                           )
    
    
model = CatBoostRanker(iterations=5_000,
                              random_state=2024,
                              max_depth=8,
                              #max_leaves=32,
                              learning_rate=0.05,
                              # subsample=0.75,
                              task_type='GPU',
                       border_count = 254,
                        # bootstrap_type='Bernoulli',
                              #custom_metric = 'MAP:top=50',
                              #eval_metric = 'MAP',
                             # sampling_frequency=5,
                              # colsample_bylevel=0.75,
                            # custom_metric=['AUC', 'NDCG:top=10'],
                              #thread_count=16,
                              #scale_pos_weight=10
                              loss_function="PairLogitPairwise"#"QueryCrossEntropy"#'PairLogitPairwise'#"QueryCrossEntropy",#'PairLogitPairwise'
                           )

# model = CatBoostRanker(iterations=5_000,
#                               random_state=2024,
#                                  num_leaves=64,   #max_leaves=32,
#                               # max_depth=8,
#                               #max_leaves=32,
#                               learning_rate=0.1,
#                               subsample=0.8,
#                               task_type='GPU',
#                         bootstrap_type='Bernoulli',
#                               #custom_metric = 'MAP:top=50',
#                               #eval_metric = 'MAP',
#                              # sampling_frequency=5,
#                               colsample_bylevel=0.4,
#                             # custom_metric=['AUC', 'NDCG:top=10'],
#                               #thread_count=16,
#                               #scale_pos_weight=10
#                               loss_function='PairLogitPairwise'
#                            )

model.fit(train_set,
             verbose=100,
             early_stopping_rounds=100,
             eval_set=valid_set,
             )
model.save_model(f'./models/catboost_pair_rank_loss_0620.cat')
# model.save_model(f'./models/catboost_query_cross_loss_0605.cat')

feature_importances = model.get_feature_importance(data=valid_set)#model.feature_importances_#model.get_feature_importance(data=valid_set)#model.feature_importances_#model.get_feature_importance(data=valid_set,type=EFstrType.TotalGain)



imp = pd.DataFrame( feature_importances, index=model.feature_names_, columns=["score"]).sort_values(
    by=["score"], ascending=False
)#.style.background_gradient("coolwarm")
print(imp.head(300))


del train_set,valid_set,valid
gc.collect()

print(model.best_score_)


valid = pl.scan_parquet("../dataset/validation_0620.parquet").drop(drop_features)
valid = valid.collect()
valid = valid.to_pandas()
print(valid.shape)

preds = model.predict(valid[feature_name])

from sklearn.metrics import roc_auc_score
valid_frame = valid[["impression_id", "click"]]
valid_frame["pred"] = preds
global_auc = roc_auc_score(valid_frame["click"], valid_frame["pred"])
print(global_auc)
import joblib

from tqdm import tqdm 
def calc_auc(x):
    return roc_auc_score(x["click"], x["pred"])

aucs = joblib.Parallel(n_jobs=48, backend="multiprocessing")(
    joblib.delayed(calc_auc)(group) for _, group in tqdm(valid_frame.groupby("impression_id"))
)
gauc = np.mean(aucs)
print(global_auc,gauc)


val_sub = valid[['impression_id','article_ids_inview','user_id']]
val_sub['pred'] = preds
val_sub.to_parquet(f"./oof/val_0620_pair_rankloss_oof_{round(gauc, 4)}_filter_feature.parquet")
# val_sub.to_parquet(f"./oof/val_0618_query_cross_entropy_oof_{round(gauc, 4)}.parquet")
# imp.to_csv(f"./oof/imp_catboost_{round(gauc, 4)}.csv")
# imp.to_csv(f"./oof/imp_catboost_0601_query_cross_entropy_{round(gauc, 4)}.csv")
imp.to_csv(f"./oof/imp_catboost_0620_pair_rank_loss_{round(gauc, 4)}_filter_feature.csv")

del valid
gc.collect()

test = pl.scan_parquet("../dataset/test_0620.parquet").drop(drop_features)
test = test.collect()
test = test.to_pandas()
print(test.shape)


feature_name = model.feature_names_
online_preds = model.predict(test[feature_name])#preds = model.predict(valid[feature_name])#model.predict_proba(valid[feature_name])

infer = test[["impression_id", "article_ids_inview", "user_id"]]
infer['pred'] = online_preds
# infer.to_parquet(f"./oof/test_oof_0605_query_cross_entropy_{round(gauc, 4)}_train.parquet")
infer.to_parquet(f"./oof/test_oof_0620_pair_rank_loss_{round(gauc, 4)}_filter_feature_train.parquet")

import gc
del test
gc.collect()

print(infer.head(20))

infer = pl.from_pandas(infer)

test = pl.read_parquet("../inputs/large/test/behaviors.parquet").with_columns(
    [
        pl.col("impression_id").cast(pl.UInt32),
        pl.col("user_id").cast(pl.UInt32)
        
    ]
).explode(['article_ids_inview']).select(['impression_id','user_id','article_ids_inview'])
test = test.with_columns(pl.col('article_ids_inview').cast(pl.Int32))
infer = test.join(infer,on=['impression_id','user_id','article_ids_inview'],how="left")

print(infer.head(20))



to_infer = infer.group_by(["impression_id", "user_id"]).agg([pl.col("pred").rank(descending=True).cast(pl.UInt32)])
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

with open("../submit/predictions.txt", "w") as f:
    for value in tqdm(submit.values):
        v = ",".join([str(i) for i in value[1]])
        v = f"{value[0]} [{v}]"
        f.write(v)
        f.write("\n")

import shutil

shutil.make_archive(f"../submit/0620_pair_rank_loss_gauc={round(gauc, 4)}_filter_feature_train", "zip", "../submit", "predictions.txt")







