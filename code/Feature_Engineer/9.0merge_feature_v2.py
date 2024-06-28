import os

import numpy as np
import pandas as pd
import polars as pl
from joblib import Parallel, delayed
from tqdm import tqdm

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import os

import joblib
from sklearn.decomposition import PCA


def generated_embedding(df):
    name = df.columns[1].split("/")[-1]
    df = df.rename({df.columns[1] : name})
    if not os.path.exists(f"../../features/vectors/{name}-pca-{n_components}d.parquet"):
        pca = PCA(n_components=n_components, random_state=0)
        emb = pca.fit_transform(np.array(df[name].to_list()))
        item_dict = df[["article_id"]].with_columns(pl.Series(name, emb.astype(np.float32)))
        item_dict.write_parquet(f"../../features/vectors/{name}-pca-{n_components}d.parquet")

    return pl.read_parquet(f"../../features/vectors/{name}-pca-{n_components}d.parquet")


n_components = 10
contrastive_vector = pl.read_parquet("../../inputs/vectors/Ekstra_Bladet_contrastive_vector/contrastive_vector.parquet")
word2vec = pl.read_parquet("../../inputs/vectors/Ekstra_Bladet_word2vec/document_vector.parquet")

cl_dict = generated_embedding(contrastive_vector)

w2v_dict = generated_embedding(word2vec)

cl_mapping = dict(cl_dict.to_pandas().values)

w2v_mapping = dict(w2v_dict.to_pandas().values)
import gc
del cl_mapping,w2v_mapping
gc.collect()

cl_dict = cl_dict.to_pandas()
w2v_dict = w2v_dict.to_pandas()

cl_dict_new = cl_dict['contrastive_vector'].apply(pd.Series)
cl_dict_new.columns = ['contrastive_vector' + str(i+1) for i in cl_dict_new.columns]
cl_dict = pd.concat([cl_dict.drop('contrastive_vector', axis=1), cl_dict_new], axis=1)

w2v_dict_new = w2v_dict['document_vector'].apply(pd.Series)
w2v_dict_new.columns = ['document_vector' + str(i+1) for i in w2v_dict_new.columns]
w2v_dict = pd.concat([w2v_dict.drop('document_vector', axis=1), w2v_dict_new], axis=1)

del w2v_dict_new,cl_dict_new
gc.collect()

w2v_dict.to_parquet("../../features/w2v_pca_10_feature.parquet")
cl_dict.to_parquet("../../features/cl_pca_10_feature.parquet")

del_cols = ['cl_his_mean_item_sim_list',
 'cl_his_item_all_mean_sim',
 'cl_inview_item_all_mean_sim_list',
 'cl_his_mean_item_euclidean_list',
 'cl_inreviws_mean_std',
 'cl_inreviws_mean_skew',
 'cl_inreviws_mean_kurt',
 'cl_his_mean_std',
 'cl_his_mean_skew',
 'cl_his_mean_kurt',
 'cl_his_item_dot_list',
 'img_his_mean_item_sim_list',
 'img_his_item_all_mean_sim',
 'img_inview_item_all_mean_sim_list',
 'img_his_mean_item_euclidean_list',
 'img_inreviws_mean_std',
 'img_inreviws_mean_skew',
 'img_inreviws_mean_kurt',
 'img_his_mean_std',
 'img_his_mean_skew',
 'img_his_mean_kurt',
 'img_his_item_dot_list',
 'w2v_his_mean_item_sim_list',
 'w2v_his_item_all_mean_sim',
 'w2v_inview_item_all_mean_sim_list',
 'w2v_his_mean_item_euclidean_list',
 'w2v_inreviws_mean_std',
 'w2v_inreviws_mean_skew',
 'w2v_inreviws_mean_kurt',
 'w2v_his_mean_std',
 'w2v_his_mean_skew',
 'w2v_his_mean_kurt',
 'w2v_his_item_dot_list',
 'xlm_his_mean_item_sim_list',
 'xlm_his_item_all_mean_sim',
 'xlm_inview_item_all_mean_sim_list',
 'xlm_his_mean_item_euclidean_list',
 'xlm_inreviws_mean_std',
 'xlm_inreviws_mean_skew',
 'xlm_inreviws_mean_kurt',
 'xlm_his_mean_std',
 'xlm_his_mean_skew',
 'xlm_his_mean_kurt',
 'xlm_his_item_dot_list',
 'bert_his_mean_item_sim_list',
 'bert_his_item_all_mean_sim',
 'bert_inview_item_all_mean_sim_list',
 'bert_his_mean_item_euclidean_list',
 'bert_inreviws_mean_std',
 'bert_inreviws_mean_skew',
 'bert_inreviws_mean_kurt',
 'bert_his_mean_std',
 'bert_his_mean_skew',
 'bert_his_mean_kurt',
 'bert_his_item_dot_list']

#######

train = pl.scan_parquet("../../caches/lgb_train_new.parquet")
valid = pl.scan_parquet("../../caches/lgb_valid_new.parquet")

train = train.drop(del_cols)
valid = valid.drop(del_cols)

# ####add article曝光24小时数据
article_num = pl.scan_parquet("../../features/article_publish_48hour_num.parquet")
w2v_dict = pl.scan_parquet("../../features/w2v_pca_10_feature.parquet")
cl_dict = pl.scan_parquet("../../features/cl_pca_10_feature.parquet")


train = train.join(article_num,left_on = "article_ids_inview",right_on="article_id",how="left")
train = train.join(w2v_dict,left_on = "article_ids_inview",right_on="article_id",how="left")
train = train.join(cl_dict,left_on = "article_ids_inview",right_on="article_id",how="left")

valid = valid.join(article_num,left_on = "article_ids_inview",right_on="article_id",how="left")
valid = valid.join(w2v_dict,left_on = "article_ids_inview",right_on="article_id",how="left")
valid = valid.join(cl_dict,left_on = "article_ids_inview",right_on="article_id",how="left")
    
train = train.collect().to_pandas()

train.to_parquet("../../dataset/train.parquet")

import gc 
del train 
gc.collect()

valid = valid.collect().to_pandas()
valid.to_parquet("../../dataset/valid.parquet")
del valid 
gc.collect()

test = pl.scan_parquet("../../caches/lgb_test_new.parquet")
test = test.drop(del_cols)
test = test.join(article_num,left_on = "article_ids_inview",right_on="article_id",how="left")
test = test.join(w2v_dict,left_on = "article_ids_inview",right_on="article_id",how="left")
test = test.join(cl_dict,left_on = "article_ids_inview",right_on="article_id",how="left")

test = test.collect().to_pandas()
test.to_parquet("../../dataset/test.parquet")


