import pandas as pd
from tqdm import tqdm 
import os 
import numpy as np
import polars as pl


#### load emb
cl_emb = pd.read_parquet("../../inputs/vectors/Ekstra_Bladet_contrastive_vector/contrastive_vector.parquet")
img_emb = pd.read_parquet("../../inputs/vectors/Ekstra_Bladet_image_embeddings/image_embeddings.parquet")
w2v_emb = pd.read_parquet("../../inputs/vectors/Ekstra_Bladet_word2vec/document_vector.parquet")
xlm_emb = pd.read_parquet("../../inputs/vectors/FacebookAI_xlm_roberta_base/xlm_roberta_base.parquet")
bert_emb = pd.read_parquet("../../inputs/vectors/google_bert_base_multilingual_cased/bert_base_multilingual_cased.parquet")


cl_emb_dict = dict(zip(list(cl_emb['article_id']),list(cl_emb['contrastive_vector'])))
img_emb_dict = dict(zip(list(img_emb['article_id']),list(img_emb['image_embedding'])))
w2v_emb_dict = dict(zip(list(w2v_emb['article_id']),list(w2v_emb['document_vector'])))
xlm_emb_dict = dict(zip(list(xlm_emb['article_id']),list(xlm_emb['FacebookAI/xlm-roberta-base'])))
bert_emb_dict = dict(zip(list(bert_emb['article_id']),list(bert_emb['google-bert/bert-base-multilingual-cased'])))

from gensim.models import Word2Vec
w2v = Word2Vec.load(f'./features/article_id.model')
article_emb_dict = w2v.wv

w2v = Word2Vec.load(f'./features/category.model')
cate_emb_dict = w2v.wv

w2v = Word2Vec.load(f'./features/subcategory1.model')
subcate_emb_dict = w2v.wv

title_emb_dict = np.load("./features/title_emb_dict.npy",allow_pickle=True).item()
body_emb_dict = np.load("./features/body_emb_dict.npy",allow_pickle=True).item()
topic_emb_dict = np.load("./features/topics_emb_dict.npy",allow_pickle=True).item()

cl_emb_dict[-1] = np.zeros(768)
img_emb_dict[-1] = np.zeros(1024)
w2v_emb_dict[-1] = np.zeros(300)
xlm_emb_dict[-1] = np.zeros(768)
bert_emb_dict[-1] = np.zeros(768)
article_emb_dict[-1] = np.zeros(128)
cate_emb_dict[-1] = np.zeros(128)
subcate_emb_dict[-1] = np.zeros(128)
title_emb_dict[-1] = np.zeros(1024)
body_emb_dict[-1] = np.zeros(1024)
topic_emb_dict[-1] = np.zeros(1024)



# 使用 lazy 模式加载数据
train = pl.scan_parquet("../../inputs/large/train/behaviors.parquet").select(['impression_id','impression_time','article_ids_inview','user_id'])

# 加载文章数据
article = pl.scan_parquet("./features/articles.parquet").select(['article_id','category','subcategory1']).collect().to_pandas()

# 加载历史数据
train_his = pl.scan_parquet("./features/train_his.parquet").select(['user_id','his_article_ids', 'his_cate_ids','his_subcate_ids'])

# 合并数据
train = train.join(train_his, on="user_id", how="left").collect().to_pandas()

# 创建字典
news2cat = dict(zip(list(article["article_id"]), list(article["category"])))
news2subcat = dict(zip(list(article["article_id"]), list(article["subcategory1"])))

# 映射数据
train['cate_ids_inview'] = train['article_ids_inview'].map(lambda x:[news2cat[i] for i in x])

train['subcate_ids_inview'] = train['article_ids_inview'].map(lambda x:[news2subcat[i] for i in x])

def cal_his_cosine(ids_inview,his_ids,emb_dict):
    default_emb = emb_dict[-1]
    sim_res = []
    for item in ids_inview:
        id_emb = emb_dict[item] if item in emb_dict else default_emb
        temp = []
        
        for his_id in his_ids:
            his_emb = emb_dict[his_id] if his_id in emb_dict else default_emb
            s = cosine_similarity(his_emb,id_emb)
            temp.append(s)
        sim_res.append(temp)
        
    return np.array(sim_res)

# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np

# def cal_his_cosine(ids_inview, his_ids, emb_dict):
#     default_emb = emb_dict[-1]
#     # Reshape embeddings to 2D arrays
#     id_embs = np.array([emb_dict[item] if item in emb_dict else default_emb for item in ids_inview])
#     his_embs = np.array([emb_dict[item] if item in emb_dict else default_emb for his_id in his_ids])
    
#     # Calculate cosine similarity
#     sim_res = np.array([cosine_similarity(id_emb, his_embs) for id_emb in id_embs])
#     sim_res = sim_res.reshape(len(ids_inview), len(his_ids))
    
#     return sim_res

from scipy.stats import kurtosis, skew
from scipy.spatial.distance import cdist, euclidean, braycurtis


def w2v_mean(token_list,emb_dict):
    emb_size = len(emb_dict[-1])
    emb_list = [emb_dict[token] if token in emb_dict else np.zeros(emb_size) for token in token_list]
    
    return np.mean(emb_list, axis=0)


def cosine_similarity(A, B):
    dot_product = np.dot(A, B)
    norm_a = np.linalg.norm(A)
    norm_b = np.linalg.norm(B)
    return dot_product / (norm_a * norm_b)

def cal_his_mean_sim_list(ids_inview,his_emb_mean,emb_dict):
    
    emb_list = [emb_dict[item] if item in emb_dict else np.zeros_like(his_emb_mean) for item in ids_inview]
    sim_list = np.dot(emb_list, his_emb_mean) / (np.linalg.norm(emb_list, axis=1) * np.linalg.norm(his_emb_mean)+ 1e-8)
    inviews_emb_mean = w2v_mean(ids_inview,emb_dict)
    inviews_sim_list = np.dot(emb_list, inviews_emb_mean) / (np.linalg.norm(emb_list, axis=1) * np.linalg.norm(his_emb_mean)+ 1e-8)
    
    dis_list = [euclidean(emb, his_emb_mean) for emb in emb_list]
    return sim_list,dis_list,inviews_sim_list


def user_item_dot(user_id, item_id, emb_dict):
    default_emb = emb_dict[-1]
    u_mat = np.mean([emb_dict[u] if u in emb_dict else np.zeros_like(default_emb) for u in user_id],axis=0)
    i_mat = np.stack([emb_dict[i] if i in emb_dict else np.zeros_like(default_emb) for i in item_id])
    # print(u_mat)
    # print(i_mat)
    return np.sum(u_mat * i_mat, axis=1)

def gen_all_article_sims(ids_inview,his_ids,article_emb_dicts):
    results = {}
    for k,v in article_emb_dicts.items():
        # print(k)
        his_emb_mean = w2v_mean(his_ids,v)
        inviews_emb_mean = w2v_mean(ids_inview,v)
        sim_list,dis_list,inviews_sim_list = cal_his_mean_sim_list(ids_inview,his_emb_mean,v)
        results[k+"_his_mean_item_sim_list"] = sim_list
        results[k+'_his_item_all_mean_sim'] = cosine_similarity(his_emb_mean,inviews_emb_mean)
        results[k+'_inview_item_all_mean_sim_list'] = inviews_sim_list
        results[k+"_his_mean_item_euclidean_list"] = dis_list
        results[k+'_inreviws_mean_std'] = np.std(inviews_emb_mean)
        results[k+'_inreviws_mean_skew'] = skew(inviews_emb_mean)
        results[k+'_inreviws_mean_kurt'] = kurtosis(inviews_emb_mean)
        results[k+'_his_mean_std'] = np.std(his_emb_mean)
        results[k+'_his_mean_skew'] = skew(his_emb_mean)
        results[k+'_his_mean_kurt'] = kurtosis(his_emb_mean)
        results[k+'_his_item_dot_list'] = user_item_dot(his_ids,ids_inview,v)
        # his_item_sim = cal_his_cosine(ids_inview,his_ids,v)
        # results[k+'_his_inview_item_sim_max_list'] = his_item_sim.max(1)
        # results[k+'_his_inview_item_sim_mean_list'] = his_item_sim.mean(1)
        # results[k+'_his_inview_item_sim_std_list'] = his_item_sim.std(1)
        # results[k+'_his_inview_item_sim_skew_list'] = skew(his_item_sim,axis=1)
              
        
    return results

article_emb_dicts = {
    "cl":cl_emb_dict,
    "img":img_emb_dict,
    "w2v":w2v_emb_dict,
    "xlm":xlm_emb_dict,
    "bert":bert_emb_dict,
    "title":title_emb_dict,
    "body":body_emb_dict,
    "topic":topic_emb_dict,
    "user":article_emb_dict,
    
}

cate_emb_dicts = {
    "cate_emb_sim":cate_emb_dict
}
subcate_emb_dicts = {
    "subcate_emb_sim":subcate_emb_dict
}

embs = {
    "article":article_emb_dicts,
    "cate":cate_emb_dicts,
    "subcate":subcate_emb_dicts
}



def process_item(item):
    sample = {}

    for key in ['article', 'cate', 'subcate']:
        ids_inview = item[f'{key}_ids_inview']
        his_ids = item[f'his_{key}_ids']
        emb_dicts = embs[key]
        sample.update(gen_all_article_sims(ids_inview, his_ids, emb_dicts))

    # sample['impression_id'] = int(item['impression_id'])
    # sample['user_id'] = int(item['user_id'])

    for key, value in sample.items():
        if key not in ['user_id', 'impression_id']:
            try:
                sample[key] = [round(float(sim), 10) for sim in value]
            except:
                sample[key] = round(float(value), 10)

    return sample


from joblib import Parallel, delayed
from tqdm import tqdm
import json
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

def gen_article_sim_feature(data, save_path):
    chunk_size = 5000000
    num_chunks = len(data) // chunk_size + 1

    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i+1) * chunk_size, len(data))
        chunk = data[start:end]

        results = Parallel(n_jobs=64,backend="multiprocessing")(delayed(process_item)(item) for item in tqdm(chunk))
        
        df = pl.DataFrame(results).to_pandas()
                 
        # df = reduce_memory_usage_pl(df)
        df.to_parquet(save_path+f"_{i}.parquet")

    return "write ok"



####train
import gc 
train = train.to_dict('records')
import warnings
warnings.filterwarnings('ignore')
train_sim = gen_article_sim_feature(train,save_path="../../features/train_sim")
del train 
gc.collect()


val = pl.scan_parquet("../../inputs/large/validation/behaviors.parquet").select(['impression_id','impression_time','article_ids_inview','user_id'])
val_his = pl.scan_parquet("./features/val_his.parquet").select(['user_id','his_article_ids', 'his_cate_ids','his_subcate_ids'])
val = val.join(val_his, on="user_id", how="left").collect().to_pandas()
val['cate_ids_inview'] = val['article_ids_inview'].map(lambda x:[news2cat[i] for i in x])
val['subcate_ids_inview'] = val['article_ids_inview'].map(lambda x:[news2subcat[i] for i in x])

####val
val = val.to_dict('records')
val_sim = gen_article_sim_feature(val,save_path="../../features/val_sim")
del val 
gc.collect()

test = pl.scan_parquet("../../inputs/large/test/behaviors.parquet").select(['impression_id','impression_time','article_ids_inview','user_id'])
test_his = pl.scan_parquet("./features/test_his.parquet").select(['user_id','his_article_ids', 'his_cate_ids','his_subcate_ids'])
test = test.join(test_his, on="user_id", how="left").collect().to_pandas()
test['cate_ids_inview'] = test['article_ids_inview'].map(lambda x:[news2cat[i] for i in x])
test['subcate_ids_inview'] = test['article_ids_inview'].map(lambda x:[news2subcat[i] for i in x])


####test
import gc 
test = test.to_dict('records')
test_sim = gen_article_sim_feature(test,save_path="../../features/test_sim")
del test 
gc.collect()






        
    
    


            



