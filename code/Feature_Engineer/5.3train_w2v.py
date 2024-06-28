import pandas as pd 
from tqdm import tqdm 
import numpy as np 
import os

from gensim.models import Word2Vec
from collections import defaultdict
import logging

train_his = pd.read_parquet("./features/train_his.parquet")
val_his = pd.read_parquet("./features/val_his.parquet")
test_his = pd.read_parquet("./features/test_his.parquet")

data_his = pd.concat([train_his,val_his,test_his])
del train_his,val_his,test_his
import gc 
gc.collect()

train = pd.read_parquet("../../inputs/large/train/behaviors.parquet")
val = pd.read_parquet("../../inputs/large/validation/behaviors.parquet")
test = pd.read_parquet("../../inputs/large/test/behaviors.parquet")

train = train[['article_ids_inview','user_id']]
val = val[['article_ids_inview','user_id']]
test = test[['article_ids_inview','user_id']]

###load article 
article = pd.read_parquet("./features/articles.parquet")
article = article[['article_id','category','subcategory1']]

news2cat = dict(zip(list(article["article_id"]), list(article["category"])))
news2subcat = dict(zip(list(article["article_id"]), list(article["subcategory1"])))

train['cate_ids_inview'] = train['article_ids_inview'].map(lambda x:[news2cat[i] for i in x])
val['cate_ids_inview'] = val['article_ids_inview'].map(lambda x:[news2cat[i] for i in x])
test['cate_ids_inview'] = test['article_ids_inview'].map(lambda x:[news2cat[i] for i in x])

train['subcate_ids_inview'] = train['article_ids_inview'].map(lambda x:[news2subcat[i] for i in x])
val['subcate_ids_inview'] = val['article_ids_inview'].map(lambda x:[news2subcat[i] for i in x])
test['subcate_ids_inview'] = test['article_ids_inview'].map(lambda x:[news2subcat[i] for i in x])


data_be = pd.concat([train,val,test])
del train,val,test
import gc 
gc.collect()

data_his = data_his[['user_id','his_article_ids','his_cate_ids','his_subcate_ids']]
data_his.columns = ['user_id','article_id','category','subcategory1']

data_be = data_be[['user_id','article_ids_inview','cate_ids_inview','subcate_ids_inview']]
data_be.columns = ['user_id','article_id','category','subcategory1']

data_his = pd.concat([data_his,data_be],ignore_index=True)
del data_be
gc.collect()


cols =  ['article_id','category','subcategory1']
logging.basicConfig(
    format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)

for col in cols:
    print(col)
    
    embed_size=128
    save_name=f'features/{col}.model'
    input_docs = []
    for value in tqdm(data_his[col].values):
        input_docs.append(list(value))
    
    w2v = Word2Vec(input_docs, vector_size=embed_size, sg=1, seed=42, workers=12, window=128, min_count=1, epochs=5)
    w2v.save(save_name)