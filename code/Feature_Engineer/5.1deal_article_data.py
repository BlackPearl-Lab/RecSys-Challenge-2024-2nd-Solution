import pandas as pd  
import numpy as np 
import os 
from tqdm import tqdm 

####load article 
article = pd.read_parquet("../../inputs/large/articles.parquet")

# import os
# os.environ['http_proxy'] = 'http://10.253.34.172:6666'
# os.environ['https_proxy'] = 'http://10.253.34.172:6666' 

# #模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('Xorbits/bge-m3',cache_dir="emb_models")

article['titles'] = article['title']+article['subtitle']
article['topics'] = article['topics'].map(lambda x:" ".join(x))

titles = list(article['titles'])
bodys = list(article['body'])
topics = list(article['topics'])

def gen_subcategory(cate_list):
    try:
        return cate_list[0]
    except:
        return "no category"
    
article['subcategory1'] = article['subcategory'].map(gen_subcategory)

####获取emb
from sentence_transformers import SentenceTransformer
import torch
EMBED_SIZE = 1024
model_path = "./emb_models/Xorbits/bge-m3"
model = SentenceTransformer(model_path)# #moka-ai/m3e-base#'BAAI/bge-base-en-v1.5'
# model = torch.nn.DataParallel(model)
# model.to('cuda')
ids = list(article['article_id'])

def gen_emb(article,texts):
    batch_size = 32 
    buckets = len(article)//batch_size+1
    outputs = []
    for i in tqdm(range(buckets)):
        temp_texts = texts[i*batch_size:(i+1)*batch_size]
        output = model.encode(temp_texts)
        outputs.extend(output)
        
    return outputs
    
title_embs = gen_emb(article,titles)

def format_emb_dict(ids,title_embs):
    res = {}
    for id,emb in zip(ids,title_embs):
        res[id] = emb 
    return res 

if not os.path.exists("features"):
    os.mkdir("features")

title_emb_dict = format_emb_dict(ids,title_embs)
np.save("./features/title_emb_dict",title_emb_dict)

# #####bodys 
body_embs = gen_emb(article,bodys)
body_emb_dict = format_emb_dict(ids,body_embs)
np.save("./features/body_emb_dict",body_emb_dict)

# #####topics 
topics_embs = gen_emb(article,topics)
topics_emb_dict = format_emb_dict(ids,topics_embs)
np.save("./features/topics_emb_dict",topics_emb_dict)

article['titles_len'] = article['titles'].map(lambda x:x.split())
article['topics_len'] = article['topics'].map(lambda x:x.split(" "))

article = article[['article_id','premium','published_time','article_type','category','subcategory1','total_inviews','total_pageviews','total_read_time','sentiment_score','sentiment_label','titles_len','topics_len']]




for col in ['premium','article_type','sentiment_label','subcategory1','category']:
    col_dict =  dict(zip(list(article[col].unique()), range(len(article[col].unique()))))
    article[col] = article[col].map(col_dict)
    
article.to_parquet("./features/articles.parquet")
