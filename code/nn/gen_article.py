from pathlib import Path
import polars as pl
import numpy as np
import os
import polars as pl
import numpy as np
import os
from pandas.core.common import flatten
from datetime import datetime
from sklearn.decomposition import PCA
import gc
import warnings
warnings.filterwarnings("ignore")

# root_path
os.chdir('../../')

image_emb_path = "image_embeddings.parquet"
contrast_emb_path = "contrastive_vector.parquet"

base_dataset = 'nn_data'

root = 'inputs'
data_path = f"{root}/large"
train_path = f"{data_path}/train"
dev_path = f"{data_path}/validation"
test_path = f"{data_path}/test"
vector_path = f"{root}/vectors"

feature_path = f"features"

os.makedirs(f'{feature_path}/{base_dataset}/article', exist_ok=True)


def map_feat_id_func(df, column, seq_feat=False):
    feat_set = set(flatten(df[column].to_list()))
    feat_list = list(feat_set)
    feat_list.sort() # important for recover result
    map_dict = dict(zip(feat_list, range(1, 1 + len(feat_set))))
    print(feat_list[0:10])
    if seq_feat:
        df = df.with_columns(pl.col(column).apply(lambda x: [map_dict.get(i, 0) for i in x]))
    else:
        df = df.with_columns(pl.col(column).apply(lambda x: map_dict.get(x, 0)).cast(str))
    return df

def tokenize_seq(df, column, map_feat_id=True, max_seq_length=5, sep="^"):
    df = df.with_columns(pl.col(column).apply(lambda x: x[-max_seq_length:]))
    if map_feat_id:
        df = map_feat_id_func(df, column, seq_feat=True)
    df = df.with_columns(pl.col(column).apply(lambda x: f"{sep}".join(str(i) for i in x)))
    return df


print("Preprocess news info...")
news_file = os.path.join(data_path, "articles.parquet")
news = pl.scan_parquet(news_file)
news = news.fill_null("")
news = (
    news.with_columns(subcat1=pl.col('subcategory').apply(lambda x: str(x[0]) if len(x) > 0 else ""))
    .collect()
)

news = news.with_columns(pl.col('title').str.lengths().alias('title_len'))
news = news.with_columns(pl.col('subtitle').str.lengths().alias('subtitle_len'))
news = news.with_columns(pl.col('body').str.lengths().alias('body_len'))
news = news.with_columns(pl.col('image_ids').list.len().alias('image_ids_num'))
news = news.with_columns(pl.col('total_pageviews').truediv(pl.col('total_inviews')).clip_max(1.0).alias('ctr'))

news = map_feat_id_func(news, "sentiment_label")
news = map_feat_id_func(news, "article_type")
news = tokenize_seq(news, 'ner_clusters', map_feat_id=True, max_seq_length=5) # 取最大长度5个
news = tokenize_seq(news, 'topics', map_feat_id=True, max_seq_length=5)
news = tokenize_seq(news, 'subcategory', map_feat_id=False, max_seq_length=5)
news = tokenize_seq(news, 'entity_groups', map_feat_id=True, max_seq_length=5) # 取最大长度5个

news = news.drop(['title', 'subtitle', 'body', 'image_ids', 'url'])

publich_info_df = pl.read_parquet(feature_path + '/article_publish_48hour_num.parquet')
news = news.join(publich_info_df, on='article_id', how="left")



print("Preprocess pretrained embeddings...")
image_emb_df = pl.read_parquet(os.path.join(vector_path, image_emb_path)) # 1024维
pca = PCA(n_components=64, random_state=42)

image_emb = pca.fit_transform(np.array(image_emb_df["image_embedding"].to_list()))
print("image_embedding.shape", image_emb.shape)
item_dict = {
    "key": image_emb_df["article_id"].cast(str),
    "value": image_emb
}
print("Save image_emb_dim64.npz...")
np.savez(f"{feature_path}/{base_dataset}/article/image_emb_dim64.npz", **item_dict)


contrast_emb_df = pl.read_parquet(os.path.join(vector_path, contrast_emb_path))
contrast_emb = pca.fit_transform(np.array(contrast_emb_df["contrastive_vector"].to_list()))
print("contrast_emb.shape", contrast_emb.shape)
item_dict = {
    "key": contrast_emb_df["article_id"].cast(str),
    "value": contrast_emb
}
print("Save contrast_emb_dim64.npz...")
np.savez(f"{feature_path}/{base_dataset}/article/contrast_emb_dim64.npz", **item_dict)
print("All done.")


use_article_features = [
    'article_id', 'premium', 'article_type', 'ner_clusters', 'entity_groups', 'topics', 'category', 'subcat1',
    'subcategory', 'category_str', 'total_inviews', 'total_pageviews', 'total_read_time',  'ctr', 
    'sentiment_label', 'title_len', 'subtitle_len', 'body_len', 'sentiment_score',
] + ['num_publish_hour_' + str(i) for i in range(48)]

df_article = news.select(use_article_features)

df_article.write_parquet(f'{feature_path}/{base_dataset}/article/article_feats.parquet')
