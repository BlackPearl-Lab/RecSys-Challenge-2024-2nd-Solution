import numpy as np
import pandas as pd
import polars as pl
# from scikit_tabular.feature.multivalue_graph_features import MultiValueGraphFeatureEncoder
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

article = pl.read_parquet("../../inputs/large/articles.parquet")

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


n_components = 64
contrastive_vector = pl.read_parquet("../../inputs/vectors/Ekstra_Bladet_contrastive_vector/contrastive_vector.parquet")
bert_base_multilingual_cased = pl.read_parquet(
    "../../inputs/vectors/google_bert_base_multilingual_cased/bert_base_multilingual_cased.parquet"
)
word2vec = pl.read_parquet("../../inputs/vectors/Ekstra_Bladet_word2vec/document_vector.parquet")
roberta = pl.read_parquet("../../inputs/vectors/FacebookAI_xlm_roberta_base/xlm_roberta_base.parquet")
images = pl.read_parquet("../../inputs/vectors/Ekstra_Bladet_image_embeddings/image_embeddings.parquet")

cl_dict = generated_embedding(contrastive_vector)
bert_dict = generated_embedding(bert_base_multilingual_cased)
w2v_dict = generated_embedding(word2vec)
roberta_dict = generated_embedding(roberta)
images_dict = generated_embedding(images)

cl_mapping = dict(cl_dict.to_pandas().values)
bert_mapping = dict(bert_dict.to_pandas().values)
w2v_mapping = dict(w2v_dict.to_pandas().values)
roberta_mapping = dict(roberta_dict.to_pandas().values)
images_mapping = dict(images_dict.to_pandas().values)

all_ids = article['article_id'].unique().to_list()
for mapping in tqdm([cl_mapping, bert_mapping, w2v_mapping, roberta_mapping, images_mapping]):
    for ids in all_ids:
        if ids not in mapping.keys():
            mapping[ids] = np.zeros(n_components)

from scipy.stats import kurtosis, skew
from scipy.spatial.distance import cdist, euclidean, braycurtis

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    return dot_product / (norm_a * norm_b)


def calc_vector_features(v1, v2):

    similarity_features = {}

    for mapping in [
        ("cl", cl_mapping),
        ("bert", bert_mapping),
        ("w2v", w2v_mapping),
        ("roberta", roberta_mapping),
        ("images", images_mapping),
    ]:
        v1_embedding = np.mean([mapping[1][v] for v in v1], axis=0)
        v2_embedding = np.mean([mapping[1][v] for v in v2], axis=0)
        
        similarity_features.update({
            f"{mapping[0]}_all_mean_cosine": cosine_similarity(v1_embedding, v2_embedding),
            f"{mapping[0]}_each_cosine": [cosine_similarity(mapping[1][i], v2_embedding) for i in v1],
            f"{mapping[0]}_each_euclidean": [euclidean(mapping[1][i], v2_embedding) for i in v1],
            f"{mapping[0]}_inview_std": np.std(v1_embedding),
            f"{mapping[0]}_inview_skew": skew(v1_embedding, axis=None),
            f"{mapping[0]}_inview_kurt": kurtosis(v1_embedding, axis=None),
            f"{mapping[0]}_history_std": np.std(v2_embedding),
            f"{mapping[0]}_history_skew": skew(v2_embedding, axis=None),
        })

    return similarity_features

from joblib import Parallel, delayed


def calc_embedding_for_dataset(phase):
    history = pl.read_parquet(f"../../inputs/large/{phase}/history.parquet", low_memory=True)
    behaviors = pl.read_parquet(f"../../inputs/large/{phase}/behaviors.parquet", low_memory=True)
    behaviors = behaviors.join(history[["user_id", "article_id_fixed"]], how="left", on="user_id")
    to_calc = behaviors[["article_ids_inview", "article_id_fixed"]].to_numpy()
    vector_df = pl.DataFrame(
        pd.DataFrame(
            Parallel(n_jobs=64, backend="multiprocessing")(
                delayed(calc_vector_features)(x, y) for x, y in tqdm(to_calc)
            )
        )
    )
    return vector_df

for phase in ["train", "validation", "test"]:
    data = calc_embedding_for_dataset(phase)
    data.write_parquet(f"../../features/{phase}_all_{n_components}D_vectors.parquet")