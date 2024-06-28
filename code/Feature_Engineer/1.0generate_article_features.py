import os
from collections import Counter

import numpy as np
import pandas as pd
import polars as pl
import textstat
from joblib import Parallel, delayed, dump, load
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from spellchecker import SpellChecker
from tqdm.auto import tqdm

n_components = 64

single_encode_name = ["article_type", "premium", "category", "first_image", "first_sub_category", "sentiment_label"]
multi_encode_name = ["ner_clusters", "topics", "subcategory", "entity_groups"]
dense_name = ["total_inviews", "total_pageviews", "total_read_time", "sentiment_score"]

contrastive_vector = pl.read_parquet("../../inputs/vectors/Ekstra_Bladet_contrastive_vector/contrastive_vector.parquet")
bert_base_multilingual_cased = pl.read_parquet(
    "../../inputs/vectors/google_bert_base_multilingual_cased/bert_base_multilingual_cased.parquet"
)
word2vec = pl.read_parquet("../../inputs/vectors/Ekstra_Bladet_word2vec/document_vector.parquet")
roberta = pl.read_parquet("../../inputs/vectors/FacebookAI_xlm_roberta_base/xlm_roberta_base.parquet")
images = pl.read_parquet("../../inputs/vectors/Ekstra_Bladet_image_embeddings/image_embeddings.parquet")
article = pl.read_parquet("../../inputs/large/articles.parquet")
article = article.with_columns(first_sub_category=article["subcategory"].list.first())
article = article.with_columns(first_image=article["image_ids"].list.first())
logger.info("Data Prepare...")

def generated_embedding(df):
    name = df.columns[1].split("/")[-1]
    df = df.rename({df.columns[1]: name})
    if not os.path.exists(f"../../features/vectors/{name}-pca-{n_components}d.parquet"):
        pca = PCA(n_components=n_components, random_state=0)
        emb = pca.fit_transform(np.array(df[name].to_list()))
        item_dict = df[["article_id"]].with_columns(pl.Series(name, emb.astype(np.float32)))
        item_dict.write_parquet(f"../../features/vectors/{name}-pca-{n_components}d.parquet")

    return pl.read_parquet(f"../../features/vectors/{name}-pca-{n_components}d.parquet")


cl_dict = generated_embedding(contrastive_vector)
bert_dict = generated_embedding(bert_base_multilingual_cased)
w2v_dict = generated_embedding(word2vec)
roberta_dict = generated_embedding(roberta)
images_dict = generated_embedding(images)

def calc_vector_features(x, y, z):
    similarity_features = {
        "cl_vector_std": x.std(),
        "bert_vector_std": y.std(),
        "roberta_vector_std": z.std(),
        "cl_roberta_cosine_similarity": cosine_similarity([x], [y])[0][0],
        "cl_bert_cosine_similarity": cosine_similarity([x], [z])[0][0],
        "bert_roberta_cosine_similarity": cosine_similarity([y], [z])[0][0],
    }
    return similarity_features


if not os.path.exists("../../features/article_vector_features.parquet"):
    joined_df = contrastive_vector.join(bert_base_multilingual_cased, on="article_id", suffix="_right").join(
        roberta, on="article_id"
    )
    joined_df.columns = ["article_id", "emb1", "emb2", "emb3"]
    vector_df = pd.DataFrame(
        Parallel(n_jobs=64, backend="loky")(
            delayed(calc_vector_features)(x, y, z)
            for x, y, z in tqdm(zip(joined_df["emb1"], joined_df["emb2"], joined_df["emb3"]))
        )
    )
    vector_df["article_id"] = joined_df["article_id"]
    vector_df.to_parquet("../../features/article_vector_features.parquet")
else:
    vector_df = pl.read_parquet("../../features/article_vector_features.parquet")

logger.info("Pariwise vector feature finish...")

def readability(text):
    spell = SpellChecker()
    spelling_errors = len(spell.unknown(text.split()))

    readability_scores = {
        "spelling_errors": spelling_errors,
        "avg_char_per_word": textstat.avg_character_per_word(text),
        "flesch_reading_ease": textstat.flesch_reading_ease(text),
        "smog_index": textstat.smog_index(text),
        "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
        "coleman_liau_index": textstat.coleman_liau_index(text),
        "automated_readability_index": textstat.automated_readability_index(text),
        "dale_chall_readability_score": textstat.dale_chall_readability_score(text),
        "dale_chall_readability_score_v2": textstat.dale_chall_readability_score_v2(text),
        "difficult_words": textstat.difficult_words(text),
        "difficult_words_ratio": textstat.difficult_words(text) / (textstat.miniword_count(text) + 1),
        "linsear_write_formula": textstat.linsear_write_formula(text),
        "gunning_fog": textstat.gunning_fog(text),
        "text_standard": textstat.text_standard(text),
        "ts_sentence_counts": textstat.sentence_count(text),
    }

    return readability_scores


if not os.path.exists("../../features/article_read_features.parquet"):
    body_read_features = pd.DataFrame(
        Parallel(n_jobs=64, backend="loky")(delayed(readability)(text) for text in tqdm(article["body"].to_list()))
    ).add_prefix("body_")
    title_read_features = pd.DataFrame(
        Parallel(n_jobs=64, backend="loky")(delayed(readability)(text) for text in tqdm(article["title"].to_list()))
    ).add_prefix("title_")
    read_features = pd.concat([body_read_features, title_read_features], axis=1)
    read_features["article_id"] = article["article_id"].to_list()
    read_features.to_parquet("../../features/article_read_features.parquet")
else:
    read_features = pl.read_parquet("../../features/article_read_features.parquet")
    
logger.info("Text readability feature finish...")

def article_single_encode(article, encode_columns):
    df = article[["article_id"]]
    for encode_col in encode_columns:
        to_map = article[encode_col]
        unique = to_map.unique().sort().to_list()
        encode_dict = {j: i + 1 for i, j in enumerate(unique)}
        df = df.with_columns(to_map.map_elements(lambda x: encode_dict.get(x, 0)).fill_null(0).cast(pl.Int32))
    return df


def get_multi_encoder_sequence(feat, min_count=1):
    def map_features(column_name, threshold=-1, use_min_vocab=False):
        to_map = article[column_name]
        to_map_length = to_map.map_elements(len)
        flatten_features = np.concatenate(to_map.to_numpy())
        feature_counts = Counter(flatten_features)
        selected_features = [feat for feat, count in feature_counts.items() if count > threshold]
        maxlen = len(selected_features)
        map_dict = {feat: i + 1 for i, feat in enumerate(selected_features)}
        if use_min_vocab:
            return_encode = to_map.map_elements(lambda x: [map_dict.get(feat, maxlen + 1) for feat in x])
        else:
            return_encode = to_map.map_elements(lambda x: [map_dict[feat] for feat in x if feat in map_dict])

        return return_encode, map_dict, to_map_length.max()

    encode, map_dicts, max_length = map_features(feat, min_count, True)
    return encode, len(map_dicts) + 2, max_length


def get_multi_qcut_category(feat):
    tmp = article[feat].qcut(10, allow_duplicates=True)
    return LabelEncoder().fit_transform(tmp)


if not os.path.exists("../../features/article_all_features.parquet"):
    category_name = single_encode_name.copy()

    article_features = article_single_encode(article, single_encode_name)

    for col in dense_name:
        article_features = article_features.with_columns(
            [
                article[col].fill_null(0),
                pl.Series(f"qcut_{col}", get_multi_qcut_category(col)),
            ]
        )
        category_name.append(f"qcut_{col}")

    id_nuniques = {}

    for col in multi_encode_name:
        encode, max_nunique, max_length = get_multi_encoder_sequence(col)
        id_nuniques[col] = max_nunique
        article_features = article_features.with_columns(encode)
    
    
    vector_df = vector_df.with_columns(vector_df['article_id'].cast(pl.Int32))
    read_features = read_features.with_columns(vector_df['article_id'].cast(pl.Int32))
    article_features = article_features.join(vector_df, how="left", on="article_id").join(
        read_features, how="left", on="article_id"
    )
    article_features = article_features.with_columns(
        article_single_encode(article_features, ["title_text_standard", "body_text_standard"])
    )
    category_name += ["title_text_standard", "body_text_standard"]
    id_nuniques.update(dict((article_features[category_name].max() + 1).to_pandas().T.reset_index().values))

    article_features = article_features.with_columns(
        article["published_time"],
        article["subtitle"].str.len_bytes().alias("subtitle_length"),
        article["body"].str.len_bytes().alias("body_length"),
    )
    article_features.write_parquet("../../features/article_all_features.parquet")
    dump(id_nuniques, "../../features/article_id_nunique.pkl")
else:
    article_features = pl.read_parquet("../../features/article_all_features.parquet")
    id_nuniques = load("../../features/article_id_nunique.pkl")
    
logger.info("All feature finish...")
logger.info("All Done")