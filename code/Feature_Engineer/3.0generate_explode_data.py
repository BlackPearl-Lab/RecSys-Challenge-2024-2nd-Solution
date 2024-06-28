import os

import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm

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

def to_coord(x: pl.Expr, max_: int, name: str):
    rad = 2 * np.pi * (x % max_) / max_
    x_sin = rad.sin()
    x_cos = rad.cos()

    return [x_sin.alias(f"{name}_sin"), x_cos.alias(f"{name}_cos")]

def dense_stats_function(col):
    funcs = [
        pl.col(col).mean().alias(f"{col}_mean"),
        pl.col(col).std().alias(f"{col}_std"),
        pl.col(col).skew().alias(f"{col}_skew"),
    ]
    funcs = [i.cast(pl.Float32) for i in funcs]
    return funcs


def sparse_stats_function(col):
    funcs = [
        pl.col(col).n_unique().cast(pl.Int32).alias(f"{col}_nunique"),
        pl.col(col).entropy().cast(pl.Float32).alias(f"{col}_entropy"),
    ]

    return funcs


def generated_user_history_features(history_explode):
    group_col = pd.core.common.flatten(
        [dense_stats_function(i) for i in ["history_read_time", "history_scroll_percentage", "history_time_diff"]]
        + [sparse_stats_function(i) for i in ["history_article_id"]]
        + [
            pl.col("history_impression_time").last().alias(f"history_last_impression_time"),
        ]
    )
    history_features = history_explode.group_by("user_id").agg(group_col)
    return history_features


def generate_features(phase):
    debug_n_rows = None
    if debug:
        debug_n_rows = 300_0000
    behaviors = pl.concat(
        [
            pl.scan_parquet(f"../../inputs/large/{phase}/behaviors.parquet", low_memory=True, n_rows=debug_n_rows),
            pl.scan_parquet(f"../../features/{phase}_all_64D_vectors.parquet", low_memory=True, n_rows=debug_n_rows),
        ],
        how="horizontal",
    )

    history = pl.scan_parquet(f"../../inputs/large/{phase}/history.parquet", low_memory=True, n_rows=debug_n_rows)
    # history_user = pl.scan_parquet(
    #     f"../../features/{phase}_history_user_graph_feat.parquet", low_memory=True, n_rows=debug_n_rows
    # )
    history_explode = (
        history.rename(
            {
                "impression_time_fixed": "history_impression_time",
                "article_id_fixed": "history_article_id",
                "read_time_fixed": "history_read_time",
                "scroll_percentage_fixed": "history_scroll_percentage",
            }
        )
        .explode(["history_impression_time", "history_article_id", "history_read_time", "history_scroll_percentage"])
        .sort(["user_id", "history_impression_time"])
        .with_columns(
            (
                pl.col("history_impression_time").diff().over("user_id").fill_null(0).alias("history_time_diff") / 1e6
            ).cast(pl.Int64)
        )
    )

    behaviors = (
        behaviors.with_row_index()
        .sort(["session_id", "impression_time"])
        .with_columns(
            pl.col("impression_time").diff().over("session_id").dt.total_seconds().alias("last_exp_diff"),
            pl.col("impression_time").diff(-1).over("session_id").dt.total_seconds().alias("next_exp_diff"),
            pl.col("read_time").shift().over("session_id").alias("last_read_time_sort"),
            pl.col("read_time").shift(-1).over("session_id").alias("next_read_time_sort"),
        )
        .sort("index")
        .drop("index")
    )

    list_name = [i for i in behaviors.columns if "_each_" in i]

    if phase != "test":
        explode = (
            behaviors.rename({"article_id": "trigger_id"})
            .explode([INVIEW_COL] + list_name)
            .with_columns(pl.col(INVIEW_COL).is_in(pl.col(CLICK_COL)).alias(LABEL_COL).cast(pl.Int8))
            .drop(CLICK_COL)
        )
    else:
        explode = behaviors.with_columns([pl.lit(0).alias("trigger_id"), pl.lit(-1).alias(LABEL_COL)]).explode(
            [INVIEW_COL] + list_name
        )

    explode = (
        (
            explode.join(
                article_features.select(["article_id", "published_time"] + ARTICLE_COL),
                how="left",
                left_on=INVIEW_COL,
                right_on="article_id",
            )
            .join(
                generated_user_history_features(history_explode),
                how="left",
                on="user_id",
            )
            .with_columns(
                distance_publish_days=(pl.col("impression_time") - pl.col("published_time"))
                .dt.total_days()
                .cast(pl.Int32),
                distance_publish_hours=(pl.col("impression_time") - pl.col("published_time"))
                .dt.total_hours()
                .cast(pl.Int32),
            )
            .with_columns(
                *to_coord(pl.col("published_time").dt.hour(), 24, "published_hour"),
                *to_coord(pl.col("published_time").dt.weekday(), 7, "published_weekday"),
                pl.col("distance_publish_days").clip(upper_bound=3).alias("pulish_3day"),
                pl.col("distance_publish_days").clip(upper_bound=7).alias("pulish_7day"),
                pl.col("distance_publish_days").clip(upper_bound=30),
                pl.col("distance_publish_hours").clip(upper_bound=24),
                *to_coord(pl.col("distance_publish_hours").clip(upper_bound=24), 25, "distance_published_hour"),
            )
            .drop("published_time")
        )
        .with_columns(
            ((pl.col("impression_time") - pl.col("history_last_impression_time")).dt.total_hours())
            .cast(pl.Int64)
            .alias("distance_last_click_time"),
            pl.col("impression_time").dt.hour().cast(pl.Int32).alias("impression_hour"),
            pl.col("impression_time").dt.weekday().cast(pl.Int32).alias("impression_weekday"),
            *to_coord(pl.col("impression_time").dt.hour(), 24, "imporession_hour"),
            *to_coord(pl.col("impression_time").dt.weekday(), 7, "imporession_weekday"),
            pl.arange(1, pl.len() + 1).over("impression_id").alias("impression_position").clip(upper_bound=100),
            pl.lit(phase).alias("phase"),
        )
        .drop(["history_last_impression_time"])
        .collect()
    )

    return reduce_memory_usage_pl(explode)

article_features = pl.scan_parquet("../../features/article_all_features.parquet")
INVIEW_COL = "article_ids_inview"
CLICK_COL = "article_ids_clicked"
LABEL_COL = "click"
ARTICLE_COL = [
    "article_type",
    "category",
    "first_sub_category",
    "sentiment_label",
    "total_inviews",
    "total_pageviews",
    "total_read_time",
    "sentiment_score",
] + [
    "cl_roberta_cosine_similarity",
    "cl_bert_cosine_similarity",
    "bert_roberta_cosine_similarity",
    "title_spelling_errors",
    "title_difficult_words",
    "body_spelling_errors",
    "body_avg_char_per_word",
    "body_flesch_reading_ease",
    "body_smog_index",
    "body_flesch_kincaid_grade",
    "body_coleman_liau_index",
    "body_automated_readability_index",
    "body_dale_chall_readability_score",
    "body_dale_chall_readability_score_v2",
    "body_difficult_words",
    "body_difficult_words_ratio",
    "body_linsear_write_formula",
    "body_gunning_fog",
    "body_text_standard",
    "body_ts_sentence_counts",
]

debug = False
for phase in tqdm(["train", "validation", "test"]):
    data = generate_features(phase)
    data.write_parquet(f"../../caches/{phase}.parquet")
