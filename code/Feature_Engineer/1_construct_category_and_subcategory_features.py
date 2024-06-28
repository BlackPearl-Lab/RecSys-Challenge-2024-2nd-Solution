import os
import polars as pl
import pandas as pd
import argparse

from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

le = LabelEncoder()


def extract_hist_category_feature(article_df, hist_df, bhv_df):
    hist_df = hist_df.select(
        ['user_id', 'impression_time_fixed', 'article_id_fixed', 'scroll_percentage_fixed', 'read_time_fixed']
    ).explode(['impression_time_fixed', 'article_id_fixed', 'scroll_percentage_fixed', 'read_time_fixed']).rename({
        'article_id_fixed': 'article_id'
    })

    hist_df = hist_df.join(
        article_df.select(['article_id', 'category']),
        on=['article_id'],
        how='left'
    ).unique(['user_id', 'article_id'])

    hist_df = hist_df.with_columns(
        category_hist_click_num=pl.col("category").count().over(["user_id", "category"]),
        hist_category_length=pl.col("category").count().over(["user_id"]),
    ).with_columns(
        category_hist_click_num_ratio=pl.col("category_hist_click_num") / pl.col("hist_category_length"),
        category_hist_click_read_time_sum_ratio=pl.col("read_time_fixed").sum().over(["user_id", "category"]) / pl.col(
            "read_time_fixed").sum().over(["user_id"]),
        category_hist_click_scroll_percentage_mean_ratio=pl.col("scroll_percentage_fixed").mean().over(
            ["user_id", "category"]) / pl.col("scroll_percentage_fixed").mean().over(["user_id"]),
    ).drop([
        'hist_category_imarticle_idr_hour',
        'impression_time_fixed', 'scroll_percentage_fixed', 'read_time_fixed'
    ]).unique(['category', 'user_id'])

    bhv_df = bhv_df.collect().join(
        hist_df.collect(),
        on=['category', 'user_id'],
        how='left'
    ).with_columns(
        impr_category_cnt_new=pl.col("category").count().over(["impression_id", "category"]),
        impr_inview_cnt=pl.col("category").count().over(["impression_id"]),
    ).with_columns(
        impr_category_ratio=pl.col("impr_category_cnt_new") / pl.col("impr_inview_cnt"),
    ).with_columns(
        user_impr_category_num_std=pl.col("impr_category_cnt_new").std().over(["impression_id"]),
        user_impr_category_ratio_std=pl.col("impr_category_ratio").std().over(["impression_id"]),
    ).drop(['category'])

    return bhv_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="construct extended features recovered from yp version.")
    parser.add_argument('--root', type=str, required=True, help='target path')
    args = parser.parse_args()

    root = os.path.join('../../inputs/large', args.root)
    art_filename = 'articles.parquet'
    hist_filename = 'history.parquet'
    bhv_filename = 'behaviors.parquet'

    train_hist_df = pl.scan_parquet(
        os.path.join(root, hist_filename),
        # n_rows=1000
    )

    train_eval_article_df = pl.scan_parquet(
        os.path.join('../../inputs/large', art_filename),
        # n_rows=100
    )

    train_bhv_df = pl.scan_parquet(
        os.path.join('../../inputs/large', args.root, bhv_filename),
    ).select(['impression_id', 'article_ids_inview', 'user_id']).explode("article_ids_inview").rename({
        "article_ids_inview": "article_id"
    }).join(
        train_eval_article_df.select(['article_id', 'category']),
        on=['article_id'],
        how='left'
    )

    train_impr_time_hist_df = extract_hist_category_feature(train_eval_article_df, train_hist_df, train_bhv_df)

    filename = 'zzj0618_v1.parquet'
    ori_add_train_feats = pl.scan_parquet(
        os.path.join('../../features', args.root, filename)
    ).select([
        'impression_id', 'article_id', 'user_id', 'article_impr_hour_inview_mean', 'impr_pub_hour_imprs_mean_impr',
        'total_inviews',
        'total_inviews_mean_impr_diff', 'impr_article_impr_hour_article_cnt_rank_reverse', 'total_ctr',
        'impr_pub_hour_imprs_mean_impr_diff',
        'impr_article_impr_pub_hour_imprs_rank', 'impr_hour_article_cnt_mean_impr_diff',
        'impr_article_impr_pub_interval_rank', 'total_avg_time_mean_impr_diff',
        'impr_pub_hour_imprs', 'impr_article_impr_pub_hour_imprs_rank_reverse', 'impr_pub_hour_imprs_diff',
        'impr_article_total_inviews_rank',
        'impr_article_impr_pub_interval_rank_reverse', 'subcate_str'
    ]).collect()

    ori_add_train_feats = ori_add_train_feats.join(
        train_impr_time_hist_df,
        on=['impression_id', 'user_id', 'article_id'],
        how='left'
    )

    if not os.path.exists(os.path.join('../../features', 'all_extend_feature_zzj01')):
        os.makedirs(os.path.join('../../features', 'all_extend_feature_zzj01'))

    ori_add_train_feats.to_pandas().to_parquet(
        os.path.join('../../features', 'all_extend_feature_zzj01', f'{args.root}.parquet')
    )

    print('../../features/' + 'all_extend_feature_zzj01' + f'{args.root}.parquet' + " save done.")