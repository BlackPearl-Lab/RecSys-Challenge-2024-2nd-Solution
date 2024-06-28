import os
import polars as pl
import pandas as pd
import argparse

from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

le = LabelEncoder()

bhv_filename = 'behaviors.parquet'
art_filename = 'articles.parquet'
hist_filename = 'history.parquet'


def extract_extend_features(base, root):
    train_art_df = pd.read_parquet(
        os.path.join(base, art_filename)
    )[['article_id', 'subcategory', 'last_modified_time', 'published_time', 'total_pageviews', 'total_inviews',
       'total_read_time']]
    train_art_df['subcate_str'] = train_art_df['subcategory'].apply(lambda x: "_".join([str(i) for i in x]))
    train_art_df['subcate_str'] = le.fit_transform(train_art_df['subcate_str'])
    train_art_df['mod_time'] = train_art_df['last_modified_time'].astype('int64') // 10 ** 6
    train_art_df['pub_time'] = train_art_df['published_time'].astype('int64') // 10 ** 6
    train_art_df['time_interval'] = train_art_df['mod_time'] - train_art_df['pub_time']
    train_art_df['total_ctr'] = train_art_df['total_pageviews'] / train_art_df['total_inviews']
    train_art_df['total_avg_time'] = train_art_df['total_read_time'] / train_art_df['total_pageviews']
    train_art_df = train_art_df[
        ['article_id', 'total_inviews', 'subcate_str', 'pub_time', 'total_ctr', 'total_avg_time']]

    train_bhv_data = pd.read_parquet(
        os.path.join(root, bhv_filename),
        # n_rows=100
    )[['impression_id', 'impression_time', 'user_id', 'article_ids_inview']]

    train_bhv_data['articles_num'] = train_bhv_data['article_ids_inview'].apply(lambda x: len(x))

    train_bhv_data = train_bhv_data.explode("article_ids_inview")
    train_bhv_data.rename({"article_ids_inview": "article_id"}, axis=1, inplace=True)

    train_bhv_data = train_bhv_data.merge(
        train_art_df,
        on=["article_id"],
        how="left"
    )
    train_bhv_data['impr_time'] = train_bhv_data['impression_time'].astype('int64') // 10 ** 6
    train_bhv_data['impr_pub_interval'] = train_bhv_data['impr_time'] - train_bhv_data['pub_time']

    train_bhv_data['impr_pub_hour'] = train_bhv_data['impr_pub_interval'] / 3600
    train_bhv_data['impr_pub_hour'] = train_bhv_data['impr_pub_hour'].astype('int32')

    k = 24
    for i in tqdm(range(24)):
        tmp = train_bhv_data[(train_bhv_data['impr_pub_interval'] <= (i + 1) * 3600) & (
                    train_bhv_data['impr_pub_interval'] > i * 3600)].groupby('article_id').agg(
            {'impr_pub_interval': 'count'}).reset_index()
        tmp.rename(columns={'impr_pub_interval': str(i + 1) + '_hour_impr_num'}, inplace=True)
        train_bhv_data = train_bhv_data.merge(tmp, on='article_id', how='left')

    for i in tqdm(range(k - 1)):
        train_bhv_data[str(i + 2) + '-' + str(i + 1) + '_hour_impr_num'] = train_bhv_data[
                                                                               str(i + 2) + '_hour_impr_num'] - \
                                                                           train_bhv_data[str(i + 1) + '_hour_impr_num']

    df_feat = pd.DataFrame()
    for i in tqdm(range(k)):
        if i < k - 1:
            tmp = train_bhv_data[train_bhv_data['impr_pub_hour'] == i][
                ['article_id', 'impr_pub_hour', str(i + 1) + '_hour_impr_num',
                 str(i + 2) + '-' + str(i + 1) + '_hour_impr_num']]
            tmp.rename(columns={str(i + 1) + '_hour_impr_num': 'impr_pub_hour_imprs'}, inplace=True)
            tmp.rename(columns={str(i + 2) + '-' + str(i + 1) + '_hour_impr_num': 'impr_pub_hour_imprs_diff'},
                       inplace=True)
            tmp = tmp.drop_duplicates()
            if i == 0:
                df_feat = tmp
            else:
                df_feat = pd.concat([df_feat, tmp], ignore_index=True)
        else:
            tmp = train_bhv_data[train_bhv_data['impr_pub_hour'] == i][
                ['article_id', 'impr_pub_hour', str(i + 1) + '_hour_impr_num']]
            tmp = tmp.drop_duplicates()
            tmp.rename(columns={str(i + 1) + '_hour_impr_num': 'impr_pub_hour_imprs'}, inplace=True)
            df_feat = pd.concat([df_feat, tmp], ignore_index=True)
    train_bhv_data = train_bhv_data.merge(
        df_feat, on=['article_id', 'impr_pub_hour'], how='left'
    )
    train_bhv_data.drop(columns=['impr_pub_hour'], inplace=True)
    train_bhv_data.drop(columns=[col for col in train_bhv_data.columns if '_hour_impr_num' in col], inplace=True)

    df_feat = train_bhv_data[['article_id', 'impr_time', 'impression_id']]
    df_feat['article_impr_hour'] = (df_feat['impr_time'] / 3600).astype(int)
    df_tmp = df_feat.groupby(['article_id', 'article_impr_hour']).size().reset_index(name='impr_hour_article_cnt')
    train_bhv_data['article_impr_hour'] = (train_bhv_data['impr_time'] / 3600).astype(int)
    train_bhv_data = train_bhv_data.merge(df_tmp, on=['article_id', 'article_impr_hour'], how='left')
    # train_bhv_data.drop(columns=['article_impr_hour'],inplace=True)

    cols = ['impr_pub_hour_imprs', 'total_inviews', 'total_avg_time', 'impr_hour_article_cnt']
    df_feat = train_bhv_data[cols + ['article_id', 'impression_id']].drop_duplicates()
    for c in cols:
        df_feat[c + '_mean_impr'] = df_feat.groupby(['impression_id'])[c].transform('mean')
    df_feat.drop(columns=cols, inplace=True)
    df_feat.drop_duplicates(inplace=True)
    train_bhv_data = train_bhv_data.merge(df_feat, on=['impression_id', 'article_id'], how='left')
    for c in cols:
        train_bhv_data[c + '_mean_impr_diff'] = train_bhv_data[c] - train_bhv_data[c + '_mean_impr']

    train_bhv_data.drop(columns=['impression_time'], inplace=True)

    cols = ['impr_pub_interval', 'impr_pub_hour_imprs', 'total_inviews', 'impr_hour_article_cnt']
    for c in tqdm(cols):
        train_bhv_data = train_bhv_data.sort_values(by=['impression_id', c])
        print(c + '_rank')
        train_bhv_data['impr_article_' + c + '_rank'] = train_bhv_data.groupby(['impression_id']).cumcount() + 1
        print(c + '_rank_reverse')
        train_bhv_data = train_bhv_data.sort_values(by=['impression_id', c], ascending=False)
        train_bhv_data['impr_article_' + c + '_rank_reverse'] = train_bhv_data.groupby(['impression_id']).cumcount() + 1

    train_bhv_data['article_impr_hour'] = (train_bhv_data['impr_time'] / 3600).astype(int)
    df_tmp = train_bhv_data.groupby(['article_id', 'article_impr_hour'])['articles_num'].mean().reset_index()
    df_tmp.rename(columns={'articles_num': 'article_impr_hour_inview_mean'}, inplace=True)
    train_bhv_data = train_bhv_data.merge(df_tmp, on=['article_id', 'article_impr_hour'], how='left')
    train_bhv_data.drop(columns=['article_impr_hour'], inplace=True)
    print(train_bhv_data.shape)

    return train_bhv_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="construct extended features recovered from yp version.")
    parser.add_argument('--root', type=str, required=True, help='target path')
    args = parser.parse_args()

    root = os.path.join('../../inputs/large', args.root)

    extend_sample_features = extract_extend_features('../../inputs/large', root)

    if not os.path.exists(os.path.join('features', args.root)):
        os.makedirs(os.path.join('features', args.root))
    extend_sample_features.to_parquet(
        os.path.join('features', args.root, 'zzj0618_v1.parquet')
    )
    print(f"../../features/{args.root}/zzj0618_v1.parquet save done.")