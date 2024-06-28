import os

import implicit
import numpy as np
import pandas as pd
import polars as pl
import scipy.sparse as sparse
from tqdm import tqdm

USER_ID, ITEM_ID = "user_id", "article_ids_inview"
dim = 128
default_emb = np.zeros(dim + 1)

def get_bpr_embedding(df, emb_size=128):
    user_label, user_idx = pd.factorize(df[USER_ID].to_numpy())
    item_label, item_idx = pd.factorize(df[ITEM_ID].to_numpy())
    sparse_item_user = sparse.csr_matrix((np.ones(len(df)), (user_label, item_label)))
    epoch = 1000
    SEED = 42
    model = implicit.bpr.BayesianPersonalizedRanking(
        factors=emb_size, regularization=1e-3, iterations=epoch, num_threads=8, random_state=SEED, use_gpu=True
    )
    model.fit(sparse_item_user)
    u2emb = dict(zip(user_idx, model.user_factors.to_numpy()))
    i2emb = dict(zip(item_idx, model.item_factors.to_numpy()))

    return u2emb, i2emb


def user_item_dot(user_id, item_id, u2emb, i2emb):
    u_mat = np.stack([u2emb.get(u, default_emb) for u in user_id])
    i_mat = np.stack([i2emb.get(i, default_emb) for i in item_id])
    return np.sum(u_mat * i_mat, axis=1)


def get_bpr_recommend(df):
    name = "bpr"
    u2emb, i2emb = get_bpr_embedding(df, emb_size=dim)
    chunk_size = 1000000
    chunk_cnt = len(df) // chunk_size
    pred = np.concatenate(
        [
            user_item_dot(
                df[USER_ID].to_numpy()[(c * chunk_size): ((c + 1) * chunk_size)],
                df[ITEM_ID].to_numpy()[(c * chunk_size): ((c + 1) * chunk_size)],
                u2emb,
                i2emb,
            )
            for c in tqdm(range(chunk_cnt + 1))
        ]
    )
    return pred


def get_inputs(phase):
    behavior = pl.read_parquet(f"../../caches/{phase}.parquet").select([USER_ID, ITEM_ID])
    history = (
        pl.read_parquet(f"../../inputs/large/{phase}/history.parquet")
        .select([USER_ID, "article_id_fixed"])
        .explode(["article_id_fixed"])
        .rename({"article_id_fixed": ITEM_ID})
    )
    df = pl.concat([behavior, history])
    return df, len(behavior)

for phase in ["train", "validation", "test"]:
    df, length = get_inputs(phase)
    dots = get_bpr_recommend(df)
    np.save(f"../../features/{phase}-bpr.npy", dots[:length])