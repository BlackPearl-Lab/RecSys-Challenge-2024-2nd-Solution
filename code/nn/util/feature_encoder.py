import numpy as np
from collections import Counter, OrderedDict
import pandas as pd
import polars as pl
import pickle
import os
import logging
import json
import re
import shutil
import glob
from pathlib import Path
import sklearn.preprocessing as sklearn_preprocess
from fuxictr.features import FeatureMap
from fuxictr.preprocess.tokenizer import Tokenizer
from fuxictr.preprocess.normalizer import Normalizer
import multiprocessing as mp
from tqdm import tqdm
import gc




class CusFeatureProcessor(object):
    def __init__(self,
                 feature_cols=[],
                 label_col=[],
                 dataset_id=None, 
                 data_root="../data/",
                 **kwargs):
        logging.info("Set up feature processor...")
        self.data_dir = os.path.join(data_root, dataset_id)
        self.pickle_file = os.path.join(self.data_dir, "feature_processor.pkl")
        self.json_file = os.path.join(self.data_dir, "feature_map.json")
        self.vocab_file = os.path.join(self.data_dir, "feature_vocab.json")
        self.feature_cols = self._complete_feature_cols(feature_cols)
        self.label_cols = label_col if type(label_col) == list else [label_col]
        self.feature_map = FeatureMap(dataset_id, self.data_dir)
        self.feature_map.labels = [col["name"] for col in self.label_cols]
        self.feature_map.group_id = kwargs.get("group_id", None)
        self.dtype_dict = dict((feat["name"], eval(feat["dtype"]) if type(feat["dtype"]) == str else feat["dtype"]) 
                                for feat in self.feature_cols + self.label_cols)
        self.processor_dict = dict()

    def _complete_feature_cols(self, feature_cols):
        full_feature_cols = []
        for col in feature_cols:
            name_or_namelist = col["name"]
            if isinstance(name_or_namelist, list):
                for _name in name_or_namelist:
                    _col = col.copy()
                    _col["name"] = _name
                    full_feature_cols.append(_col)
            else:
                full_feature_cols.append(col)
        return full_feature_cols

    def read_csv(self, data_path, sep=",", n_rows=None, **kwargs):
        logging.info("Reading file: " + data_path)
        file_names = sorted(glob.glob(data_path))
        assert len(file_names) > 0, f"Invalid data path: {data_path}"
        # Require python >= 3.8 for use polars to scan multiple csv files
        file_names = file_names[0]
        ddf = pl.scan_csv(source=file_names, separator=sep, dtypes=self.dtype_dict,
                          low_memory=False, n_rows=n_rows)
        return ddf

    def preprocess(self, ddf):
        logging.info("Preprocess feature columns...")
        all_cols = self.label_cols + self.feature_cols[::-1]
        
        active_cols = []
   
        for col in all_cols:
            name = col["name"]

            if name not in ddf.columns:
                continue
                
            if col.get("active") != True:
                continue
                
            if name in ddf.columns:
                fill_na = "" if col["dtype"] in ["str", str] else 0
                fill_na = col.get("fill_na", fill_na)
                ddf = ddf.with_columns(pl.col(name).fill_null(fill_na).cast(self.dtype_dict[name])) # cast
                
            if col.get("preprocess"):
                preprocess_args = re.split(r"\(|\)", col["preprocess"])
                preprocess_fn = getattr(self, preprocess_args[0])
                ddf = preprocess_fn(ddf, name, *preprocess_args[1:-1])
                ddf = ddf.with_columns(pl.col(name).cast(self.dtype_dict[name]))
                
            active_cols.append(name)
        
        if 'impression_day' in ddf.columns and 'impression_day' not in active_cols:
            ddf = ddf.select(active_cols + ['impression_day'])
        else:
            ddf = ddf.select(active_cols)
            
        return ddf

    def fit_features(self, train_ddf, train_history_ddf, article_ddf, num_buckets=10, min_categr_count=10, ignore_cols=[], **kwargs):
        for col in self.feature_cols:
            name = col["name"]
            
            if name in ignore_cols:
                print(f'skip feat={name}')
                continue
                
            if col["active"]:
                logging.info("Processing column: {}".format(col))
                
                if name in train_ddf:
                    col_series = train_ddf.select(name).collect().to_series().to_pandas()
                elif name in train_history_ddf:
                    col_series = train_history_ddf.select(name).collect().to_series().to_pandas()
                elif name in article_ddf:
                    col_series = article_ddf.select(name).collect().to_series().to_pandas()
                else:
                    print(f'miss col={name}')
                    
                if col["type"] == "meta": # e.g. group_id
                    self.fit_meta_col(col)
                elif col["type"] == "numeric":
                    self.fit_numeric_col(col, col_series)
                elif col["type"] == "categorical":
                    self.fit_categorical_col(col, col_series,
                                             min_categr_count=min_categr_count,
                                             num_buckets=num_buckets)
                elif col["type"] == "sequence":
                    self.fit_sequence_col(col, col_series, 
                                          min_categr_count=min_categr_count)
                else:
                    raise NotImplementedError("feature type={}".format(col["type"]))
        
                    
    def fit(self, train_ddf, train_history_ddf, article_ddf, **kwargs):    
        logging.info("Fit feature processor...")
        self.fit_features(train_ddf, train_history_ddf, article_ddf,  **kwargs)
        self.handle_pretrain_vocab()
        self.save()
    
    def save(self):
        self.feature_map.num_fields = self.feature_map.get_num_fields()
        self.feature_map.set_column_index()
        self.save_pickle(self.pickle_file)
        self.save_vocab(self.vocab_file)
        self.feature_map.save(self.json_file)
        logging.info("Set feature processor done.") 
        
    def handle_pretrain_vocab(self, ignore_cols=[]):
        # Expand vocab from pretrained_emb
        os.makedirs(self.data_dir, exist_ok=True)
        for col in self.feature_cols:
            name = col["name"]
            if name in ignore_cols: 
                continue
            if "pretrained_emb" in col:
                logging.info("Loading pretrained embedding: " + name)
                if "pretrain_dim" in col:
                    self.feature_map.features[name]["pretrain_dim"] = col["pretrain_dim"]
                ext = Path(col["pretrained_emb"]).suffix
                shutil.copy(col["pretrained_emb"],
                            os.path.join(self.data_dir, "pretrained_{}{}".format(name, ext)))
                self.feature_map.features[name]["pretrained_emb"] = "pretrained_{}{}".format(name, ext)
                self.feature_map.features[name]["freeze_emb"] = col.get("freeze_emb", True)
                self.feature_map.features[name]["pretrain_usage"] = col.get("pretrain_usage", "init")
                tokenizer = self.processor_dict[name + "::tokenizer"]
                tokenizer.load_pretrained_vocab(self.dtype_dict[name], col["pretrained_emb"])
                self.processor_dict[name + "::tokenizer"] = tokenizer
                self.feature_map.features[name]["vocab_size"] = tokenizer.vocab_size()
                # Handle share_embedding vocab re-assign
                
        for name, spec in self.feature_map.features.items():
            if name in ignore_cols: 
                continue
            if spec["type"] == "numeric":
                self.feature_map.total_features += 1
            elif spec["type"] in ["categorical", "sequence"]:
                if "share_embedding" in spec:
                    # sync vocab from the shared_embedding field
                    tokenizer = self.processor_dict[name + "::tokenizer"]
                    tokenizer.vocab = self.processor_dict[spec["share_embedding"] + "::tokenizer"].vocab
                    self.processor_dict[name + "::tokenizer"] = tokenizer
                    self.feature_map.features[name].update({"oov_idx": tokenizer.vocab["__OOV__"],
                                                            "vocab_size": tokenizer.vocab_size()})
                else:
                    self.feature_map.total_features += self.feature_map.features[name]["vocab_size"]
                if "pretrained_emb" not in spec: # "oov_idx" not used without pretrained_emb
                    if 'oov_idx' in self.feature_map.features[name]:
                        del self.feature_map.features[name]["oov_idx"]
        
        
        
    def fit_inc(self, train_ddf, train_history_ddf, **kwargs):
        already_fit_columns = {k.split("::")[0] for k in self.processor_dict.keys()}
        logging.info(f'already fitting columns={already_fit_columns}')
        logging.info("Inc Fit...")
        
        self.fit_features(train_ddf, train_history_ddf, ignore_cols=already_fit_columns, **kwargs)
        self.handle_pretrain_vocab(ignore_cols=already_fit_columns)
        self.save()
        
    def copy_from_encoder(self, feature_encoder):
        self.processor_dict = feature_encoder.processor_dict
        for name in feature_encoder.feature_map.features:
            self.feature_map.features[name] = feature_encoder.feature_map.features[name]
        self.feature_map.total_features = feature_encoder.feature_map.total_features

    def fit_meta_col(self, col):
        name = col["name"]
        feature_type = col["type"]
        self.feature_map.features[name] = {"type": feature_type}
        if col.get("remap", True):
            # No need to fit, update vocab in encode_meta()
            tokenizer = Tokenizer(min_freq=1, remap=True)
            self.processor_dict[name + "::tokenizer"] = tokenizer

    def fit_numeric_col(self, col, col_series):
        name = col["name"]
        feature_type = col["type"]
        feature_source = col.get("source", "")
        self.feature_map.features[name] = {"source": feature_source,
                                                "type": feature_type}
        if "feature_encoder" in col:
            self.feature_map.features[name]["feature_encoder"] = col["feature_encoder"]
        if "normalizer" in col:
            normalizer = Normalizer(col["normalizer"])
            normalizer.fit(col_series.dropna().values)
            self.processor_dict[name + "::normalizer"] = normalizer

    def fit_categorical_col(self, col, col_series, min_categr_count=1, num_buckets=10):
        name = col["name"]
        feature_type = col["type"]
        feature_source = col.get("source", "")
        min_categr_count = col.get("min_categr_count", min_categr_count)
        self.feature_map.features[name] = {"source": feature_source,
                                                "type": feature_type}
        if "feature_encoder" in col:
            self.feature_map.features[name]["feature_encoder"] = col["feature_encoder"]
        if "embedding_dim" in col:
            self.feature_map.features[name]["embedding_dim"] = col["embedding_dim"]
        if "emb_output_dim" in col:
            self.feature_map.features[name]["emb_output_dim"] = col["emb_output_dim"]
        if "category_processor" not in col:
            tokenizer = Tokenizer(min_freq=min_categr_count, 
                                  na_value=col.get("fill_na", ""), 
                                  remap=col.get("remap", True))
            tokenizer.fit_on_texts(col_series)
            if "share_embedding" in col:
                self.feature_map.features[name]["share_embedding"] = col["share_embedding"]
                tknzr_name = col["share_embedding"] + "::tokenizer"
                # update vocab of both tokenizers
                self.processor_dict[tknzr_name] = tokenizer.merge_vocab(self.processor_dict[tknzr_name])
                self.feature_map.features[col["share_embedding"]] \
                                .update({"oov_idx": self.processor_dict[tknzr_name].vocab["__OOV__"],
                                         "vocab_size": self.processor_dict[tknzr_name].vocab_size()})
            self.processor_dict[name + "::tokenizer"] = tokenizer
            self.feature_map.features[name].update({"padding_idx": 0,
                                                    "oov_idx": tokenizer.vocab["__OOV__"],
                                                    "vocab_size": tokenizer.vocab_size()})
        else:
            category_processor = col["category_processor"]
            self.feature_map.features[name]["category_processor"] = category_processor
            if category_processor == "quantile_bucket": # transform numeric value to bucket
                num_buckets = col.get("num_buckets", num_buckets)
                qtf = sklearn_preprocess.KBinsDiscretizer(n_bins=num_buckets, encode='ordinal', strategy='quantile')
                # qtf = sklearn_preprocess.QuantileTransformer(n_quantiles=num_buckets + 1)
                qtf.fit(col_series.values.reshape(-1, 1))
                self.feature_map.features[name]["vocab_size"] = num_buckets
                self.processor_dict[name + "::quantiler"] = qtf
                
            elif category_processor == "hash_bucket":
                num_buckets = col.get("num_buckets", num_buckets)
                self.feature_map.features[name]["vocab_size"] = num_buckets
                self.processor_dict[name + "::num_buckets"] = num_buckets
            else:
                raise NotImplementedError("category_processor={} not supported.".format(category_processor))

    def fit_sequence_col(self, col, col_series, min_categr_count=1):
        name = col["name"]
        feature_type = col["type"]
        feature_source = col.get("source", "")
        min_categr_count = col.get("min_categr_count", min_categr_count)
        self.feature_map.features[name] = {"source": feature_source,
                                           "type": feature_type}
        feature_encoder = col.get("feature_encoder", "layers.MaskedAveragePooling()")
        if feature_encoder not in [None, "null", "None", "none"]:
            self.feature_map.features[name]["feature_encoder"] = feature_encoder
        if "embedding_dim" in col:
            self.feature_map.features[name]["embedding_dim"] = col["embedding_dim"]
        if "emb_output_dim" in col:
            self.feature_map.features[name]["emb_output_dim"] = col["emb_output_dim"]
        splitter = col.get("splitter")
        na_value = col.get("fill_na", "")
        max_len = col.get("max_len", 0)
        padding = col.get("padding", "post") # "post" or "pre"
        tokenizer = Tokenizer(min_freq=min_categr_count, splitter=splitter, 
                              na_value=na_value, max_len=max_len, padding=padding,
                              remap=col.get("remap", True))
        tokenizer.fit_on_texts(col_series)
        if "share_embedding" in col:
            self.feature_map.features[name]["share_embedding"] = col["share_embedding"]
            tknzr_name = col["share_embedding"] + "::tokenizer"
            # update vocab of both tokenizers
            self.processor_dict[tknzr_name] = tokenizer.merge_vocab(self.processor_dict[tknzr_name])
            self.feature_map.features[col["share_embedding"]] \
                            .update({"oov_idx": self.processor_dict[tknzr_name].vocab["__OOV__"],
                                     "vocab_size": self.processor_dict[tknzr_name].vocab_size()})
        self.processor_dict[name + "::tokenizer"] = tokenizer
        self.feature_map.features[name].update({"padding_idx": 0,
                                                "oov_idx": tokenizer.vocab["__OOV__"],
                                                "max_len": tokenizer.max_len,
                                                "vocab_size": tokenizer.vocab_size()})

    def transform(self, ddf):
        logging.info("Transform feature columns with ID mapping...")
        data_dict = dict()
        for feature, feature_spec in self.feature_map.features.items():
            if feature in ddf.columns:
                feature_type = feature_spec["type"]
                # logging.info(f'processing, feature={feature}, type={feature_type}')
                col_series = ddf[feature]
                if feature_type == "meta":
                    if feature + "::tokenizer" in self.processor_dict:
                        tokenizer = self.processor_dict[feature + "::tokenizer"]
                        data_dict[feature] = tokenizer.encode_meta(col_series)
                        # Update vocab in tokenizer
                        self.processor_dict[feature + "::tokenizer"] = tokenizer
                    else:
                        data_dict[feature] = col_series.values
                elif feature_type == "numeric":
                    col_values = col_series.values
                    normalizer = self.processor_dict.get(feature + "::normalizer")
                    if normalizer:
                         col_values = normalizer.transform(col_values)
                    data_dict[feature] = col_values
                elif feature_type == "categorical":
                    category_processor = feature_spec.get("category_processor")
                    if category_processor is None:
                        data_dict[feature] = self.processor_dict.get(feature + "::tokenizer").encode_category(col_series)
                    
                    elif category_processor == "quantile_bucket":
                        # raise NotImplementedError
                        transform_data = self.processor_dict.get(feature + "::quantiler").transform(col_series.values.reshape(-1,1))
                        data_dict[feature] = transform_data.squeeze()
                        # print(transform_data)
                    elif category_processor == "hash_bucket":
                        raise NotImplementedError
                        
                elif feature_type == "sequence":
                    data_dict[feature] = self.processor_dict.get(feature + "::tokenizer").encode_sequence(col_series)
        
        for label in self.feature_map.labels:
            if label in ddf.columns:
                data_dict[label] = ddf[label].values
                
        return data_dict

    def load_pickle(self, pickle_file=None):
        """ Load feature processor from cache """
        if pickle_file is None:
            pickle_file = self.pickle_file
        logging.info("Load feature_processor from pickle: " + pickle_file)
        if os.path.exists(pickle_file):
            pickled_feature_processor = pickle.load(open(pickle_file, "rb"))
            if pickled_feature_processor.feature_map.dataset_id == self.feature_map.dataset_id:
                return pickled_feature_processor
        raise IOError("pickle_file={} not valid.".format(pickle_file))

    def save_pickle(self, pickle_file):
        logging.info("Pickle feature_encode: " + pickle_file)
        pickle.dump(self, open(pickle_file, "wb"))

    def save_vocab(self, vocab_file):
        logging.info("Save feature_vocab to json: " + vocab_file)
        vocab = dict()
        for feature, spec in self.feature_map.features.items():
            if spec["type"] in ["categorical", "sequence"]:
                if feature + "::tokenizer" in self.processor_dict:
                    vocab[feature] = OrderedDict(
                        sorted(self.processor_dict[feature + "::tokenizer"].vocab.items(), key=lambda x:x[1]))
        with open(vocab_file, "w") as fd:
            fd.write(json.dumps(vocab, indent=4))

    def copy_from(self, ddf, name, src_name):
        ddf = ddf.with_columns(pl.col(src_name).alias(name))
        return ddf

    

def save_npz(darray_dict, data_path):
    logging.info("Saving data to npz: " + data_path)
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    np.savez(data_path, **darray_dict)
    
    

def save_npz_compressed(darray_dict, data_path):
    logging.info("Saving data to npz compressed: " + data_path)
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    np.savez_compressed(data_path, **darray_dict)


def transform_block(feature_encoder, df_block, filename, compressed=False):
    
    if 'impression_day' in df_block.columns:
        del df_block['impression_day']
        
    darray_dict = feature_encoder.transform(df_block)
    
    if compressed:
        save_npz_compressed(darray_dict, os.path.join(feature_encoder.data_dir, filename))
    else:
        save_npz(darray_dict, os.path.join(feature_encoder.data_dir, filename))
    
def transform(feature_encoder, ddf, filename, block_size=0, num_thread=None):
    ddf = ddf.collect().to_pandas()
    if block_size > 0:
        if num_thread is None:
            num_thread = mp.cpu_count() // 2
        print(f"num_thread={num_thread}")
        
        pool = mp.Pool(num_thread)
        block_id = 0
        for idx in range(0, len(ddf), block_size):
            df_block = ddf.iloc[idx:(idx + block_size)]
            pool.apply_async(
                transform_block,
                args=(feature_encoder,
                      df_block,
                      '{}/part_{:05d}.npz'.format(filename, block_id))
            )
            block_id += 1
        pool.close()
        pool.join()
    else:
        transform_block(feature_encoder, ddf, filename)
       
    
def preprocess(feature_encoder, df_block):
    all_cols = feature_encoder.label_cols + feature_encoder.feature_cols[::-1]
    for col in all_cols:
        name = col["name"]
        if name in df_block.columns:
            fill_na = "" if col["dtype"] in ["str", str] else 0
            fill_na = col.get("fill_na", fill_na)
            df_block[name] = df_block[name].fillna(fill_na)
        if col.get("preprocess"):
            preprocess_args = re.split(r"\(|\)", col["preprocess"])
            if preprocess_args[0] == 'copy_from':
                src_name = preprocess_args[1:-1][0]
                df_block[name] = df_block[src_name]
            else:
                raise NotImplementedError

            df_block[name] = df_block[name].astype(feature_encoder.dtype_dict[name])
                
    active_cols = [col["name"] for col in all_cols if col.get("active") != False]
    
    if 'impression_day' in df_block.columns:
        df_block = df_block[active_cols + ['impression_day']]
    else:
        df_block = df_block[active_cols]
        
    return df_block
  
def transform_by_chunk_parquet(feature_encoder, data_path, history_path, article_path, filename, 
                               block_size=0, num_thread=10, save_npz_compressed=True):
    logging.info("Reading file: " + data_path)
    file_names = sorted(glob.glob(data_path))
    assert len(file_names) > 0, f"Invalid data path: {data_path}"
    # Require python >= 3.8 for use polars to scan multiple csv files
    file_names = file_names[0]
    block_id = 0
        
    import pyarrow.parquet as pq
    reader = pq.ParquetFile(file_names)
    dtype_map = {c: feature_encoder.dtype_dict[c] for c in reader.schema.names if c in feature_encoder.dtype_dict}


    history_df = pd.read_parquet(history_path)
    history_df['user_id'] = history_df['user_id'].astype(int)
    
    article_df = pd.read_parquet(article_path)
    article_df['article_id'] = article_df['article_id'].astype(int)
    
    pool = mp.Pool(num_thread)
    
    df_ans = pd.DataFrame()
        
    start = True
    for ddf in tqdm(reader.iter_batches(batch_size=block_size*num_thread)):
        ddf = ddf.to_pandas().astype(dtype_map)
        ddf['user_id'] = ddf['user_id'].astype(int)
        ddf['article_id'] = ddf['article_id'].astype(int)
        
        ddf = ddf.merge(history_df, on='user_id', how='left')
        ddf = ddf.merge(article_df, on='article_id', how='left')
        
        if start:
            print(ddf.columns)
            print(ddf[['impression_id', 'user_id', 'hist_id', 'hist_cat', 'article_id', 'category', 'total_inviews']].head(10))
            start = False
            
        ddf = preprocess(feature_encoder, ddf)
        for idx in range(0, len(ddf), block_size):
            df_block = ddf.iloc[idx:(idx + block_size)]
            
            df_ans = pd.concat([df_ans, df_block[['impression_id', 'user_id', 'article_id', 'click']]], ignore_index=True)

            pool.apply_async(
                transform_block,
                args=(feature_encoder,
                      df_block,
                      '{}/part_{:05d}.npz'.format(filename, block_id),
                      save_npz_compressed)
            )
            block_id += 1

    pool.close()
    pool.join()
    
    return df_ans

    
def transform_by_chunk_day_parquet(feature_encoder, data_path, history_path, article_path, 
                                   filename, block_size=0, num_thread=10, save_npz_compressed=True):
    logging.info("Reading file: " + data_path)
    file_names = sorted(glob.glob(data_path))
    assert len(file_names) > 0, f"Invalid data path: {data_path}"
    # Require python >= 3.8 for use polars to scan multiple csv files
    file_names = file_names[0]
    
    history_df = pd.read_parquet(history_path)
    history_df['user_id'] = history_df['user_id'].astype(int)
    
    article_df = pd.read_parquet(article_path)
    article_df['article_id'] = article_df['article_id'].astype(int)
    
    import pyarrow.parquet as pq
    reader = pq.ParquetFile(file_names)
    dtype_map = {c: feature_encoder.dtype_dict[c] for c in reader.schema.names if c in feature_encoder.dtype_dict}

    pool = mp.Pool(num_thread)
    
    df_ans = {}
    
    impression_day_block_id_dict = {}
    
    start = True
    for ddf in tqdm(reader.iter_batches(batch_size=block_size*num_thread)):
        ddf = ddf.to_pandas().astype(dtype_map)
        ddf['user_id'] = ddf['user_id'].astype(int)
        ddf['article_id'] = ddf['article_id'].astype(int)
        
        ddf = ddf.merge(history_df, on='user_id', how='left')
        ddf = ddf.merge(article_df, on='article_id', how='left')

        if start:
            print(ddf.columns)
            print(ddf[['impression_id', 'user_id', 'hist_id', 'article_id', 'category', 'total_inviews']].head(10))
            start = False
            
        ddf = preprocess(feature_encoder, ddf)

        # 按照impression_day分组
        for impression_day, group in ddf.groupby('impression_day', sort=False):
            # 初始化编号
            if impression_day not in impression_day_block_id_dict:
                impression_day_block_id_dict[impression_day] = 0
            if impression_day not in df_ans:
                df_ans[impression_day] = pd.DataFrame()
                
            for idx in range(0, len(group), block_size):
                df_block = group.iloc[idx:(idx + block_size)]
                df_ans[impression_day] = pd.concat([df_ans[impression_day], df_block[['impression_id', 'user_id', 'article_id', 'impression_day', 'click']]], ignore_index=True)
                pool.apply_async(
                    transform_block,
                    args=(feature_encoder,
                          df_block,
                          '{}/{}/part_{:05d}.npz'.format(filename, impression_day, 
                                                         impression_day_block_id_dict[impression_day]),
                          save_npz_compressed)
                )
                impression_day_block_id_dict[impression_day] += 1
            gc.collect()
    
    df_ans_final = pd.DataFrame()
    for day in sorted(list(df_ans.keys())): # 排序
        print(day)
        df_ans_final = pd.concat([df_ans_final, df_ans[day]], ignore_index=True)
        
    del df_ans
    gc.collect()
    
    pool.close()
    pool.join()   

    return df_ans_final
    
    
def check_npz(d, all_cols):
    try:
        data_dict = np.load(d)
        save_keys = set(data_dict.keys())
        success = True
        for c in all_cols:
            if c not in save_keys:
                # print(f'miss=col={c}, path={d}')
                success = False
                break
        if not success:
            print(f"文件完整性检查不通过, path={d}")
    except:
        print(f"文件完整性检查不通过, path={d}")
        success = False
        return False
        
    return success
    
def transform_by_chunk_day_parquet_oom_restart(feature_encoder, data_path, history_path, 
                                               article_path, filename, block_size=0, num_thread=10, all_cols=None,
                                               save_npz_compressed=True):
    logging.info("Reading file: " + data_path)
    file_names = sorted(glob.glob(data_path))
    assert len(file_names) > 0, f"Invalid data path: {data_path}"
    # Require python >= 3.8 for use polars to scan multiple csv files
    file_names = file_names[0]
    
    history_df = pd.read_parquet(history_path)
    history_df['user_id'] = history_df['user_id'].astype(int)
    
    article_df = pd.read_parquet(article_path)
    article_df['article_id'] = article_df['article_id'].astype(int)
    
    import pyarrow.parquet as pq
    reader = pq.ParquetFile(file_names)
    dtype_map = {c: feature_encoder.dtype_dict[c] for c in reader.schema.names if c in feature_encoder.dtype_dict}

    pool = mp.Pool(num_thread)
    
    df_ans = {}
    
    impression_day_block_id_dict = {}
    
    start = True
    for ddf in tqdm(reader.iter_batches(batch_size=block_size*num_thread)):
        ddf = ddf.to_pandas().astype(dtype_map)
        ddf['user_id'] = ddf['user_id'].astype(int)
        ddf['article_id'] = ddf['article_id'].astype(int)
        
        ddf = ddf.merge(history_df, on='user_id', how='left')
        ddf = ddf.merge(article_df, on='article_id', how='left')

        if start:
            print(ddf.columns)
            print(ddf[['impression_id', 'user_id', 'hist_id', 'article_id', 'category', 'total_inviews']].head(10))
            start = False
            
        ddf = preprocess(feature_encoder, ddf)

        # 按照impression_day分组
        for impression_day, group in ddf.groupby('impression_day', sort=False):
            # 初始化编号
            if impression_day not in impression_day_block_id_dict:
                impression_day_block_id_dict[impression_day] = 0
            if impression_day not in df_ans:
                df_ans[impression_day] = pd.DataFrame()
            
            for idx in range(0, len(group), block_size):
                df_block = group.iloc[idx:(idx + block_size)]
                df_ans[impression_day] = pd.concat([df_ans[impression_day], df_block[['impression_id', 'user_id', 'article_id', 'impression_day', 'click']]], ignore_index=True)
                save_path = '{}/{}/part_{:05d}.npz'.format(filename, impression_day, 
                                                         impression_day_block_id_dict[impression_day])
                do_transfrom = True
                check_path = os.path.join(feature_encoder.data_dir, save_path)
                # 存在路径且校验通过，则skip
                if os.path.exists(check_path) and check_npz(check_path, all_cols):
                    do_transfrom = False
                    print(f'skip path={check_path}...')
                    
                if do_transfrom:
                    pool.apply_async(
                        transform_block,
                        args=(feature_encoder,
                              df_block,
                              save_path,
                              save_npz_compressed)
                    )
                
                impression_day_block_id_dict[impression_day] += 1
                
            del df_block
            gc.collect()
            
        del ddf
        gc.collect()
    
    df_ans_final = pd.DataFrame()
    for day in sorted(list(df_ans.keys())): # 排序
        print(day)
        df_ans_final = pd.concat([df_ans_final, df_ans[day]], ignore_index=True)
        
    del df_ans
    gc.collect()
    
    pool.close()
    pool.join()   

    return df_ans_final



def transform_by_chunk_day_parquet_oom_restart_opt(feature_encoder, data_path, history_path, 
                                               article_path, filename, block_size=0, num_thread=10, all_cols=None,
                                               save_npz_compressed=True):
    logging.info("Reading file: " + data_path)
    file_names = sorted(glob.glob(data_path))
    assert len(file_names) > 0, f"Invalid data path: {data_path}"
    # Require python >= 3.8 for use polars to scan multiple csv files
    file_names = file_names[0]
    
    history_df = pd.read_parquet(history_path)
    history_df['user_id'] = history_df['user_id'].astype(int)
    
    article_df = pd.read_parquet(article_path)
    article_df['article_id'] = article_df['article_id'].astype(int)
    
    import pyarrow.parquet as pq
    reader = pq.ParquetFile(file_names)
    dtype_map = {c: feature_encoder.dtype_dict[c] for c in reader.schema.names if c in feature_encoder.dtype_dict}

    pool = mp.Pool(num_thread)
    
    df_ans = {}
    
    impression_day_block_id_dict = {}
    
    start = True
    for ddf in tqdm(reader.iter_batches(batch_size=block_size*num_thread)):
        ddf = ddf.to_pandas().astype(dtype_map)
        ddf['user_id'] = ddf['user_id'].astype(int)
        ddf['article_id'] = ddf['article_id'].astype(int)
        
        # 按照impression_day分组
        for impression_day, group in ddf.groupby('impression_day', sort=False):
            # 初始化编号
            if impression_day not in impression_day_block_id_dict:
                impression_day_block_id_dict[impression_day] = 0
            if impression_day not in df_ans:
                df_ans[impression_day] = pd.DataFrame()
            
            for idx in range(0, len(group), block_size):
                df_block = group.iloc[idx:(idx + block_size)]
                df_ans[impression_day] = pd.concat([df_ans[impression_day], df_block[['impression_id', 'user_id', 'article_id', 'impression_day', 'click']]], ignore_index=True)
                save_path = '{}/{}/part_{:05d}.npz'.format(filename, impression_day, 
                                                         impression_day_block_id_dict[impression_day])
                do_transfrom = True
                check_path = os.path.join(feature_encoder.data_dir, save_path)
                # 存在路径且校验通过，则skip
                if os.path.exists(check_path) and check_npz(check_path, all_cols):
                    do_transfrom = False
                    print(f'skip path={check_path}...')
                    
                if do_transfrom:
                    # 延迟分block处理
                    df_block = df_block.merge(history_df, on='user_id', how='left')
                    df_block = df_block.merge(article_df, on='article_id', how='left')
                    df_block = preprocess(feature_encoder, df_block)
                    
                    if start:
                        print(df_block.columns)
                        print(df_block[['impression_id', 'user_id', 'hist_id', 'article_id', 'category', 'total_inviews']].head(10))
                        start = False
  
                    pool.apply_async(
                        transform_block,
                        args=(feature_encoder,
                              df_block,
                              save_path,
                              save_npz_compressed)
                    )
                
                impression_day_block_id_dict[impression_day] += 1
                
            del df_block
            gc.collect()
            
        del ddf
        gc.collect()
    
    df_ans_final = pd.DataFrame()
    for day in sorted(list(df_ans.keys())): # 排序
        print(day)
        df_ans_final = pd.concat([df_ans_final, df_ans[day]], ignore_index=True)
        
    del df_ans
    gc.collect()
    
    pool.close()
    pool.join()   

    return df_ans_final


def transform_by_chunk_parquet_oom_restart_opt(feature_encoder, data_path, history_path, article_path, filename, 
                               block_size=0, num_thread=10, all_cols=None, save_npz_compressed=True):
    logging.info("Reading file: " + data_path)
    file_names = sorted(glob.glob(data_path))
    assert len(file_names) > 0, f"Invalid data path: {data_path}"
    # Require python >= 3.8 for use polars to scan multiple csv files
    file_names = file_names[0]
    block_id = 0
        
    import pyarrow.parquet as pq
    reader = pq.ParquetFile(file_names)
    dtype_map = {c: feature_encoder.dtype_dict[c] for c in reader.schema.names if c in feature_encoder.dtype_dict}


    history_df = pd.read_parquet(history_path)
    history_df['user_id'] = history_df['user_id'].astype(int)
    
    article_df = pd.read_parquet(article_path)
    article_df['article_id'] = article_df['article_id'].astype(int)
    
    pool = mp.Pool(num_thread)
    
    df_ans = pd.DataFrame()
        
    start = True
    for ddf in tqdm(reader.iter_batches(batch_size=block_size*num_thread)):
        ddf = ddf.to_pandas().astype(dtype_map)
        ddf['user_id'] = ddf['user_id'].astype(int)
        ddf['article_id'] = ddf['article_id'].astype(int)
    
        for idx in range(0, len(ddf), block_size):
            df_block = ddf.iloc[idx:(idx + block_size)]
            
            df_ans = pd.concat([df_ans, df_block[['impression_id', 'user_id', 'article_id', 'click']]], ignore_index=True)

            save_path = '{}/part_{:05d}.npz'.format(filename, block_id)
            do_transfrom = True
            check_path = os.path.join(feature_encoder.data_dir, save_path)
            
            if os.path.exists(check_path) and check_npz(check_path, all_cols):
                do_transfrom = False
                print(f'skip path={check_path}...')
                
            if do_transfrom:
                df_block = df_block.merge(history_df, on='user_id', how='left')
                df_block = df_block.merge(article_df, on='article_id', how='left')
                df_block = preprocess(feature_encoder, df_block)
                
                if start:
                    print(df_block.columns)
                    print(df_block[['impression_id', 'user_id', 'hist_id', 'hist_cat', 'article_id', 'category', 'total_inviews']].head(10))
                    start = False
            
                pool.apply_async(
                    transform_block,
                    args=(feature_encoder,
                          df_block,
                          save_path,
                          save_npz_compressed)
                )
                
            block_id += 1

    pool.close()
    pool.join()
    
    return df_ans