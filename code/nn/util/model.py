import torch
from torch import nn
import h5py
import os
import io
import json
import numpy as np
import logging
from collections import OrderedDict
from fuxictr.pytorch.torch_utils import get_initializer
from fuxictr.pytorch import layers
import sys
from pandas.core.common import flatten
from .base_model import CusBaseModel
from fuxictr.pytorch.layers import MLP_Block, DIN_Attention, Dice
from fuxictr.pytorch.torch_utils import get_activation
from tqdm import tqdm




class PretrainedEmbedding(nn.Module):
    def __init__(self,
                 feature_name,
                 feature_spec,
                 pretrain_path,
                 vocab_path,
                 embedding_dim,
                 pretrain_dim,
                 pretrain_usage="init",
                 freeze_emb=None,
                 debug=False):
        """
        Fusion pretrained embedding with ID embedding
        :param: fusion_type: init/sum/concat
        """
        super().__init__()
        assert pretrain_usage in ["init", "sum", "concat"]
        self.pretrain_usage = pretrain_usage
        padding_idx = feature_spec.get("padding_idx", None)
        self.oov_idx = feature_spec["oov_idx"]
        self.freeze_emb = freeze_emb if freeze_emb is not None else feature_spec["freeze_emb"]
        if debug:
            print(f'PretrainedEmbedding: feature_name={feature_name}, vocab_size={feature_spec["vocab_size"]}, pretrain_dim={pretrain_dim}, pretrain_path={pretrain_path}, vocab_path={vocab_path}')
        
        self.debug = debug
        
        self.pretrain_embedding = self.load_pretrained_embedding(feature_spec["vocab_size"],
                                                                 pretrain_dim,
                                                                 pretrain_path,
                                                                 vocab_path,
                                                                 feature_name,
                                                                 freeze=self.freeze_emb,
                                                                 padding_idx=padding_idx)
        if pretrain_usage != "init":
            self.id_embedding = nn.Embedding(feature_spec["vocab_size"],
                                             embedding_dim,
                                             padding_idx=padding_idx)
        self.proj = None
        if pretrain_usage in ["init", "sum"] and embedding_dim != pretrain_dim:
            if debug:
                print('PretrainedEmbedding', 'not_same_dim', pretrain_dim, embedding_dim)
            self.proj = nn.Linear(pretrain_dim, embedding_dim, bias=False) # 维度不一样会手动映射下
        if pretrain_usage == "concat":
            self.proj = nn.Linear(pretrain_dim + embedding_dim, embedding_dim, bias=False)
            



    def reset_parameters(self, embedding_initializer):
        if self.pretrain_usage in ["sum", "concat"]:
            nn.init.zeros_(self.id_embedding.weight) # set oov token embeddings to zeros
            embedding_initializer(self.id_embedding.weight[1:self.oov_idx, :])

    def get_pretrained_embedding(self, pretrain_path):
        logging.info("Loading pretrained_emb: {}".format(pretrain_path))
        if pretrain_path.endswith("h5"):
            with h5py.File(pretrain_path, 'r') as hf:
                keys = hf["key"][:]
                embeddings = hf["value"][:]
        elif pretrain_path.endswith("npz"):
            npz = np.load(pretrain_path)
            keys, embeddings = npz["key"], npz["value"]
        return keys, embeddings

    def load_feature_vocab(self, vocab_path, feature_name):
        with io.open(vocab_path, "r", encoding="utf-8") as fd:
            vocab = json.load(fd)
            vocab_type = type(list(vocab.items())[1][0]) # get key dtype
        return vocab[feature_name], vocab_type

    def load_pretrained_embedding(self, vocab_size, pretrain_dim, pretrain_path, vocab_path,
                                  feature_name, freeze=False, padding_idx=None):
        embedding_layer = nn.Embedding(vocab_size,
                                       pretrain_dim,
                                       padding_idx=padding_idx)
        if self.debug:
            print(f'feat={feature_name}, freeze={freeze}')
            
        if freeze:
            embedding_matrix = np.zeros((vocab_size, pretrain_dim))
        else:
            embedding_matrix = np.random.normal(loc=0, scale=1.e-4, size=(vocab_size, pretrain_dim))
            if padding_idx:
                embedding_matrix[padding_idx, :] = np.zeros(pretrain_dim) # set as zero for PAD
        keys, embeddings = self.get_pretrained_embedding(pretrain_path)
        assert embeddings.shape[-1] == pretrain_dim, f"pretrain_dim={pretrain_dim} not correct."
        vocab, vocab_type = self.load_feature_vocab(vocab_path, feature_name)
        keys = keys.astype(vocab_type) # ensure the same dtype between pretrained keys and vocab keys
        for idx, word in enumerate(keys):
            if word in vocab:
                embedding_matrix[vocab[word]] = embeddings[idx]
        embedding_layer.weight = torch.nn.Parameter(torch.from_numpy(embedding_matrix).float())
        if freeze:
            embedding_layer.weight.requires_grad = False
        return embedding_layer

    def forward(self, inputs):
        mask = (inputs <= self.oov_idx).float()
        pretrain_emb = self.pretrain_embedding(inputs)
        if not self.freeze_emb:
            pretrain_emb = pretrain_emb * mask.unsqueeze(-1)
        if self.pretrain_usage == "init":
            if self.proj is not None:
                feature_emb = self.proj(pretrain_emb)
            else:
                feature_emb = pretrain_emb
        else:
            id_emb = self.id_embedding(inputs)
            id_emb = id_emb * mask.unsqueeze(-1)
            if self.pretrain_usage == "sum":
                if self.proj is not None:
                    feature_emb = self.proj(pretrain_emb) + id_emb
                else:
                    feature_emb = pretrain_emb + id_emb
            elif self.pretrain_usage == "concat":
                feature_emb = torch.cat([pretrain_emb, id_emb], dim=-1)
                feature_emb = self.proj(feature_emb)
        return feature_emb
    

class FeatureEmbedding(nn.Module):
    def __init__(self, 
                 feature_map, 
                 embedding_dim,
                 embedding_initializer="partial(nn.init.normal_, std=1e-4)",
                 required_feature_columns=None,
                 not_required_feature_columns=None,
                 use_pretrain=True,
                 use_sharing=True):
        super(FeatureEmbedding, self).__init__()
        self.embedding_layer = CusFeatureEmbeddingDict(feature_map, 
                                                    embedding_dim,
                                                    embedding_initializer=embedding_initializer,
                                                    required_feature_columns=required_feature_columns,
                                                    not_required_feature_columns=not_required_feature_columns,
                                                    use_pretrain=use_pretrain,
                                                    use_sharing=use_sharing)

    def forward(self, X, feature_source=[], feature_type=[], flatten_emb=False):
        feature_emb_dict = self.embedding_layer(X, feature_source=feature_source, feature_type=feature_type)
        feature_emb = self.embedding_layer.dict2tensor(feature_emb_dict, flatten_emb=flatten_emb)
        return feature_emb


class CusFeatureEmbeddingDict(nn.Module):
    def __init__(self, 
                 feature_map, 
                 embedding_dim, 
                 embedding_initializer="partial(nn.init.normal_, std=1e-4)",
                 required_feature_columns=None,
                 not_required_feature_columns=None,
                 use_pretrain=True,
                 use_sharing=True,
                 seq_encoder_dict=None,
                 freeze_emb=None,
                 debug=False):
        super(CusFeatureEmbeddingDict, self).__init__()
        self._feature_map = feature_map
        self.required_feature_columns = required_feature_columns
        self.not_required_feature_columns = not_required_feature_columns
        self.use_pretrain = use_pretrain
        self.embedding_initializer = embedding_initializer
        self.embedding_layers = nn.ModuleDict()
        self.feature_encoders = nn.ModuleDict()
        self.debug = debug
        for feature, feature_spec in self._feature_map.features.items():
            if self.is_required(feature):
                if self.debug:
                    print('CusFeatureEmbeddingDict', feature, feature_spec, feature_spec.get("embedding_dim", -1))
                    
                if not (use_pretrain and use_sharing) and embedding_dim == 1:
                    feat_dim = 1 # in case for LR
                    if feature_spec["type"] == "sequence":
                        self.feature_encoders[feature] = layers.MaskedSumPooling()
                else:
                    feat_dim = feature_spec.get("embedding_dim", embedding_dim)
                    if feature_spec.get("feature_encoder", None):
                        self.feature_encoders[feature] = self.get_feature_encoder(feature_spec["feature_encoder"])

                    if seq_encoder_dict and feature in seq_encoder_dict:
                        print(f'override seq encoder for feature={feature}...')
                        self.feature_encoders[feature] = seq_encoder_dict[feature] # 覆盖

                # Set embedding_layer according to share_embedding
                if use_sharing and feature_spec.get("share_embedding") in self.embedding_layers:
                    self.embedding_layers[feature] = self.embedding_layers[feature_spec["share_embedding"]]
                    continue

                if feature_spec["type"] == "numeric":
                    self.embedding_layers[feature] = nn.Linear(1, feat_dim, bias=False)
                    
                elif feature_spec["type"] in ["categorical", "sequence"]:
                    if use_pretrain and "pretrained_emb" in feature_spec:
                        pretrain_path = os.path.join(feature_map.data_dir, 
                                                     feature_spec["pretrained_emb"])
                        vocab_path = os.path.join(feature_map.data_dir, 
                                                  "feature_vocab.json")
                        pretrain_dim = feature_spec.get("pretrain_dim", feat_dim)
                        pretrain_usage = feature_spec.get("pretrain_usage", "init")
                        if self.debug:
                            print('CusFeatureEmbeddingDict', 'pretrain', feature, feat_dim, pretrain_dim)
                            
                        self.embedding_layers[feature] = PretrainedEmbedding(feature,
                                                                             feature_spec,
                                                                             pretrain_path,
                                                                             vocab_path,
                                                                             feat_dim,
                                                                             pretrain_dim,
                                                                             pretrain_usage,
                                                                             freeze_emb=freeze_emb,
                                                                             debug=debug)
                    else:
                        padding_idx = feature_spec.get("padding_idx", None)
                        self.embedding_layers[feature] = nn.Embedding(feature_spec["vocab_size"], 
                                                                      feat_dim, 
                                                                      padding_idx=padding_idx)
                        if self.debug:
                            print('CusFeatureEmbeddingDict', feature, feat_dim)
                            
        self.reset_parameters()

    def get_feature_encoder(self, encoder):
        try:
            if type(encoder) == list:
                encoder_list = []
                for enc in encoder:
                    encoder_list.append(eval(enc))
                encoder_layer = nn.Sequential(*encoder_list)
            else:
                encoder_layer = eval(encoder)
            return encoder_layer
        except:
            raise ValueError("feature_encoder={} is not supported.".format(encoder))
        
    def reset_parameters(self):
        embedding_initializer = get_initializer(self.embedding_initializer)
        for k, v in self.embedding_layers.items():
            if "share_embedding" in self._feature_map.features[k]:
                continue
            if type(v) == PretrainedEmbedding: # skip pretrained
                v.reset_parameters(embedding_initializer)
            elif type(v) == nn.Embedding:
                if v.padding_idx is not None:
                    embedding_initializer(v.weight[1:, :]) # set padding_idx to zero
                else:
                    embedding_initializer(v.weight)
                       
    def is_required(self, feature):
        """ Check whether feature is required for embedding """
        feature_spec = self._feature_map.features[feature]
        if feature_spec["type"] == "meta":
            return False
        elif self.required_feature_columns and (feature not in self.required_feature_columns):
            return False
        elif self.not_required_feature_columns and (feature in self.not_required_feature_columns):
            return False
        else:
            return True

    def dict2tensor(self, embedding_dict, feature_list=[], feature_source=[], feature_type=[], 
                    flatten_emb=False, ret_feat_name_list=False):
        if type(feature_source) != list:
            feature_source = [feature_source]
        if type(feature_type) != list:
            feature_type = [feature_type]
        feature_emb_list = []
        feature_names = []
        for feature, feature_spec in self._feature_map.features.items():
            if feature_source and feature_spec["source"] not in feature_source:
                continue
            if feature_type and feature_spec["type"] not in feature_type:
                continue
            if feature_list and feature not in feature_list:
                continue
            if feature in embedding_dict:
                feature_emb_list.append(embedding_dict[feature])
                feature_names.append(feature)
                if self.debug:
                    print('CusFeatureEmbeddingDict', 'dict2tensor', feature, embedding_dict[feature].shape)
                
        if flatten_emb:
            feature_emb = torch.cat(feature_emb_list, dim=-1)
        else:
            feature_emb = torch.stack(feature_emb_list, dim=1)
            
        if ret_feat_name_list:
            return feature_emb, feature_names
        else:
            return feature_emb

    def forward(self, inputs, feature_source=[], feature_type=[]):
        if type(feature_source) != list:
            feature_source = [feature_source]
        if type(feature_type) != list:
            feature_type = [feature_type]
        feature_emb_dict = OrderedDict()
        for feature, feature_spec in self._feature_map.features.items():
            if feature_source and feature_spec["source"] not in feature_source:
                continue
            if feature_type and feature_spec["type"] not in feature_type:
                continue
            if feature in self.embedding_layers:
                if feature_spec["type"] == "numeric":
                    inp = inputs[feature].float().view(-1, 1)
                    embeddings = self.embedding_layers[feature](inp)
                elif feature_spec["type"] == "categorical":
                    inp = inputs[feature].long()
                    embeddings = self.embedding_layers[feature](inp)
                elif feature_spec["type"] == "sequence":
                    inp = inputs[feature].long()
                    embeddings = self.embedding_layers[feature](inp)
                else:
                    raise NotImplementedError
                if feature in self.feature_encoders and self.feature_encoders[feature] is not None:
                    if self.debug:
                        print('feature encoder', feature, embeddings.shape)
                    embeddings = self.feature_encoders[feature](embeddings)
                feature_emb_dict[feature] = embeddings
        return feature_emb_dict

    


class CusDIN(CusBaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="DIN", 
                 gpu=-1, 
                 dnn_hidden_units=[512, 128, 64],
                 dnn_activations="ReLU",
                 attention_hidden_units=[64],
                 attention_hidden_activations="Dice",
                 attention_output_activation=None,
                 attention_dropout=0,
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 net_dropout=0, 
                 batch_norm=False, 
                 din_target_field=[("item_id", "cate_id")],
                 din_sequence_field=[("click_history", "cate_history")],
                 din_use_softmax=False,
                 embedding_regularizer=None, 
                 net_regularizer=None,
                 not_required_feature_columns=None,
                 debug=False,
                 freeze_emb=None,
                 **kwargs):
        super(CusDIN, self).__init__(feature_map,
                                  model_id=model_id, 
                                  gpu=gpu, 
                                  embedding_regularizer=embedding_regularizer, 
                                  net_regularizer=net_regularizer,
                                  **kwargs)
        if not isinstance(din_target_field, list):
            din_target_field = [din_target_field]

        self.num_fields = self.feature_map.num_fields
        
        if not_required_feature_columns:
            self.num_fields = self.num_fields-len(not_required_feature_columns) 
            
        print(f'embedding_dim={embedding_dim}\n,\
                attention_output_activation={attention_output_activation}\n,\
                din_use_softmax={din_use_softmax}\n,\
                attention_hidden_units={attention_hidden_units}\n,\
                dnn_activations={dnn_activations}\n,\
                attention_dropout={attention_dropout}\n,\
                dnn_hidden_units={dnn_hidden_units}\n,\
                batch_norm={batch_norm}\n,\
                embedding_regularizer={embedding_regularizer}\n,\
                net_regularizer={net_regularizer}')
            
        # [('article_id', 'category', 'subcat1', 'sentiment_label', 'article_type'), ('article_id_img', 'article_id_text')]
        self.din_target_field = din_target_field
        if not isinstance(din_sequence_field, list):
            din_sequence_field = [din_sequence_field]
            
        # [('hist_id', 'hist_cat', 'hist_subcat1', 'hist_sentiment', 'hist_type'), ('hist_id_img', 'hist_id_text')]
        self.din_sequence_field = din_sequence_field

        assert len(self.din_target_field) == len(self.din_sequence_field), \
               "len(din_target_field) != len(din_sequence_field)"
        
        
        if isinstance(dnn_activations, str) and dnn_activations.lower() == "dice":
            dnn_activations = [Dice(units) for units in dnn_hidden_units]
            
        self.feature_map = feature_map
        self.embedding_dim = embedding_dim
        
        self.embedding_layer = CusFeatureEmbeddingDict(feature_map, embedding_dim, 
                                                       not_required_feature_columns=not_required_feature_columns,
                                                       debug=debug, freeze_emb=freeze_emb)
        
        self.attention_layers = nn.ModuleList(
            [DIN_Attention(embedding_dim * len(sequence_field) if type(sequence_field) == tuple \
                           else embedding_dim, # 64
                           attention_units=attention_hidden_units,
                           hidden_activations=attention_hidden_activations, # RELU
                           output_activation=attention_output_activation, # None
                           dropout_rate=attention_dropout,
                           use_softmax=din_use_softmax)
             for sequence_field in self.din_sequence_field])
        
        self.din_hidden_layer = MLP_Block(input_dim=len(self.din_target_field[0]) * embedding_dim, 
                                              output_dim=len(self.din_sequence_field[0]) * embedding_dim)
        
        print(f'num_field={self.num_fields}, embedding_dims={self.num_fields * embedding_dim}')
        
        self.dnn = MLP_Block(input_dim=self.num_fields * embedding_dim,
                             output_dim=1,
                             hidden_units=dnn_hidden_units,
                             hidden_activations=dnn_activations,
                             output_activation=self.output_activation, 
                             dropout_rates=net_dropout,
                             batch_norm=batch_norm)
        
                   
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        X = self.get_inputs(inputs)
        feature_emb_dict = self.embedding_layer(X)

        for idx, (target_field, sequence_field) in enumerate(zip(self.din_target_field, 
                                                                 self.din_sequence_field)):
            target_emb = self.get_embedding(target_field, feature_emb_dict)
            sequence_emb = self.get_embedding(sequence_field, feature_emb_dict)
            if target_emb.shape[1] != sequence_emb.shape[2]:
                before_dim = target_emb.shape[1]
                target_emb = self.din_hidden_layer(target_emb)
            
            seq_field = list(flatten([sequence_field]))[0] # flatten nested list to pick the first sequence field
            mask = X[seq_field].long() != 0 # padding_idx = 0 required
            pooling_emb = self.attention_layers[idx](target_emb, sequence_emb, mask)
            
            for field, field_emb in zip(list(flatten([sequence_field])),
                                        pooling_emb.split(self.embedding_dim, dim=-1)):
                feature_emb_dict[field] = field_emb # 覆盖
                
        feature_emb = self.embedding_layer.dict2tensor(feature_emb_dict, flatten_emb=True)
        
        # print(feature_emb.shape)
        y_pred = self.dnn(feature_emb)
        return_dict = {"y_pred": y_pred}
        return return_dict

    def get_embedding(self, field, feature_emb_dict):
        if type(field) == tuple:
            emb_list = [feature_emb_dict[f] for f in field]
            return torch.cat(emb_list, dim=-1)
        else:
            return feature_emb_dict[field]
        
        
class CusMaskNet(CusBaseModel):
    def __init__(self, 
                 feature_map,
                 model_id="MaskNet",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=64,
                 dnn_hidden_units=[64,64,64],
                 dnn_hidden_activations="ReLU",
                 model_type="SerialMaskNet",
                 parallel_num_blocks=1,
                 parallel_block_dim=64,
                 reduction_ratio=1,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 net_dropout=0,
                 emb_layernorm=True,
                 net_layernorm=True,
                 model_id_suffix=None,
                 attention_hidden_units=[64],
                 attention_hidden_activations="Dice",
                 attention_output_activation=None,
                 attention_dropout=0,
                 din_target_field=[("item_id", "cate_id")],
                 din_sequence_field=[("click_history", "cate_history")],
                 din_use_softmax=False,
                 seq_encoder_dict=None,
                 not_required_feature_columns=None,
                 use_din=True,
                 freeze_emb=None,
                 emb_batchnorm=True,
                 debug=False,
                 **kwargs):
        super(CusMaskNet, self).__init__(feature_map,
                                      model_id=model_id,
                                      gpu=gpu,
                                      embedding_regularizer=embedding_regularizer,
                                      net_regularizer=net_regularizer,
                                      model_id_suffix=model_id_suffix,
                                      **kwargs)
        # self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.embedding_layer = CusFeatureEmbeddingDict(feature_map, embedding_dim, 
                                                       seq_encoder_dict=seq_encoder_dict,
                                                       not_required_feature_columns=not_required_feature_columns,
                                                       debug=debug, freeze_emb=freeze_emb)
        
        self.num_fields = self.feature_map.num_fields
        
        if not_required_feature_columns:
            self.num_fields = self.num_fields-len(not_required_feature_columns) 
            
        print(f'num field={self.num_fields}')
        
        self.use_din = use_din
        
        if self.use_din:
            print('use din...')
            if not isinstance(din_target_field, list):
                din_target_field = [din_target_field]

            self.din_target_field = din_target_field

            if not isinstance(din_sequence_field, list):
                din_sequence_field = [din_sequence_field]
            self.din_sequence_field = din_sequence_field

            # [('hist_id', 'hist_cat', 'hist_subcat1', 'hist_sentiment', 'hist_type'), ('hist_id_img', 'hist_id_text')]
            self.din_sequence_field = din_sequence_field

            assert len(self.din_target_field) == len(self.din_sequence_field), \
                   "len(din_target_field) != len(din_sequence_field)"
                
            self.attention_layers = nn.ModuleList(
                [DIN_Attention(embedding_dim * len(sequence_field) if type(sequence_field) == tuple \
                               else embedding_dim, # 64
                               attention_units=attention_hidden_units,
                               hidden_activations=attention_hidden_activations, # RELU
                               output_activation=attention_output_activation, # None
                               dropout_rate=attention_dropout,
                               use_softmax=din_use_softmax)
                 for sequence_field in self.din_sequence_field])
            
            
            self.din_hidden_layer = MLP_Block(input_dim=len(self.din_target_field[0]) * embedding_dim, 
                                              output_dim=len(self.din_sequence_field[0]) * embedding_dim)
            
        print(f'num_field={self.num_fields}, embedding_dims={self.num_fields * embedding_dim}')
        
        if model_type == "SerialMaskNet":
            self.mask_net = SerialMaskNet(input_dim=self.num_fields * embedding_dim,
                                          output_dim=1,
                                          output_activation=self.output_activation,
                                          hidden_units=dnn_hidden_units,
                                          hidden_activations=dnn_hidden_activations,
                                          reduction_ratio=reduction_ratio,
                                          dropout_rates=net_dropout,
                                          layer_norm=net_layernorm)
        elif model_type == "ParallelMaskNet":
            self.mask_net = ParallelMaskNet(input_dim=self.num_fields * embedding_dim,
                                            output_dim=1,
                                            output_activation=self.output_activation,
                                            num_blocks=parallel_num_blocks, 
                                            block_dim=parallel_block_dim, 
                                            hidden_units=dnn_hidden_units,
                                            hidden_activations=dnn_hidden_activations,
                                            reduction_ratio=reduction_ratio,
                                            dropout_rates=net_dropout,
                                            layer_norm=net_layernorm)
            
        self.embedding_dim = embedding_dim
        
        if emb_layernorm:
            self.emb_norm = nn.ModuleList(nn.LayerNorm(embedding_dim) for _ in range(self.num_fields))
        else:
            self.emb_norm = None
        
        if emb_batchnorm:
            self.bn = nn.BatchNorm1d(self.num_fields * embedding_dim)
        else:
            self.bn = None
            
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()

        self.model_to_device()
        self.debug=debug
    
    def forward(self, inputs):
        X = self.get_inputs(inputs)
        feature_emb_dict = self.embedding_layer(X)
        
        if self.use_din:
            for idx, (target_field, sequence_field) in enumerate(zip(self.din_target_field, 
                                                                     self.din_sequence_field)):
                target_emb = self.get_embedding(target_field, feature_emb_dict)
                sequence_emb = self.get_embedding(sequence_field, feature_emb_dict)
                
                if target_emb.shape[1] != sequence_emb.shape[2]:
                    before_dim = target_emb.shape[1]
                    target_emb = self.din_hidden_layer(target_emb)
                    if self.debug:
                        print(idx, before_dim, target_emb.shape[1], sequence_emb.shape[2])
                    
                seq_field = list(flatten([sequence_field]))[0] # flatten nested list to pick the first sequence field
                mask = X[seq_field].long() != 0 # padding_idx = 0 required
                pooling_emb = self.attention_layers[idx](target_emb, sequence_emb, mask)

                for field, field_emb in zip(list(flatten([sequence_field])),
                                            pooling_emb.split(self.embedding_dim, dim=-1)):
                    feature_emb_dict[field] = field_emb # 覆盖
                
        feature_emb, feature_name_list = self.embedding_layer.dict2tensor(feature_emb_dict, flatten_emb=True, ret_feat_name_list=True)
        
        if self.emb_norm is not None:
            feat_list = feature_emb.chunk(self.num_fields, dim=1)
            V_hidden = torch.cat([self.emb_norm[i](feat) for i, feat in enumerate(feat_list)], dim=1)
        else:
            V_hidden = feature_emb
        
        # 参考内容搜索用法
        if self.bn is not None:
            V_hidden = self.bn(V_hidden)
        
        
        y_pred, y_weight_list = self.mask_net(feature_emb.flatten(start_dim=1), V_hidden.flatten(start_dim=1))
        
        # 计算特征重要度
        weight_dict = {}
        for y_weight in y_weight_list:
            y_weight_feat_list = y_weight.chunk(self.num_fields, dim=1)
            for i, weight in enumerate(y_weight_feat_list):
                if feature_name_list[i] not in weight_dict:
                    weight_dict[feature_name_list[i]] = torch.norm(weight, p=2)
                else:
                    weight_dict[feature_name_list[i]] = torch.norm(weight, p=2)

        return_dict = {"y_pred": y_pred, "y_weight": weight_dict}
        
        return return_dict
    
    def get_feature_importance(self, data_generator):
        import collections
        model = self.eval()  # set to evaluation mode
        
        if self.gpus:
            print('wrap model DataParallel')
            model = torch.nn.DataParallel(model, device_ids=self.gpus)
            
        with torch.no_grad():
            y_weight = collections.defaultdict(float)
            logging.info('begin eval feature importance......')
            if self._verbose > 0:
                data_generator = tqdm(data_generator, disable=False, file=sys.stdout)
            
            for batch_data in data_generator:
                return_dict = model(batch_data)
                y_weight_dict = return_dict["y_weight"]
                for k, v in y_weight_dict.items():
                    y_weight[k] += v.data.cpu().numpy()

            return y_weight
    
    def get_embedding(self, field, feature_emb_dict):
        if type(field) == tuple:
            emb_list = [feature_emb_dict[f] for f in field]
            return torch.cat(emb_list, dim=-1)
        else:
            return feature_emb_dict[field]
        

class SerialMaskNet(nn.Module):
    def __init__(self, input_dim, output_dim=None, output_activation=None, hidden_units=[], 
                 hidden_activations="ReLU", reduction_ratio=1, dropout_rates=0, layer_norm=True):
        super(SerialMaskNet, self).__init__()
        if not isinstance(dropout_rates, list):
            dropout_rates = [dropout_rates] * len(hidden_units)
        if not isinstance(hidden_activations, list):
            hidden_activations = [hidden_activations] * len(hidden_units)
        self.hidden_units = [input_dim] + hidden_units
        self.mask_blocks = nn.ModuleList()
        for idx in range(len(self.hidden_units) - 1):
            self.mask_blocks.append(MaskBlock(input_dim, 
                                              self.hidden_units[idx], 
                                              self.hidden_units[idx + 1], 
                                              hidden_activations[idx], 
                                              reduction_ratio, 
                                              dropout_rates[idx],
                                              layer_norm))
        fc_layers = []
        if output_dim is not None:
            fc_layers.append(nn.Linear(self.hidden_units[-1], output_dim))
        if output_activation is not None:
            fc_layers.append(get_activation(output_activation))
        self.fc = None
        if len(fc_layers) > 0:
            self.fc = nn.Sequential(*fc_layers)

    def forward(self, V_emb, V_hidden):
        v_out = V_hidden
        for idx in range(len(self.hidden_units) - 1):
            v_out, v_mask = self.mask_blocks[idx](V_emb, v_out)
        if self.fc is not None:
            v_out = self.fc(v_out)
        return v_out, [v_mask]


class ParallelMaskNet(nn.Module):
    def __init__(self, input_dim, output_dim=None, output_activation=None, num_blocks=1, block_dim=64, 
                 hidden_units=[], hidden_activations="ReLU", reduction_ratio=1, dropout_rates=0, 
                 layer_norm=False):
        super(ParallelMaskNet, self).__init__()
        self.num_blocks = num_blocks
        self.mask_blocks = nn.ModuleList([MaskBlock(input_dim, 
                                                    input_dim, 
                                                    block_dim, 
                                                    hidden_activations, 
                                                    reduction_ratio, 
                                                    dropout_rates,
                                                    layer_norm) for _ in range(num_blocks)])

        self.dnn = MLP_Block(input_dim=block_dim * num_blocks,
                             output_dim=output_dim, 
                             hidden_units=hidden_units,
                             hidden_activations=hidden_activations,
                             output_activation=output_activation,
                             dropout_rates=dropout_rates)

    def forward(self, V_emb, V_hidden):
        block_out = []
        v_masks = []
        for i in range(self.num_blocks):
            v_out, v_mask = self.mask_blocks[i](V_emb, V_hidden)
            block_out.append(v_out)
            v_masks.append(v_mask)
            
        concat_out = torch.cat(block_out, dim=-1)
        v_out = self.dnn(concat_out)
        return v_out, v_masks


class MaskBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, hidden_activation="ReLU", reduction_ratio=1, 
                 dropout_rate=0, layer_norm=True):
        super(MaskBlock, self).__init__()
        self.mask_layer = nn.Sequential(nn.Linear(input_dim, int(hidden_dim * reduction_ratio)),
                                        nn.ReLU(),
                                        nn.Linear(int(hidden_dim * reduction_ratio), hidden_dim))
        hidden_layers = [nn.Linear(hidden_dim, output_dim, bias=False)]
        if layer_norm:
            hidden_layers.append(nn.LayerNorm(output_dim))
        hidden_layers.append(get_activation(hidden_activation))
        if dropout_rate > 0:
            hidden_layers.append(nn.Dropout(p=dropout_rate))
        self.hidden_layer = nn.Sequential(*hidden_layers)

    def forward(self, V_emb, V_hidden):
        V_mask = self.mask_layer(V_emb)
        v_out = self.hidden_layer(V_mask * V_hidden)
        return v_out, V_mask


class ListWiseMaskNet(CusMaskNet):
    def __init__(self, 
                 feature_map,
                 model_id="MaskNet",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=64,
                 dnn_hidden_units=[64,64,64],
                 dnn_hidden_activations="ReLU",
                 model_type="SerialMaskNet",
                 parallel_num_blocks=1,
                 parallel_block_dim=64,
                 reduction_ratio=1,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 net_dropout=0,
                 emb_layernorm=True,
                 net_layernorm=True,
                 model_id_suffix=None,
                 attention_hidden_units=[64],
                 attention_hidden_activations="Dice",
                 attention_output_activation=None,
                 attention_dropout=0,
                 din_target_field=[("item_id", "cate_id")],
                 din_sequence_field=[("click_history", "cate_history")],
                 din_use_softmax=False,
                 seq_encoder_dict=None,
                 not_required_feature_columns=None,
                 use_din=True,
                 freeze_emb=None,
                 emb_batchnorm=True,
                 debug=False,
                 **kwargs):
        super(ListWiseMaskNet, self).__init__(feature_map,
                                         model_id=model_id,
                                         learning_rate=learning_rate,
                                         embedding_dim=embedding_dim,
                                         dnn_hidden_units=dnn_hidden_units,
                                         dnn_hidden_activations=dnn_hidden_activations,
                                         model_type=model_type,
                                         parallel_num_blocks=parallel_num_blocks,
                                         parallel_block_dim=parallel_block_dim,
                                         reduction_ratio=reduction_ratio,
                                         embedding_regularizer=embedding_regularizer,
                                         net_regularizer=net_regularizer,
                                         net_dropout=net_dropout,
                                         emb_layernorm=emb_layernorm,
                                         net_layernorm=net_layernorm,
                                         model_id_suffix=model_id_suffix,
                                         attention_hidden_units=attention_hidden_units,
                                         attention_hidden_activations=attention_hidden_activations,
                                         attention_output_activation=attention_output_activation,
                                         attention_dropout=attention_dropout,
                                         din_target_field=din_target_field,
                                         din_sequence_field=din_sequence_field,
                                         din_use_softmax=din_use_softmax,
                                         seq_encoder_dict=seq_encoder_dict,
                                         not_required_feature_columns=not_required_feature_columns,
                                         use_din=use_din,
                                         freeze_emb=freeze_emb,
                                         emb_batchnorm=emb_batchnorm,
                                         debug=debug,
                                         **kwargs)
    
    def compute_loss_v2(self, y_pred, y_true, tau=1.0):
        # y_pred: NUM_IMPRESSION x 1 
        # y_true: NUM_IMPRESSION x 1
        if torch.sum(y_true) == 0:
            return -1
        
        y_pred = y_pred.reshape(1, -1) # 1 x NUM_IMPRESSION
        y_true = y_true.reshape(1, -1) # 1 x NUM_IMPRESSION
        
        y_pred = y_pred / tau # 温度参数
        return torch.sum(-torch.sum(F.softmax(y_true, dim=1) * F.log_softmax(y_pred, dim=1), dim=1))
    
    def train_step_v2(self, model, batch_data):
        self.optimizer.zero_grad()
        batch_size = batch_data.shape[0]
        
        return_dict = model(batch_data)
        
        labels = self.get_labels(batch_data)
        predictions = return_dict['y_pred']
        
        loss = self.loss_fn(predictions, labels, reduction='mean')
        loss += self.regularization_loss()
        
        X_impression_ids = batch_data[:, self.feature_map.get_column_index('impression_id')].to(self.device).view(-1, 1)
        _, idx = torch.unique(X_impression_ids.squeeze(), return_inverse=True)
        def dynamic_partition(data, partitions, num_partitions):
            res = []
            for i in range(num_partitions):
                query_level_data = data[partitions == i]
                if query_level_data.shape[0] > 0:
                    res.append(query_level_data)
                else:
                    break
            return res
        
        batch_labels = dynamic_partition(labels, idx, batch_size)
        batch_predictions = dynamic_partition(predictions, idx, batch_size)
 
        losses_results = [self.compute_loss(_predictions, _labels)
                              for _labels, _predictions in zip(batch_labels, batch_predictions)]
        listwise_loss = 0

        num_query = 0
        for l in losses_results:
            if l > 0: 
                loss += l
                num_query += 1
                
        listwise_loss /= 1.0*num_query  
        
        listwise_lambda = 0.2
        
        loss += listwise_lambda * listwise_loss # 辅助目标
        
  
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self._max_gradient_norm)
        self.optimizer.step()
        return loss