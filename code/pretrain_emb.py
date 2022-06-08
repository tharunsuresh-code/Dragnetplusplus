# -*- coding: utf-8 -*-
'''
@Time    : 2022-02-22 7:59 p.m.
@Author  : datasnail
@File    : pretrain_emb.py
'''
import random
import sys
import json
import pickle
import numpy as np
from tqdm import tqdm
from loguru import logger
import pretty_errors

import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from StructureModels import NodeEmbed2

class GetLoader(Dataset):
    def __init__(self, feats, labels):
        self.data = feats
        self.label = labels
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels
    def __len__(self):
        return len(self.data)

class HatePredictor(pl.LightningModule):
    def __init__(self, in_dim, hid_dim, suffix, path):
        super().__init__()
        self.suffix = suffix
        self.path = path
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.node_embed = NodeEmbed2(self.in_dim, self.hid_dim)
        self.mlp = nn.Linear(self.hid_dim, 1)

        self.embeddings = []

    def forward(self, embed_mtrx, seq_length):
        x = self.node_embed(embed_mtrx, seq_length)
        hate_intensity = self.mlp(x)
        return hate_intensity

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        embed_mtrx, seq_length, labels = train_batch

        pre = self(embed_mtrx, seq_length)
        loss = F.mse_loss(pre, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, valid_batch, batch_idx):
        embed_mtrx, seq_length, labels = valid_batch

        pre = self(embed_mtrx, seq_length)
        loss = F.mse_loss(pre, labels)
        self.log('valid_loss', loss)
        return loss

    def test_step(self, test_batch, batch_idx):
        embed_mtrx, seq_length, labels = test_batch
        x = self.node_embed(embed_mtrx, seq_length)
        return x

    def test_epoch_end(self, outputs):
        final_emb = torch.cat(outputs)
        # np.save(f'{self.path}pre_train{suffix}.npy', final_emb.cpu().numpy())
        np.save(f'{self.path}ft_embedding{self.suffix}_mpre.npy', final_emb.cpu().numpy())

# python pretrain_emb.py data_covidhate 0.6 10 50 0
try:
    dataset, m_weight, theta, min_num_nodes = str(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    gpu_id = int(sys.argv[5])
    hm_name = '_waseem'#'_davidson'  # _waseem
except:
    m_weight = 0.6
    min_num_nodes = 50
    theta = 5
    dataset = 'data_georgia'  # data_icdm
    # dataset = 'data_covidhate'
    hm_name = '_davidson'
    gpu_id = 0
print(f'{gpu_id}==>{hm_name=},dataset, m_weight, theta, min_num_nodes = {(dataset, m_weight, theta, min_num_nodes)}')


hatescore_suffix = f'{hm_name}_final{m_weight}'
emb_suffix = f'_{theta}_{min_num_nodes}'

data_path = f'/opt/qmeng/DRAGNET_ICDM21/Dataset/datasets/{dataset}/'
# train_num = int(637*0.8)  # 2760
# base_path1 = f'/opt/qmeng/DRAGNET_ICDM21/Dataset/datasets/{dataset}/folded/'
# to_path = f'/opt/qmeng/DRAGNET_ICDM21/Dragnet/{data_name}_data/folded/'

# loads tree nodes
tree_nodes = np.load(f'{data_path}tree_nodes{emb_suffix}.npy')  # num * 51
# training_tree_nodes, test_tree_nodes = tree_nodes[:train_num,:], tree_nodes[train_num:,:]
tid2embed_idx = pickle.load(open(f'{data_path}tid2embed_idx{emb_suffix}.pkl', 'rb'))  # dict


# loads feats
all_embeddings = []
with open(f'{data_path}ft_embedding{emb_suffix}.txt', 'r') as file:
    all_embeddings = file.readlines()
    # pbar = tqdm(total=len(tree_nodes)*51)
    # pbar.set_description("Loading embeddings")
    # while True:
    #     line = file.readline()
    #     if not line:
    #         break
    #     embed = np.array(eval(line))
    #     if embed.shape[0] == 0:
    #         embed = np.zeros((1, 300))
    #     all_embeddings.append(embed)
    #     pbar.update()

def load_json(data_path, suffix):
    with open(f'{data_path}hatescore{suffix}.json', 'r') as rf:
        lex_hate_dict = json.load(rf)
    return lex_hate_dict
final_hate_intensity = load_json(data_path, hatescore_suffix)

def collate_wrapper(batch):
    max_len = max([b[0].shape[0] for b in batch])
    dim = batch[0][0].shape[1]
    embed_mtrx = torch.zeros(0, max_len, dim)
    seq_length = []
    labels = torch.zeros(0, 1)
    for feat,label in batch:
        if feat.shape[1] == 0:
            feat = torch.zeros(1, 1, dim)
        feat = feat.unsqueeze(0)  # 1 * words_num * in_dim
        embed_mtrx = torch.cat(
            [embed_mtrx, F.pad(feat.float(), (0, 0, 0, max_len - feat.shape[1]), "constant", 0)])
        seq_length.append(feat.shape[1])
        labels = torch.cat(
            [labels, torch.Tensor([[label]])])
    return [embed_mtrx, seq_length, labels]

feats = []
labels = []
for tree in tqdm(tree_nodes):  # batch * 51
    for node_id in tree:
        line = all_embeddings[tid2embed_idx[f'{node_id}']]
        line = np.array(eval(line))
        feats.append((torch.from_numpy(line), final_hate_intensity[f'{node_id}']))
        # labels.append([final_hate_intensity[f'{node_id}']])
from sklearn.utils import shuffle

train_num = max(1, int(len(feats)*0.8))
print(f'Train num :{train_num}')

train_loader = DataLoader(feats[:train_num],
                          batch_size=1024,
                          shuffle=True,
                          num_workers=10,
                          collate_fn=collate_wrapper)

valid_loader = DataLoader(feats[train_num:],
                          batch_size=1024,
                          num_workers=10,
                          collate_fn=collate_wrapper)
test_loader = DataLoader(feats,
                          batch_size=1024,
                          num_workers=10,
                          collate_fn=collate_wrapper)

model = HatePredictor(300, 128, emb_suffix, data_path)
checkpoint_callback = ModelCheckpoint(
    monitor='valid_loss',
    filename='HatePre-{lr:.4f}-{epoch}-{valid_loss:.4f}',
    save_top_k=1,
    mode='min',
    save_last=True)
# early stopping
early_stopping_callback = EarlyStopping(
    monitor='valid_loss',
    min_delta=0.0,
    patience=5,
    verbose=False,
    mode='min',
    strict=True)
trainer = pl.Trainer(gpus=[gpu_id],
                     precision=32,
                     max_epochs=200,
                     callbacks=[checkpoint_callback, early_stopping_callback])

trainer.fit(model, train_loader, valid_loader)
trainer.test(ckpt_path='best',
             dataloaders=test_loader)

