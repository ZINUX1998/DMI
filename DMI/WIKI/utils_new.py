import torch
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
import numpy as np

import argparse
import os
import sys
import time
import pickle
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
from sklearn import metrics
from sklearn.metrics import average_precision_score,auc,precision_recall_fscore_support

import torch.nn as nn
from torch.nn import functional as F
from torch.nn import Parameter

from argparse import ArgumentParser
from collections import Counter

from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix, coo_matrix, vstack
from scipy.spatial.distance import pdist

from scipy import sparse

# PyTorch
from torch.nn import Linear



def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)



def save_pickle(o, fname):
    with open(fname, 'wb') as handle:
        pickle.dump(o, handle, protocol=pickle.HIGHEST_PROTOCOL)




################################### train data ############################
##### link prediction
def get_LP_train_test_data():
    
    with open('Wikipedia/train.csr.pickle', 'rb') as f:
        csr_train = pickle.load(f)
        print('train:', csr_train.shape, csr_train.nnz)
    with open('Wikipedia/test.csr.pickle', 'rb') as f:
        csr_test = pickle.load(f)
        print('test:', csr_test.shape, csr_test.nnz)
    
    num_U, num_V = csr_train.shape
    train_src, train_dst = csr_train.nonzero()
    test_src, test_dst = csr_test.nonzero()
    train = np.stack((train_src, train_dst), axis=-1)
    test = np.stack((test_src, test_dst), axis=-1)
    
    # generate LP dataset
    n_test_pos = csr_test.nnz
    n_train_pos = csr_train.nnz * 4

    max_sample = (n_test_pos + n_train_pos) * 2
    print('max_sample:', max_sample)
    rand_row = torch.randint(0, num_U, (max_sample,))
    rand_col = torch.randint(0, num_V, (max_sample,))
    neg_g = torch.sparse_coo_tensor(indices=torch.stack((rand_row, rand_col), dim=0), values=torch.ones_like(rand_row), size=(num_U, num_V))
    print('initial neg_g:', neg_g.shape, neg_g._nnz())
    
    pos_g_edges = torch.cat((torch.from_numpy(train), torch.from_numpy(test)), dim=0).t()
    pos_g = torch.sparse_coo_tensor(indices=pos_g_edges, values=-torch.ones_like(pos_g_edges[0]), size=(num_U, num_V))
    print('pos_g:', pos_g.shape, pos_g._nnz())
    
    masked_g = (neg_g * pos_g).coalesce()
    neg_g = (neg_g + masked_g).coalesce()
    neg_indices = neg_g.indices()[:, neg_g.values() > 0]  # (2, n_neg)
    neg_indices = neg_indices.t()  # (n_neg, 2)
    if len(neg_indices) < n_test_pos + n_train_pos:
        raise ValueError('Generated negative samples are less then (test positive samples + train positive samples)!')
    
    shuffled_neg_indices = np.random.permutation(len(neg_indices))

    lp_pos_train = train
    lp_pos_test = test
    print('lp_pos_train:', lp_pos_train.shape)
    print('lp_pos_test:', lp_pos_test.shape)
    train_posi_edges_number = lp_pos_train.shape[0]
    test_posi_edges_number = lp_pos_test.shape[0]

    lp_neg = neg_indices[shuffled_neg_indices]
    lp_neg_train = lp_neg[:n_train_pos]
    lp_neg_test = lp_neg[n_train_pos: n_train_pos + n_test_pos]
    print('lp_neg:', lp_neg.shape)
    print('lp_neg_train:', lp_neg_train.shape)
    print('lp_neg_test:', lp_neg_test.shape)
    train_neg_edges_number = lp_neg_train.shape[0]
    test_neg_edges_number = lp_neg_test.shape[0]


    train_edge_index = torch.cat([torch.tensor(lp_pos_train), lp_neg_train], dim=0)
    test_edge_index = torch.cat([torch.tensor(lp_pos_test), lp_neg_test], dim=0)
    print('train_edge_index:', train_edge_index.shape)
    print('test_edge_index:', test_edge_index.shape)

    train_U_TRAIN_INDEX = train_edge_index[:, 0]
    train_V_TRAIN_INDEX = train_edge_index[:, 1]
    train_edge_label = list(np.ones(train_posi_edges_number)) + list(np.zeros(train_neg_edges_number))
    train_edge_label = torch.tensor(train_edge_label)

    test_U_TRAIN_INDEX = test_edge_index[:, 0]
    test_V_TRAIN_INDEX = test_edge_index[:, 1]
    test_edge_label = list(np.ones(test_posi_edges_number)) + list(np.zeros(test_neg_edges_number))
    test_edge_label = torch.tensor(test_edge_label)

    # 先转换成torch能识别的dataset
    train_dataset = data.TensorDataset(train_U_TRAIN_INDEX, train_V_TRAIN_INDEX, train_edge_label.float())

    # 把dataset放入DataLoader
    train_data_loader = data.DataLoader(
        dataset = train_dataset,
        batch_size = 256,             # 每批提取的数量
        shuffle = True,             # 要不要打乱数据（打乱比较好）
        num_workers = 2             # 多少线程来读取数据
    )
    
    # 先转换成torch能识别的dataset
    test_dataset = data.TensorDataset(test_U_TRAIN_INDEX, test_V_TRAIN_INDEX, test_edge_label.float())

    # 把dataset放入DataLoader
    test_data_loader = data.DataLoader(
        dataset = test_dataset,
        batch_size = 1280,             # 每批提取的数量
        shuffle = True,             # 要不要打乱数据（打乱比较好）
        num_workers = 2             # 多少线程来读取数据
    )
    
    return num_U, num_V, train_data_loader, test_data_loader


#### 推荐系统
class TrainDataset(Dataset):
    def __init__(self, csr_train, csr_all, n_negs):
        # self.csr_train = csr_train
        self.num_edge = csr_train.nnz
        self.num_U, self.num_V = csr_train.shape
        self.num_negs = n_negs
        self.src, self.dst = csr_train.nonzero()
        self.src_torch = torch.from_numpy(self.src)
        self.dst_torch = torch.from_numpy(self.dst)
        self.csr_all = csr_all
        
    def __len__(self):
        return self.num_edge
    
    def __getitem__(self, idx):
        neg_idx_V = None
        if self.num_negs > 0:
            # neg_idx_V = torch.randint(0, self.num_V, (self.num_negs,))
            neg_idx = np.random.randint(self.num_V, size=2*self.num_negs)
            neg_idx_array = self.csr_all[self.src[idx], neg_idx].toarray()
            if neg_idx_array[0, :self.num_negs].sum() > 0:
                neg_idx_true = np.argwhere(neg_idx_array==0)[:,1]
                choose_neg_idx_V = neg_idx_true[:self.num_negs]
                neg_idx_V = neg_idx[choose_neg_idx_V]
            else:
                neg_idx_V = neg_idx[:self.num_negs]
        return self.src_torch[idx], self.dst_torch[idx], torch.from_numpy(neg_idx_V)
    
    @staticmethod
    def collate_fn(data):
        idx_U = torch.stack([_[0] for _ in data], dim=0)
        pos_idx_V = torch.stack([_[1] for _ in data], dim=0)
        if data[0][2] is not None:
            neg_idx_V = torch.stack([_[2] for _ in data], dim=0)
        else:
            neg_idx_V = None
        return idx_U, pos_idx_V, neg_idx_V


def get_RS_train_test_edges(n_negs = 10):
    split_RS_train_test_matrix()

    with open('dblp/all_RS.csr.pickle', 'rb') as f:
        csr_all = pickle.load(f)
        print('all:', csr_all.shape, csr_all.nnz)
    with open('dblp/train_RS.csr.pickle', 'rb') as f:
        csr_train = pickle.load(f)
        print('train:', csr_train.shape, csr_train.nnz)
    with open('dblp/test_RS.csr.pickle', 'rb') as f:
        csr_test = pickle.load(f)
        print('test:', csr_test.shape, csr_test.nnz)
    

    num_U, num_V = csr_train.shape

    # dataloader
    print('construct dataloader...')
    train_dataloader = DataLoader(
            TrainDataset(csr_train, csr_all, n_negs), 
            batch_size = 256,
            shuffle = True, 
            num_workers = 0,
            collate_fn = TrainDataset.collate_fn
        )

    return train_dataloader, csr_train, csr_test



#############################################  评价指标  ####################################
########### 链路预测
def computer_prediction(y_test, y_pred):

    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
    average_precision = average_precision_score(y_test, y_pred)
    auc_roc, auc_pr = metrics.auc(fpr, tpr), average_precision
    
    return auc_pr, auc_roc



def predict_edges(TEST_LHS, TEST_RHS, eval_model, SET_ALPHA):
    eval_model.eval()
    
    with torch.no_grad():
        
        STRUCTURE_CLUSTERS = eval_model.STRUCTURE_CLUSTERS.weight.detach()

        U_DIS = F.cosine_similarity(TEST_LHS.unsqueeze(1), STRUCTURE_CLUSTERS.unsqueeze(0), dim=2)
        U_numerator = eval_model.U_CLUSTER_NETWORK(U_DIS)       #soft assignment
        U_soft_assignments = (U_numerator.t() / torch.sum(U_numerator, 1)).t()       #soft assignment

        V_DIS = F.cosine_similarity(TEST_RHS.unsqueeze(1), STRUCTURE_CLUSTERS.unsqueeze(0), dim=2)
        V_assignments = eval_model.V_CLUSTER_NETWORK(V_DIS) # (batch_size, 1 + neg_edges, cluster_numbers)
        
        CLUSTER_SIM = torch.mul(U_soft_assignments, V_assignments)
        PROB_CLUSTER = torch.sum(CLUSTER_SIM, dim=1)
        
        distance_U_V = torch.cosine_similarity(TEST_LHS, TEST_RHS) + 1.0
        PROB_U_V = distance_U_V / 2.0
        
        LINK_PROB = SET_ALPHA*PROB_CLUSTER + (1-SET_ALPHA)*PROB_U_V

        return LINK_PROB



########## 推荐系统 
def precision_recall(r, k, n_ground_truth):
    right_pred = r[:, :k].sum(1)  # (batch, )
    n_ground_truth_denomitor = n_ground_truth.clone()
    n_ground_truth_denomitor[n_ground_truth_denomitor == 0] = 1
    batch_recall = (right_pred / n_ground_truth_denomitor).sum()
    # batch_precision = right_pred.sum() / k
    # return batch_recall, batch_precision
    return batch_recall


def ndcg(r, k, n_ground_truth):
    pred_data = r[:, :k]
    device = pred_data.device
    max_r = (torch.arange(k, device=device).expand_as(pred_data) < n_ground_truth.view(-1, 1)).float()  # (batch, k)
    idcg = torch.sum(max_r * 1. / torch.log2(torch.arange(2, k + 2, device=device)), dim=1)  # (batch, ) as a denominator
    dcg = torch.sum(pred_data * (1. / torch.log2(torch.arange(2, k + 2, device=device))), dim=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg/idcg
    batch_ndcg = ndcg.sum()
    return batch_ndcg


def test_minibatch(csr_test, csr_train, test_batch):
    num_U = len(csr_test.indptr) - 1
    for begin in range(0, num_U, test_batch):
        head = csr_test.indptr[begin: min(begin + test_batch, num_U)]
        tail = csr_test.indptr[1 + begin: 1 + begin + test_batch]
        num_pos_V = tail - head
        # print('[', begin, begin + test_batch, ']', 'pos item cnt:', num_pos_V)
        # print('sum of n_items:', num_pos_V.sum())
        ground_truth = csr_test.indices[head[0]: tail[-1]]
        
        # assert num_pos_V.sum() == len(ground_truth)  # debug
        
        # print('data:', '(', len(ground_truth), ')', ground_truth)
        
        # exclude items in training set
        head_train = csr_train.indptr[begin: min(begin + test_batch, num_U)]
        tail_train = csr_train.indptr[1 + begin: 1 + begin + test_batch]
        num_V_to_exclude = tail_train - head_train
        V_to_exclude = csr_train.indices[head_train[0]: tail_train[-1]]
        
        # assert num_V_to_exclude.sum() == len(V_to_exclude)  # debug
        
        batch_size = len(num_pos_V)
        yield np.arange(begin, begin + batch_size), num_pos_V, ground_truth, num_V_to_exclude, V_to_exclude


def get_att_dis(A, B):
    A_norm = A / torch.norm(A, dim=1, keepdim=True)
    B_norm = B / torch.norm(B, dim=1, keepdim=True)

    similarity = torch.mm(A_norm, B_norm.t())
    return similarity


def print_metrics(metrics, topk, max_K, print_max_K=True):
    if print_max_K:
        k = max_K
        # print(f'precision@{k}:', metrics[f'precision@{k}'], end='\t')
        print(f'recall@{k}:', metrics[f'recall@{k}'], end='\t')
        print(f'ndcg@{k}:', metrics[f'ndcg@{k}'])
    else:
        for i, k in enumerate(topk):
            # if i > 0:
            #     print('--')
            # print(f'precision@{k}:', metrics[f'precision@{k}'], end='\t')
            print(f'recall@{k}:', metrics[f'recall@{k}'], end='\t')
            print(f'ndcg@{k}:', metrics[f'ndcg@{k}'], end='\t')
            print()




def ranking_edges(TEST_LHS, TEST_RHS, eval_model, epoch_id):
    eval_model.eval()
    
    with torch.no_grad():
        
        STRUCTURE_CLUSTERS = eval_model.STRUCTURE_CLUSTERS.weight.detach()
        # Calculate the attention score (in nodes) for each U node to each U cluster

        U_DIS = F.cosine_similarity(TEST_LHS.unsqueeze(1), STRUCTURE_CLUSTERS.unsqueeze(0), dim=2)
        U_numerator = eval_model.U_CLUSTER_NETWORK(U_DIS)       #soft assignment
        U_soft_assignments = (U_numerator.t() / torch.sum(U_numerator, 1)).t()       #soft assignment
        # (batch_size, cluster_numbers)
        
        V_DIS = F.cosine_similarity(TEST_RHS.unsqueeze(1), STRUCTURE_CLUSTERS.unsqueeze(0), dim=2)
        V_assignments = eval_model.V_CLUSTER_NETWORK(V_DIS) # (batch_size, 1 + neg_edges, cluster_numbers)

        #########################  根据类簇计算节点间的链接概率  ####################################
        PROB_CLUSTER = torch.sum(U_soft_assignments.unsqueeze(1) * V_assignments, dim=2)


        PROB_U_V = torch.cosine_similarity(TEST_LHS.unsqueeze(1), TEST_RHS.unsqueeze(0), dim=2) + 1.0
        PROB_U_V = PROB_U_V / 2.0

        LINK_PROB = 0.5*PROB_CLUSTER + 0.5*PROB_U_V

        return LINK_PROB

        
def batch_evaluation(trained_model, csr_test, csr_train, epoch, test_batch, topk, max_K):

    trained_model.eval()

    U_EMDS = trained_model.U_emb.detach()
    V_EMDS = trained_model.V_emb.detach()
    
    num_test_U = 0
    
    metrics = {}
    for k in topk:
        metrics[f'epoch'] = epoch
        # metrics[f'precision@{k}'] = 0.
        metrics[f'recall@{k}'] = 0.
        metrics[f'ndcg@{k}'] = 0.
    
    with tqdm(total=csr_test.shape[0], desc=f'eval epoch {epoch}') as pbar:
        for i, batch in enumerate(test_minibatch(csr_test, csr_train, test_batch)):
            # print('-' * 20)
            # print('batch', i)
            idx_U, n_ground_truth, ground_truth, num_V_to_exclude, V_to_exclude = batch
            assert idx_U.shape == n_ground_truth.shape
            assert idx_U.shape == num_V_to_exclude.shape
            # print(idx_U.shape, n_ground_truth.shape, ground_truth.shape)

            batch_size = idx_U.shape[0]
            num_U_to_exclude = (n_ground_truth == 0).sum()  # exclude users that are not in test set
            # print('num_U_to_exclude:', num_U_to_exclude)
            num_test_U += batch_size - num_U_to_exclude
            
            # -> cuda 
            idx_U = torch.tensor(idx_U, dtype=torch.long, device='cpu')
            n_ground_truth = torch.tensor(n_ground_truth, dtype=torch.long, device='cpu')
            ground_truth = torch.tensor(ground_truth, dtype=torch.long, device='cpu')
            num_V_to_exclude = torch.tensor(num_V_to_exclude, dtype=torch.long, device='cpu')
            V_to_exclude = torch.tensor(V_to_exclude, dtype=torch.long, device='cpu')
            
            ########################################
            # metrics calculation
            
            with torch.no_grad():

                # rating = model.get_U_emb(idx_U) @ V_emb.transpose(0, 1)  # (batch, num_V)      # 计算batch中U对所有V的得分
                test_lhs = U_EMDS[idx_U]

                ##################################### 修改
                rating = ranking_edges(test_lhs, V_EMDS, trained_model, epoch)

            
                row_index = torch.arange(batch_size, device='cpu')  # (batch, )
                
                # filter out the items in the training set
                row_index_to_exclude = row_index.repeat_interleave(num_V_to_exclude)
                rating[row_index_to_exclude, V_to_exclude] = -1e6
                
                # pick the top max_K items
                _, rating_K = torch.topk(rating, k=max_K)  # rating_K: (batch, max_K)
                
                # build a test_graph based on ground truth coordinates
                row_index_ground_truth = row_index.repeat_interleave(n_ground_truth)
                test_g = torch.sparse_coo_tensor(indices=torch.stack((row_index_ground_truth, ground_truth), dim=0), values=torch.ones_like(ground_truth), size=(batch_size, V_EMDS.size(0)))
                
                # build a pred_graph based on top max_K predictions
                pred_row = row_index.repeat_interleave(max_K)
                pred_col = rating_K.flatten()
                pred_g = torch.sparse_coo_tensor(indices=torch.stack((pred_row, pred_col), dim=0), values=torch.ones_like(pred_col), size=(batch_size, V_EMDS.size(0)))
                
                # build a hit_graph based on the intersection of test_graph and pred_graph
                dense_g = (test_g * pred_g).coalesce().to_dense().float()

                r = dense_g[pred_row, pred_col].view(batch_size, -1)  # (batch, max_K)
            
                # recall, precision, ndcg
                for k in topk:
                    # recall, precision
                    # batch_recall, batch_precision = precision_recall(r, k, n_ground_truth)
                    batch_recall = precision_recall(r, k, n_ground_truth)
                    # ndcg
                    batch_ndcg = ndcg(r, k, n_ground_truth)
                    
                    # print(f'batch_precision@{k}:', batch_precision.item())
                    # print(f'batch_recall@{k}:', batch_recall.item())
                    # print(f'batch_ndcg@{k}:', batch_ndcg.item())
                    # print('--')
                    
                    # metrics[f'precision@{k}'] += batch_precision.item()
                    metrics[f'recall@{k}'] += batch_recall.item()
                    metrics[f'ndcg@{k}'] += batch_ndcg.item()
                    
            pbar.update(batch_size)
                
    for k in topk:
        # metrics[f'precision@{k}'] /= num_test_U
        metrics[f'recall@{k}'] /= num_test_U
        metrics[f'ndcg@{k}'] /= num_test_U
            
    return metrics


#################### 使用多层神经网络预测社区强度 ######################
class FC(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FC, self).__init__()
        self.layer = nn.Linear(in_dim, out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.layer.weight)
        nn.init.constant_(self.layer.bias, 0.0)

    def forward(self, input):
        return self.layer(input)


class MLP(nn.Module):
    def __init__(self, dims, act='relu', dropout=0.6):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(1, len(dims)):
            self.layers.append(FC(dims[i - 1], dims[i]))
        self.act = getattr(F, act)
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, input):
        curr_input = input
        for i in range(len(self.layers) - 1):
            hidden = self.layers[i](curr_input)
            hidden = self.act(hidden)
            if self.dropout:
                hidden = self.dropout(hidden)
            curr_input = hidden
        output = self.layers[-1](curr_input)
        return torch.sigmoid(output)


################# 获得初始向量 #############################
class get_ini_emds(nn.Module):
    def __init__(self, V_dim, U_dim, num_V, num_U):
        super(get_ini_emds, self).__init__()
        self.num_V = num_V
        self.num_U = num_U
        self.V_emb = nn.Embedding(num_V, V_dim)
        self.U_emb = nn.Embedding(num_U, U_dim)
        self.reset_parameters()

    def reset_parameters(self):
        # nn.init.xavier_normal_(self.V_emb.weight)
        # nn.init.xavier_normal_(self.U_emb.weight)
        nn.init.normal_(self.V_emb.weight, std=0.1)
        nn.init.normal_(self.U_emb.weight, std=0.1)

    def forward(self):
        INI_V_NODES_EMB = self.V_emb.weight.detach()
        INI_U_NODES_EMB = self.U_emb.weight.detach()
        return INI_V_NODES_EMB, INI_U_NODES_EMB


################ 先使用GAT来捕捉节点间的关系 ################################
class GAT(torch.nn.Module):
    def __init__(self, in_features, hidden_channels, out_features, heads):
        super().__init__()
        self.conv1 = GATConv(in_features, hidden_channels, heads)
        self.conv2 = GATConv(hidden_channels*heads, out_features, heads=1)

    def forward(self, x, edge_index, edge_feature):
        x = self.conv1(x, edge_index, edge_feature)
        x = F.elu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_feature)
        return x