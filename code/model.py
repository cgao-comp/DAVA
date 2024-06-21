from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from tqdm import tqdm

from collections import OrderedDict
import math
import numpy as np
import networkx as nx
import time

def binary_cross_entropy_weight(y_pred, y,has_weight=True, mask=None, weight_length=1, weight_max=10):
    if has_weight:
        if mask is not None:

            total_elements = mask.sum()
            num_ones = (y * mask).sum()
            num_zeros = total_elements - num_ones

            strong_weight = num_zeros / total_elements
            weak_weight = num_ones / total_elements

            epsilon = 1e-7
            predictions_clamped = torch.clamp(y_pred, epsilon, 1 - epsilon)

            loss = -(y * torch.log(predictions_clamped) * strong_weight + (1 - y) * torch.log(
                1 - predictions_clamped) * weak_weight) * mask

            loss = loss.sum() / total_elements
            return loss
        else:
            weight = torch.tensor([2]).cuda()
            y_pred = y_pred.reshape(-1, y_pred.shape[2])
            y = y.reshape(-1, y.shape[2])
            loss_fn = nn.BCELoss(weight=weight)
            loss = loss_fn(y_pred, y)

    else:
        loss = F.binary_cross_entropy(y_pred, y)

    return loss

class GRU_VAE_plain(nn.Module):
    def __init__(self, input_size, low_embedding_size, GRU_hidden_size, num_layers, has_input=True, has_output=False, output_size_to_edge_level=None):
        super(GRU_VAE_plain, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = GRU_hidden_size
        self.has_input = has_input
        self.has_output = has_output

        if has_input:
            self.input = nn.Linear(input_size, low_embedding_size)
            self.rnn = nn.GRU(input_size=low_embedding_size, hidden_size=GRU_hidden_size, num_layers=num_layers,
                              batch_first=True)

        if has_output:
            self.encode_exp_VAE = torch.nn.Linear(GRU_hidden_size, output_size_to_edge_level) 

            self.output = nn.Sequential(
                nn.Linear(GRU_hidden_size, low_embedding_size),
                nn.ReLU(),
                nn.Linear(low_embedding_size, output_size_to_edge_level)
            )

        self.relu = nn.ReLU()
        

        self.hidden = None  

        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0.25)
            elif 'weight' in name:
                nn.init.xavier_uniform(param,gain=nn.init.calculate_gain('sigmoid'))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))

    def init_hidden(self, batch_size, source_feature):
        if type(source_feature) == list:
            source_feature_hidden = torch.tensor(source_feature).unsqueeze(0).repeat(self.num_layers, 1)
            hidden_init = torch.zeros(self.num_layers, self.hidden_size)
            hidden_init[0:source_feature_hidden.shape[0], 0:source_feature_hidden.shape[1]] = source_feature_hidden
        else:
            source_feature_hidden = source_feature[:, 0, :].unsqueeze(0).repeat(self.num_layers, 1, 1)
            hidden_init = torch.zeros(self.num_layers, batch_size, self.hidden_size)
            hidden_init[0:source_feature_hidden.shape[0], 0:source_feature_hidden.shape[1], 0:source_feature_hidden.shape[2]] = source_feature_hidden
        return Variable(hidden_init).cuda()

    def forward(self, input_raw, pack=False, input_len=None):
        if self.has_input:
            input = self.input(input_raw)
            input = F.leaky_relu(input)
        else:
            input = input_raw
        if pack:
            input = pack_padded_sequence(input, input_len, batch_first=True)
        output_raw, self.hidden = self.rnn(input, self.hidden)
        if pack:
            output_raw = pad_packed_sequence(output_raw, batch_first=True)[0]
        if self.has_output:
            

            z_log_lambda = self.encode_exp_VAE(output_raw)
            z_lambda = torch.exp(z_log_lambda)
            eps = Variable(torch.rand(z_lambda.size())).cuda()
            z = -torch.log(1 - eps) / z_lambda

        return z, z_lambda

class SimpleAttentionModel(nn.Module):
    def __init__(self, d_query, d_key, d_value):
        super(SimpleAttentionModel, self).__init__()
        self.dmax = max(d_query, d_key, d_value)
        self.query_linear = nn.Linear(d_query, int(self.dmax/4))
        self.query_linear2 = nn.Linear(int(self.dmax/4), self.dmax)
        self.key_linear = nn.Linear(d_key, int(self.dmax/4))
        self.key_linear2 = nn.Linear(int(self.dmax/4), self.dmax)

        init.xavier_uniform_(self.query_linear.weight)
        init.xavier_uniform_(self.query_linear2.weight)
        init.xavier_uniform_(self.key_linear.weight)

    def forward(self, query, key, value):
        query_transformed = self.query_linear2(F.leaky_relu(self.query_linear(query)))
        key_transformed = self.key_linear2(F.leaky_relu(self.key_linear(key)))

        scores = torch.matmul(query_transformed, key_transformed.transpose(-2, -1))
        weights = torch.softmax(scores, dim=-1)

        weighted_value = torch.matmul(weights, value)
        return weighted_value

class MyAttentionModel(nn.Module):
    def __init__(self, d_embed, d_key, d_value, num_heads):
        super(MyAttentionModel, self).__init__()
        self.multihead_attention = nn.MultiheadAttention(embed_dim=d_embed,  num_heads=num_heads, kdim=d_key, vdim=d_value)
        self.linear = nn.Linear(d_embed, d_value)

    def forward(self, query, key, value):
        attention_output, _ = self.multihead_attention(query, key, value)
        output = self.linear(attention_output)
        

        return F.sigmoid(output)

def cosine_similarity(list1, list2):
    

    vec1 = np.array(list1)
    vec2 = np.array(list2)
    dot_product = np.dot(vec1, vec2)

    vec1_norm = np.linalg.norm(vec1)
    vec2_norm = np.linalg.norm(vec2)

    similarity = dot_product / (vec1_norm * vec2_norm)

    return similarity

def get_avg_simi(graph1: nx.Graph, graph2: nx.Graph):
    num = min(len(graph1.nodes), len(graph2.nodes))
    simi_totoal = 0
    graph1_nodes_l = list(graph1.nodes)
    graph2_nodes_l = list(graph2.nodes)
    for i in range(num):
        simi_totoal += cosine_similarity(graph1.nodes[graph1_nodes_l[i]]['type_list'], graph2.nodes[graph2_nodes_l[i]]['type_list'])
    return simi_totoal / num

def graph_kernel(graph1: nx.Graph, graph2: nx.Graph, sigma: float) -> float:
    """
    计算两个图之间的核函数值（这里使用高斯核函数）。
    :param graph1: networkx图对象
    :param graph2: networkx图对象
    :param sigma: 高斯核函数的参数
    :return: 核函数值
    Warnings: large scale network results in maximum recursion depth exceeded in comparison. sys.setrecursionlimit(5000) OR nx.graph_edit_distance() with high complexity!
    """
    ged_generator = nx.optimize_graph_edit_distance(graph1, graph2)
    ged_puni = (next(ged_generator))

    epsilon = 1e-7
    cos_puni = 1 - get_avg_simi(graph1, graph2) + epsilon

    

    

    return np.exp(-((cos_puni*ged_puni) ** 2) / (2 * sigma ** 2))
def calculate_mmd(graphs1: list, graphs2: list, sigma: float) -> float:
    m = len(graphs1)
    n = len(graphs2)
    
    mean_within_group1 = sum(graph_kernel(graphs1[i], graphs1[j], sigma) for i in range(m) for j in range(m)) / (m * m)
    mean_within_group2 = sum(graph_kernel(graphs2[i], graphs2[j], sigma) for i in range(n) for j in range(n)) / (n * n)
    
    mean_between_groups = sum(graph_kernel(graphs1[i], graphs2[j], sigma) for i in range(m) for j in range(n)) / (m * n)
    
    mmd = mean_within_group1 + mean_within_group2 - 2 * mean_between_groups
    return mmd

def calculate_mmd_tqdm(graphs1: list, graphs2: list, sigma: float) -> float:
    m = len(graphs1)
    n = len(graphs2)

    
    total_inner_comparisons = m * m + n * n
    inner_counter = 0

    mean_within_group1 = 0
    success_time = 0
    for i in tqdm(range(m)):
        for j in range(m):
            inner_counter += 1
            mean_within_group1_part = graph_kernel(graphs1[i], graphs1[j], sigma)
            mean_within_group1 += mean_within_group1_part
            success_time += 1
            
    mean_within_group1 /= success_time

    success_time = 0
    mean_within_group2 = 0
    for i in tqdm(range(n)):
        for j in range(n):
            inner_counter += 1
            mean_within_group2_part = graph_kernel(graphs2[i], graphs2[j], sigma)
            mean_within_group2 += mean_within_group2_part
            success_time += 1

    mean_within_group2 /= success_time
    total_between_comparisons = m * n
    between_counter = 0
    success_time = 0
    mean_between_groups = 0
    for i in tqdm(range(m)):
        for j in range(n):
            between_counter += 1
            mean_between_groups_part = graph_kernel(graphs1[i], graphs2[j], sigma)
            mean_between_groups += mean_between_groups_part
            success_time += 1
    mean_between_groups /= success_time
    mmd = mean_within_group1 + mean_within_group2 - 2 * mean_between_groups
    return mmd

import pickle
if __name__ == '__main__':
    fname = './graphs/SimpleAtt_createbig_kl_exp10_weight_fenzifenmu1/GraphRNN_VAE_Twitter_2_64_pred_200_gene10features2.pkl'
    with open(fname, "rb") as f:
        G_read_pickle201 = pickle.load(f)

    

