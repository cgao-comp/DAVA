import torch
import torchvision as tv
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
from random import shuffle
from tqdm import tqdm

import networkx as nx
import pickle as pkl
import scipy.sparse as sp
import logging
import args

import random
import shutil
import os
import time
from model import *
from utils import *


from collections import deque
def bfs_sorted_with_type(graph, root):
    visited = set()  

    cur_level = deque([root])  

    bfs_sequence = []  


    while cur_level:
        next_level = deque()  

        sorted_cur_level = sorted(cur_level, key=lambda node: graph.nodes[node]['type'], reverse=True)  


        for node in sorted_cur_level:
            if node not in visited:  

                visited.add(node)  

                bfs_sequence.append(node)  

                for neighbor in graph.neighbors(node):  

                    if neighbor not in visited:  

                        next_level.append(neighbor)  


        cur_level = next_level
    

    


    return bfs_sequence





class Graph_DataProcessing(torch.utils.data.Dataset):
    def __init__(self, G_list, max_num_node=None, max_prev_node=None, iteration=20000):
        self.adj_all = []
        self.len_all = []
        self.user_feature = []
        for DAG in tqdm(G_list):
            uid_l = [node for node, in_degree in DAG.in_degree() if in_degree == 0]
            assert len(uid_l) == 1, '数据错误'
            uid = uid_l[0]
            bfs_sequence = bfs_sorted_with_type(DAG, uid)
            node_creation_order = {}
            idx = 0
            for one_node in DAG.nodes:
                node_creation_order[one_node] = idx
                idx += 1
            bfs_sequence_creation_order = [node_creation_order[node] for node in bfs_sequence]
            bfs_sequence_features = [DAG.nodes.get(node)['type_list'] for node in bfs_sequence]

            adj_DAG = np.asarray(nx.to_numpy_matrix(DAG))
            adj_DAG = adj_DAG[np.ix_(bfs_sequence_creation_order, bfs_sequence_creation_order)]

            self.adj_all.append(adj_DAG)
            self.len_all.append(DAG.number_of_nodes())
            self.user_feature.append(bfs_sequence_features)
        if max_num_node is None:
            self.n = max(self.len_all)
        else:
            self.n = max_num_node
        if max_prev_node is None:
            self.max_prev_node = max_num_node

    def get_max_prev_node(self):
        return self.max_prev_node


    def __len__(self):
        return len(self.adj_all)
    def __getitem__(self, idx):
        adj_encoded = self.adj_all[idx].copy()
        len_batch = adj_encoded.shape[0]
        adj_encoded = adj_encoded.T
        adj_encoded = adj_encoded[1:len_batch, 0:len_batch - 1]
        x_batch = np.zeros((self.n, self.max_prev_node))  

        x_batch[0,:] = 1 

        y_batch = np.zeros((self.n, self.max_prev_node))  


        def pad_columns(adj_encoded, y_batch):
            rows, columns = adj_encoded.shape
            columns_to_pad = y_batch.shape[1] - columns
            padding_matrix = np.zeros((rows, columns_to_pad))
            padded_adj_encoded = np.concatenate((adj_encoded, padding_matrix), axis=1)
            return padded_adj_encoded

        padded_adj_encoded = pad_columns(adj_encoded, y_batch)
        y_batch[0:adj_encoded.shape[0], :] = padded_adj_encoded
        x_batch[1:adj_encoded.shape[0] + 1, :] = padded_adj_encoded

        feature_batch = self.user_feature[idx].copy()
        zero_padding = [[-1, -1, -1, -1, -1, -1] for _ in range(self.n-len_batch)]
        feature_batch.extend(zero_padding)
        feature_batch = np.array(feature_batch)
        

        return {'x': x_batch, 'y': y_batch, 'len': len_batch, 'user_feature': feature_batch}
