import networkx
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.decomposition import PCA
import logging
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from time import gmtime, strftime
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from random import shuffle
import pickle
from tensorboard_logger import configure, log_value
import scipy.misc
import time as tm

from utils import *
from model import *
from data import *
from args import Args
from tqdm import tqdm

import create_graphs
import main
import sys

import gc
def train_rnn_epoch(epoch, args, rnn, output, data_loader,
                    optimizer_rnn, optimizer_output,
                    scheduler_rnn, scheduler_output):
    print('main.target_lambda:', main.target_lambda)
    rnn.train()
    output.train()
    loss_sum = 0
    for batch_idx, data in enumerate(data_loader):
        
        
        rnn.zero_grad()
        output.zero_grad()
        user_feature_unsorted = data['user_feature'].float()
        x_unsorted = data['x'].float()
        y_unsorted = data['y'].float()
        y_len_unsorted = data['len']
        y_len_max = max(y_len_unsorted)
        x_unsorted = x_unsorted[:, 0:y_len_max, :]  
        y_unsorted = y_unsorted[:, 0:y_len_max, :]
        y_len,sort_index = torch.sort(y_len_unsorted, 0, descending=True)
        y_len = y_len.numpy().tolist()
        x = torch.index_select(x_unsorted, 0, sort_index) 
        y = torch.index_select(y_unsorted, 0, sort_index)       
        user_feature = torch.index_select(user_feature_unsorted, 0, sort_index)

        
        rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0), source_feature=user_feature)

        x = Variable(x).cuda()
        y = Variable(y).cuda()
        user_feature = Variable(user_feature).cuda()

        x = x[:, :, 0:(x.shape[2] - 1)]
        h, z_lambda = rnn(x, pack=True, input_len=y_len)
        
        output_y_len = []
        for len in y_len:
            for i in range(1, len):
                output_y_len.append(i)
            output_y_len.append(-999)
        y = y[:, 0:(y.shape[1]-1), 0:(y.shape[2]-1)]      
 
        attn_res = torch.zeros(y.shape[0], y.shape[1], y.shape[2]).cuda()
        mask = torch.zeros_like(attn_res).cuda()

        batch_belong = 0
        gc.collect()
        for len in output_y_len:
            if len == -999:
                batch_belong = batch_belong + 1
                continue
            key = user_feature[batch_belong, 0:len, :] 
            query = user_feature[batch_belong, len, :] 
            query = query.unsqueeze(0)

            output_tensor = torch.zeros(len * args.max_prev_node)
            identity_matrix = torch.eye(len)
            identity_matrix_flat = identity_matrix.view(-1)
            for i in range(len):
                output_tensor[i * (args.max_prev_node+1)] = identity_matrix_flat[i * (len + 1)]
            value = output_tensor.view(len, args.max_prev_node)
            value = value[:, 0:(value.shape[1]-1)]
            value = Variable(value).cuda()

            h_graph = h[batch_belong, len-1, :]
            h_graph = h_graph.unsqueeze(0)
            
            
            query_ = torch.cat((query, h_graph), dim=1)

            attn_res[batch_belong, len-1, :] = output(query_, key, value) 
            mask[batch_belong, len-1, :] = torch.tensor(1).cuda()

        
        loss_bce = binary_cross_entropy_weight(attn_res, y, mask=mask)
        kl_loss = torch.log(z_lambda / main.target_lambda) + (main.target_lambda / z_lambda) - 1
        kl_loss = torch.sum(kl_loss)
        kl_loss = kl_loss/ (z_lambda.size(0)*z_lambda.size(1)*z_lambda.size(2))
        print('kl_loss:', kl_loss)

        if epoch <= 5:
            loss = loss_bce
            if epoch == 5:
                main.target_lambda = kl_loss.item()
        else:
            loss = loss_bce + kl_loss

        loss.backward()
        
        optimizer_output.step()
        optimizer_rnn.step()
        scheduler_output.step()
        scheduler_rnn.step()

        if (epoch % args.epochs_log==0 and batch_idx==0) or True: 
            print('Epoch: {}/{}, train loss: {:.6f}, graph type: {}, num_layer: {}, hidden: {}'.format(
                epoch, args.epochs,loss.item(), args.graph_type, args.num_layers, args.hidden_size_rnn))

        
        log_value('loss_'+args.fname, loss.item(), epoch*args.batch_ratio+batch_idx)
        feature_dim = y.size(1)*y.size(2)
        loss_sum += loss.item()*feature_dim
    return loss_sum/(batch_idx+1)

def test_rnn_epoch_largeScale(epoch, args, rnn, output, gene_num, test_batch_size, union_graph_ori):
    
    union_graph = union_graph_ori.copy()
    rnn.eval()
    output.eval()
    
    user_info_dict = {}
    decode_adj = []
    source_idx = random.choice(list(union_graph.nodes()))
    user_visited = {}
    user_visited[source_idx] = 0
    source_node = union_graph.nodes.get(source_idx)
    user_info_dict[len(user_visited)-1] = {'uid': source_idx, 'type_list': source_node['type_list'], 'att': source_node['att']}

    rnn.hidden = rnn.init_hidden(test_batch_size, source_node['type_list'])
    rnn_step = Variable(torch.ones(1, args.exceed-1)).cuda()
    for len1 in range(1, gene_num):
        h = None
        if len1 >= args.max_num_node:
            rnn_step = rnn_step[0, len1 - args.max_num_node + 1:len1].unsqueeze(0)
        else:
            rnn_step = rnn_step[0, 0: args.max_num_node-1].unsqueeze(0)

        h, _ = rnn(rnn_step)

        def unvisted_nei(user_visited_dict, cand):
            unvisted_ = []
            for last_node_nei in union_graph.neighbors(cand):
                if last_node_nei not in user_visited_dict.keys():
                    unvisted_.append(last_node_nei)
            return unvisted_
        unvisted_global_list = unvisted_nei(user_visited, list(user_visited.keys())[-1])
        length__ = len(unvisted_global_list)

        random_idx = -111
        unexpect_num = 10
        valid_nei = None
        meaning_idx = None
        meaning = None
        if length__ == 0:
            random_idx = random.choice(list(union_graph.nodes()))
            while True:
                if random_idx in list(user_visited.keys()) or len(unvisted_nei(user_visited, random_idx))==0:
                    random_idx = random.choice(list(union_graph.nodes()))
                else:
                    break
            valid_nei = [random_idx]
            meaning_idx = torch.empty((1, 1), dtype=torch.int64)
            meaning = torch.zeros(1, len(user_visited))
        elif length__ > unexpect_num:
            
            def get_valid_nei(nei_iters, out_num=20):
                assert out_num % 2 == 0, 'invalid!!!'
                use_out_num = int(out_num/2)
                nei_dict = {}
                for nei in nei_iters:
                    nei_dict[nei] = union_graph.nodes[nei]['type']
                sorted_dict = sorted(nei_dict.items(), key=lambda x: x[1], reverse=True)
                top_num_keys1 = [item[0] for item in sorted_dict[:use_out_num]]
                top_num_keys2 = [item[0] for item in sorted_dict[-use_out_num:]]
                return (top_num_keys1 + top_num_keys2)
            valid_nei = get_valid_nei(unvisted_global_list, out_num=unexpect_num)
            
            meaning_idx = torch.empty((len(valid_nei), 1), dtype=torch.int64)
            meaning = torch.zeros(len(valid_nei), len(user_visited))
        else:
            valid_nei = unvisted_global_list
            meaning_idx = torch.empty((len(valid_nei), 1), dtype=torch.int64)
            meaning = torch.zeros(len(valid_nei), len(user_visited))

        key_list = []
        assert len(user_visited) == len1, 'error!'
        mask_ = torch.ones(len(user_visited))
        decay_coff = 0.99
        for index, (cur_idx, decay_weight) in enumerate(user_visited.items()):
            cur_node = union_graph.nodes.get(cur_idx)
            key_list.append(cur_node['type_list'])
            mask_[index] = 1 * torch.pow(torch.tensor(decay_coff), decay_weight)
        key = torch.tensor(key_list).float().cuda()
        mask_ = mask_.cuda()

        value = None
        if args.fixed_num is not True:
  
            value = Variable(torch.eye(len1)).cuda()
        else:
            output_tensor = torch.zeros(len1 * args.max_prev_node)
            identity_matrix = torch.eye(len1)
            identity_matrix_flat = identity_matrix.view(-1)
            for i in range(len1):
                output_tensor[i * (args.max_prev_node + 1)] = identity_matrix_flat[i * (len1 + 1)]
            value = output_tensor.view(len1, args.max_prev_node)  
            value = value[:, 0:(value.shape[1] - 1)]
            value = Variable(value).cuda()

        
        for index, nei_idx in enumerate(valid_nei):
            meaning_idx[index, 0] = nei_idx
            nei_node = union_graph.nodes.get(nei_idx)
            query_nei = torch.tensor(nei_node['type_list']).unsqueeze(0).cuda()
            query_nei_ = torch.cat((query_nei, h), dim=1)

            edge_prob = output(query_nei_, key, value)
            edge_prob_exist = edge_prob[:, 0:len(user_visited)]
            meaning[index, 0:] = edge_prob_exist * mask_
        

        max_row_value, _ = torch.max(meaning[:, 0:], dim=1)
        max_indices = torch.argmax(max_row_value)
        max_id = meaning_idx[:, 0][max_indices]
        new_id = max_id.item()

        if length__ > 0:
            union_graph.remove_edge(new_id, list(user_visited.keys())[-1])  
        else:
            if union_graph.has_edge(new_id, list(user_visited.keys())[-1]):
                union_graph.remove_edge(new_id, list(user_visited.keys())[-1])
        user_visited[new_id] = 0
        user_info_dict[len(user_visited)-1] = {'uid': source_idx, 'type_list': union_graph.nodes.get(new_id)['type_list'], 'att': union_graph.nodes.get(new_id)['att']}

        max_linear_index = torch.argmax(meaning)
        max_row_index = max_linear_index // meaning.shape[1]
        max_col_index = max_linear_index % meaning.shape[1]
        rnn.hidden = Variable(rnn.hidden.data).cuda()
        rnn_step = Variable(torch.zeros(1, args.exceed - 1))
        rnn_step[0, max_col_index] = 1

        pre_node = list(user_visited.keys())[max_col_index]
        user_visited[pre_node] += 1  

        rnn_list_step = rnn_step.tolist()
        decode_adj.append(rnn_list_step[0][0:gene_num-1])

        rnn_step = rnn_step.cuda()
    adj_full = np.zeros((gene_num, gene_num))
    n = adj_full.shape[0]
    decode_adj_ = np.array(decode_adj)
    adj_full[1:n, 0:n - 1] = decode_adj_
    adj_full = adj_full + adj_full.T

    adj_full = adj_full[~np.all(adj_full == 0, axis=1)]
    adj_full = adj_full[:, ~np.all(adj_full == 0, axis=0)]
    upper_triangular = np.triu(adj_full)

    one_DAG = np.asmatrix(upper_triangular)
    G = nx.from_numpy_matrix(one_DAG, create_using=nx.DiGraph())
    nx.set_node_attributes(G, user_info_dict)
    return adj_full, upper_triangular, G

def test_rnn_epoch(epoch, args, rnn, output, gene_num, test_batch_size, union_graph_ori):
    
    union_graph = union_graph_ori.copy()
    rnn.eval()
    output.eval()
    
    user_info_dict = {}
    decode_adj = []
    source_idx = random.choice(list(union_graph.nodes()))
    user_visited = {}
    user_visited[source_idx] = 0
    source_node = union_graph.nodes.get(source_idx)
    user_info_dict[len(user_visited)-1] = {'uid': source_idx, 'type_list': source_node['type_list'], 'att': source_node['att']}

    rnn.hidden = rnn.init_hidden(test_batch_size, source_node['type_list'])
    rnn_step = Variable(torch.ones(1, args.max_num_node-1)).cuda()
    for len1 in range(1, gene_num):
        h, _ = rnn(rnn_step)
        def unvisted_nei(user_visited_dict, cand):
            unvisted_ = []
            for last_node_nei in union_graph.neighbors(cand):
                if last_node_nei not in user_visited_dict.keys():
                    unvisted_.append(last_node_nei)
            return unvisted_
        unvisted_global_list = unvisted_nei(user_visited, list(user_visited.keys())[-1])
        length__ = len(unvisted_global_list)

        random_idx = -111
        unexpect_num = 10
        valid_nei = None
        meaning_idx = None
        meaning = None
        if length__ == 0:
            random_idx = random.choice(list(union_graph.nodes()))
            while True:
                if random_idx in list(user_visited.keys()) or len(unvisted_nei(user_visited, random_idx))==0:
                    random_idx = random.choice(list(union_graph.nodes()))
                else:
                    break
            valid_nei = [random_idx]
            meaning_idx = torch.empty((1, 1), dtype=torch.int64)
            meaning = torch.zeros(1, len(user_visited))
        elif length__ > unexpect_num:
            
            def get_valid_nei(nei_iters, out_num=20):
                assert out_num % 2 == 0, 'invalid!!!'
                use_out_num = int(out_num/2)
                nei_dict = {}
                for nei in nei_iters:
                    nei_dict[nei] = union_graph.nodes[nei]['type']
                sorted_dict = sorted(nei_dict.items(), key=lambda x: x[1], reverse=True)
                top_num_keys1 = [item[0] for item in sorted_dict[:use_out_num]]
                top_num_keys2 = [item[0] for item in sorted_dict[-use_out_num:]]
                return (top_num_keys1 + top_num_keys2)
            valid_nei = get_valid_nei(unvisted_global_list, out_num=unexpect_num)
            
            meaning_idx = torch.empty((len(valid_nei), 1), dtype=torch.int64)
            meaning = torch.zeros(len(valid_nei), len(user_visited))
        else:
            valid_nei = unvisted_global_list
            meaning_idx = torch.empty((len(valid_nei), 1), dtype=torch.int64)
            meaning = torch.zeros(len(valid_nei), len(user_visited))

        key_list = []
        assert len(user_visited) == len1, 'error!'
        mask_ = torch.ones(len(user_visited))
        decay_coff = 0.99
        for index, (cur_idx, decay_weight) in enumerate(user_visited.items()):
            cur_node = union_graph.nodes.get(cur_idx)
            key_list.append(cur_node['type_list'])
            mask_[index] = 1 * torch.pow(torch.tensor(decay_coff), decay_weight)
        key = torch.tensor(key_list).float().cuda()
        mask_ = mask_.cuda()

        value = None
        if args.fixed_num is not True:
            output_tensor = torch.zeros(len1 * gene_num)
            identity_matrix = torch.eye(len1)
            identity_matrix_flat = identity_matrix.view(-1)
            for i in range(len1):
                output_tensor[i * (gene_num + 1)] = identity_matrix_flat[i * (len1 + 1)]
            value = output_tensor.view(len1, gene_num)
            value = value[:, 0:(value.shape[1] - 1)]
            value = Variable(value).cuda()
        else:
            output_tensor = torch.zeros(len1 * args.max_prev_node)
            identity_matrix = torch.eye(len1)
            identity_matrix_flat = identity_matrix.view(-1)
            for i in range(len1):
                output_tensor[i * (args.max_prev_node + 1)] = identity_matrix_flat[i * (len1 + 1)]
            value = output_tensor.view(len1, args.max_prev_node)  
            value = value[:, 0:(value.shape[1] - 1)]
            value = Variable(value).cuda()

        
        for index, nei_idx in enumerate(valid_nei):
            meaning_idx[index, 0] = nei_idx
            nei_node = union_graph.nodes.get(nei_idx)
            query_nei = torch.tensor(nei_node['type_list']).unsqueeze(0).cuda()
            query_nei_ = torch.cat((query_nei, h), dim=1)

            edge_prob = output(query_nei_, key, value)
            edge_prob_exist = edge_prob[:, 0:len(user_visited)]
            meaning[index, 0:] = edge_prob_exist * mask_
        

        max_row_value, _ = torch.max(meaning[:, 0:], dim=1)
        max_indices = torch.argmax(max_row_value)
        max_id = meaning_idx[:, 0][max_indices]
        new_id = max_id.item()

        if length__ > 0:
            union_graph.remove_edge(new_id, list(user_visited.keys())[-1])  
        else:
            if union_graph.has_edge(new_id, list(user_visited.keys())[-1]):
                union_graph.remove_edge(new_id, list(user_visited.keys())[-1])
        user_visited[new_id] = 0
        user_info_dict[len(user_visited)-1] = {'uid': source_idx, 'type_list': union_graph.nodes.get(new_id)['type_list'], 'att': union_graph.nodes.get(new_id)['att']}

        max_linear_index = torch.argmax(meaning)
        max_row_index = max_linear_index // meaning.shape[1]
        max_col_index = max_linear_index % meaning.shape[1]
        rnn.hidden = Variable(rnn.hidden.data).cuda()
        rnn_step = Variable(torch.zeros(1, args.max_num_node - 1))
        rnn_step[0, max_col_index] = 1

        pre_node = list(user_visited.keys())[max_col_index]
        user_visited[pre_node] += 1  

        rnn_list_step = rnn_step.tolist()
        decode_adj.append(rnn_list_step[0][0:gene_num-1])

        rnn_step = rnn_step.cuda()
    adj_full = np.zeros((gene_num, gene_num))
    n = adj_full.shape[0]
    decode_adj_ = np.array(decode_adj)
    adj_full[1:n, 0:n - 1] = decode_adj_
    adj_full = adj_full + adj_full.T

    adj_full = adj_full[~np.all(adj_full == 0, axis=1)]
    adj_full = adj_full[:, ~np.all(adj_full == 0, axis=0)]
    upper_triangular = np.triu(adj_full)

    one_DAG = np.asmatrix(upper_triangular)
    G = nx.from_numpy_matrix(one_DAG, create_using=nx.DiGraph())
    nx.set_node_attributes(G, user_info_dict)
    return adj_full, upper_triangular, G

from copy import deepcopy
def test_rnn_epoch_decay_correct(epoch, args, rnn, output, test_batch_size, union_graph_ori):
    
    union_graph = union_graph_ori.copy()
    rnn.eval()
    output.eval()
    
    decode_adj = []
    source_idx = random.choice(list(union_graph.nodes()))
    source_node = union_graph.nodes.get(source_idx)
    rnn.hidden = rnn.init_hidden(test_batch_size, source_node['type_list'])
    rnn_step = Variable(torch.ones(1, args.max_prev_node-1)).cuda() 
    user_visited = {}
    user_visited[source_idx] = 0
    for len1 in range(1, args.max_prev_node):
        h, _ = rnn(rnn_step)
        def unvisted_nei(user_visited_dict, cand):
            unvisted_ = []
            for last_node_nei in union_graph.neighbors(cand):
                if last_node_nei not in user_visited_dict.keys():
                    unvisted_.append(last_node_nei)
            return unvisted_
        unvisted_global_list = unvisted_nei(user_visited, list(user_visited.keys())[-1])
        length__ = len(unvisted_global_list)

        random_idx = -111
        unexpect_num = 10
        valid_nei = None
        meaning_idx = None
        meaning = None
        if length__ == 0:
            random_idx = random.choice(list(union_graph.nodes()))
            while True:
                if random_idx in list(user_visited.keys()) or len(unvisted_nei(user_visited, random_idx))==0:
                    random_idx = random.choice(list(union_graph.nodes()))
                else:
                    break
            valid_nei = [random_idx]
            meaning_idx = torch.empty((1, 1), dtype=torch.int64)
            meaning = torch.zeros(1, len(user_visited))
        elif length__ > unexpect_num:
            
            def get_valid_nei(nei_iters, out_num=20):
                nei_dict = {}
                for nei in nei_iters:
                    nei_dict[nei] = union_graph.nodes[nei]['type']
                sorted_dict = sorted(nei_dict.items(), key=lambda x: x[1], reverse=True)
                top_num_keys = [item[0] for item in sorted_dict[:out_num]]
                return top_num_keys
            valid_nei = get_valid_nei(unvisted_global_list, out_num=unexpect_num)
            
            meaning_idx = torch.empty((len(valid_nei), 1), dtype=torch.int64)
            meaning = torch.zeros(len(valid_nei), len(user_visited))
        else:
            valid_nei = unvisted_global_list
            meaning_idx = torch.empty((len(valid_nei), 1), dtype=torch.int64)
            meaning = torch.zeros(len(valid_nei), len(user_visited))

        key_list = []
        assert len(user_visited) == len1, 'error!'
        mask_ = torch.ones(len(user_visited))
        decay_coff = 0.99
        for index, (cur_idx, decay_weight) in enumerate(user_visited.items()):
            cur_node = union_graph.nodes.get(cur_idx)
            key_list.append(cur_node['type_list'])
            mask_[index] = 1 * torch.pow(torch.tensor(decay_coff), decay_weight)
        key = torch.tensor(key_list).float().cuda()
        mask_ = mask_.cuda()

        output_tensor = torch.zeros(len1 * args.max_prev_node)
        identity_matrix = torch.eye(len1)
        identity_matrix_flat = identity_matrix.view(-1)
        for i in range(len1):
            output_tensor[i * (args.max_prev_node + 1)] = identity_matrix_flat[i * (len1 + 1)]
        value = output_tensor.view(len1, args.max_prev_node)
        value = value[:, 0:(value.shape[1] - 1)]
        value = Variable(value).cuda()

        
        for index, nei_idx in enumerate(valid_nei):
            meaning_idx[index, 0] = nei_idx
            nei_node = union_graph.nodes.get(nei_idx)
            query_nei = torch.tensor(nei_node['type_list']).unsqueeze(0).cuda()
            query_nei_ = torch.cat((query_nei, h), dim=1)

            edge_prob = output(query_nei_, key, value)
            edge_prob_exist = edge_prob[:, 0:len(user_visited)]
            meaning[index, 0:] = edge_prob_exist * mask_
        

        max_row_value, _ = torch.max(meaning[:, 0:], dim=1)
        max_indices = torch.argmax(max_row_value)
        max_id = meaning_idx[:, 0][max_indices]
        new_id = max_id.item()

        if length__ > 0:
            union_graph.remove_edge(new_id, list(user_visited.keys())[-1])  
        else:
            if union_graph.has_edge(new_id, list(user_visited.keys())[-1]):
                union_graph.remove_edge(new_id, list(user_visited.keys())[-1])
        user_visited[new_id] = 0

        max_linear_index = torch.argmax(meaning)
        max_row_index = max_linear_index // meaning.shape[1]
        max_col_index = max_linear_index % meaning.shape[1]
        rnn.hidden = Variable(rnn.hidden.data).cuda()
        rnn_step = Variable(torch.zeros(1, args.max_prev_node - 1))
        rnn_step[0, max_col_index] = 1

        pre_node = list(user_visited.keys())[max_col_index]
        user_visited[pre_node] += 1  

        rnn_list_step = rnn_step.tolist()
        decode_adj.append(rnn_list_step[0])

        rnn_step = rnn_step.cuda()
    adj_full = np.zeros((args.max_num_node, args.max_num_node))
    n = adj_full.shape[0]
    decode_adj_ = np.array(decode_adj)
    adj_full[1:n, 0:n - 1] = decode_adj_
    adj_full = adj_full + adj_full.T

    adj_full = adj_full[~np.all(adj_full == 0, axis=1)]
    adj_full = adj_full[:, ~np.all(adj_full == 0, axis=0)]
    upper_triangular = np.triu(adj_full)

    one_DAG = np.asmatrix(upper_triangular)
    G = nx.from_numpy_matrix(one_DAG, create_using=nx.DiGraph())
    return adj_full, upper_triangular, G

def test_rnn_epoch_ori(epoch, args, rnn, output, test_batch_size, union_graph):
    rnn.eval()
    output.eval()
    
    decode_adj = []
    source_idx = random.choice(list(union_graph.nodes()))
    source_node = union_graph.nodes.get(source_idx)
    rnn.hidden = rnn.init_hidden(test_batch_size, source_node['type_list'])
    rnn_step = Variable(torch.ones(1, args.max_prev_node-1)).cuda()
    user_visited = []
    user_visited.append(source_idx)
    for len1 in range(1, args.max_num_node):
        h,_ = rnn(rnn_step)
        meaning_idx = torch.empty((len(list(union_graph.neighbors(user_visited[-1]))), 1), dtype=torch.int64)
        meaning = torch.zeros(len(list(union_graph.neighbors(user_visited[-1]))), len(user_visited))
        
        for index, nei_idx in enumerate(union_graph.neighbors(user_visited[-1])):
            meaning_idx[index, 0] = nei_idx
            nei_node = union_graph.nodes.get(nei_idx)
            query_nei = torch.tensor(nei_node['type_list']).unsqueeze(0).cuda()
            query_nei_ = torch.cat((query_nei, h), dim=1)

            output_tensor = torch.zeros(len1 * args.max_prev_node)
            identity_matrix = torch.eye(len1)
            identity_matrix_flat = identity_matrix.view(-1)
            for i in range(len1):
                output_tensor[i * (args.max_prev_node+1)] = identity_matrix_flat[i * (len1 + 1)]
            value = output_tensor.view(len1, args.max_prev_node)
            value = value[:, 0:(value.shape[1]-1)]
            value = Variable(value).cuda()

            key_list = []
            assert len(user_visited) == len1, 'error!'
            for cur_idx in user_visited:
                cur_node = union_graph.nodes.get(cur_idx)
                key_list.append(cur_node['type_list'])
            key = torch.tensor(key_list).float().cuda()

            edge_prob = output(query_nei_, key, value)
            edge_prob_exist = edge_prob[:, 0:len(user_visited)]
            meaning[index, 0:] = edge_prob_exist
        max_row_value, _ = torch.max(meaning[:, 0:], dim=1)
        max_indices = torch.argmax(max_row_value)
        max_id = meaning_idx[:, 0][max_indices]
        new_id = max_id.item()
        user_visited.append(new_id)

        max_linear_index = torch.argmax(meaning)
        max_row_index = max_linear_index // meaning.shape[1]
        max_col_index = max_linear_index % meaning.shape[1]
        rnn.hidden = Variable(rnn.hidden.data).cuda()
        rnn_step = Variable(torch.zeros(1, args.max_prev_node - 1))
        rnn_step[0, max_col_index] = 1

        rnn_list_step = rnn_step.tolist()
        decode_adj.append(rnn_list_step[0])

        rnn_step = rnn_step.cuda()
    adj_full = np.zeros((args.max_num_node, args.max_num_node))
    n = adj_full.shape[0]
    decode_adj_ = np.array(decode_adj)
    adj_full[1:n, 0:n - 1] = decode_adj_
    adj_full = adj_full + adj_full.T

    adj_full = adj_full[~np.all(adj_full == 0, axis=1)]
    adj_full = adj_full[:, ~np.all(adj_full == 0, axis=0)]
    upper_triangular = np.triu(adj_full)

    one_DAG = np.asmatrix(upper_triangular)
    G = nx.from_numpy_matrix(one_DAG, create_using=nx.DiGraph())
    return adj_full, upper_triangular, G

def test_rnn_epoch_debugger(epoch, args, rnn, output, test_batch_size, union_graph):
    fname = args.model_save_path + args.fname + 'GRU_VAE_' + str(100) + '.dat'
    rnn.load_state_dict(torch.load(fname))
    fname = args.model_save_path + args.fname + 'edgeAttention_' + str(100) + '.dat'
    output.load_state_dict(torch.load(fname))

    rnn.eval()
    output.eval()
    
    decode_adj = []
    source_idx = random.choice(list(union_graph.nodes()))
    source_node = union_graph.nodes.get(source_idx)
    rnn.hidden = rnn.init_hidden(test_batch_size, source_node['type_list'])
    rnn_step = Variable(torch.ones(1, args.max_prev_node - 1)).cuda()
    user_visited = []
    user_visited.append(source_idx)
    for len1 in range(1, args.max_num_node):
        h, _ = rnn(rnn_step)
        meaning_idx = torch.empty((len(list(union_graph.neighbors(user_visited[-1]))), 1), dtype=torch.int64)
        meaning = torch.zeros(len(list(union_graph.neighbors(user_visited[-1]))), len(user_visited))
        
        for index, nei_idx in enumerate(union_graph.neighbors(user_visited[-1])):
            meaning_idx[index, 0] = nei_idx
            nei_node = union_graph.nodes.get(nei_idx)
            query_nei = torch.tensor(nei_node['type_list']).unsqueeze(0).cuda()
            query_nei_ = torch.cat((query_nei, h), dim=1)

            output_tensor = torch.zeros(len1 * args.max_prev_node)
            identity_matrix = torch.eye(len1)
            identity_matrix_flat = identity_matrix.view(-1)
            for i in range(len1):
                output_tensor[i * (args.max_prev_node + 1)] = identity_matrix_flat[i * (len1 + 1)]
            value = output_tensor.view(len1, args.max_prev_node)
            value = value[:, 0:(value.shape[1] - 1)]
            value = Variable(value).cuda()

            key_list = []
            assert len(user_visited) == len1, 'error!'
            for cur_idx in user_visited:
                cur_node = union_graph.nodes.get(cur_idx)
                key_list.append(cur_node['type_list'])
            key = torch.tensor(key_list).float().cuda()

            edge_prob = output(query_nei_, key, value)
            edge_prob_exist = edge_prob[:, 0:len(user_visited)]
            meaning[index, 0:] = edge_prob_exist
        print('----------------------------------------')
        print(meaning)
        print('----------------------------------------')
        max_row_value, _ = torch.max(meaning[:, 0:], dim=1)
        max_indices = torch.argmax(max_row_value)
        max_id = meaning_idx[:, 0][max_indices]
        new_id = max_id.item()
        user_visited.append(new_id)

        rnn.hidden = Variable(rnn.hidden.data).cuda()
        rnn_step = Variable(torch.zeros(1, args.max_prev_node - 1))
        rnn_step[0, max_indices] = 1

        rnn_list_step = rnn_step.tolist()
        decode_adj.append(rnn_list_step[0])

        rnn_step = rnn_step.cuda()
    adj_full = np.zeros((args.max_num_node, args.max_num_node))
    n = adj_full.shape[0]
    decode_adj_ = np.array(decode_adj)
    adj_full[1:n, 0:n - 1] = decode_adj_
    adj_full = adj_full + adj_full.T

    adj_full = adj_full[~np.all(adj_full == 0, axis=1)]
    adj_full = adj_full[:, ~np.all(adj_full == 0, axis=0)]
    upper_triangular = np.triu(adj_full)

    one_DAG = np.asmatrix(upper_triangular)
    G = nx.from_numpy_matrix(one_DAG, create_using=nx.DiGraph())
    return adj_full, upper_triangular, G

def train(args, dataset_train, rnn, output, union_graph_type, graphs_list):
    
    
    if args.load:
        fname = args.model_save_path + args.fname + 'GRU_VAE_' + str(args.load_epoch) + '.dat'
        rnn.load_state_dict(torch.load(fname))
        fname = args.model_save_path + args.fname + 'edgeAttention_' + str(args.load_epoch) + '.dat'
        output.load_state_dict(torch.load(fname))

        args.lr = 0.00001
        epoch = args.load_epoch
        print('model loaded!, lr: {}'.format(args.lr))
    else:
        epoch = 1

    
    optimizer_rnn = optim.Adam(list(rnn.parameters()), lr=args.lr)
    optimizer_output = optim.Adam(list(output.parameters()), lr=args.lr)

    scheduler_rnn = MultiStepLR(optimizer_rnn, milestones=args.milestones, gamma=args.lr_rate)
    scheduler_output = MultiStepLR(optimizer_output, milestones=args.milestones, gamma=args.lr_rate)

    
    time_all = np.zeros(args.epochs)
    for epoch in tqdm(range(1, args.epochs + 1), desc="Epochs"):
        time_start = tm.time()
        
        train_rnn_epoch(epoch, args, rnn, output, dataset_train,
                        optimizer_rnn, optimizer_output,
                        scheduler_rnn, scheduler_output)
        torch.cuda.empty_cache()  
        time_end = tm.time()
        time_all[epoch - 1] = time_end - time_start

        
        if epoch % args.epochs_test == 0 and epoch >= args.epochs_test_start:

            G_pred = []
            index_gene = 1
            G_pred_step = None
            fname = None
            while len(G_pred) < args.generate_size:
                if args.exceed > args.max_num_node:
                    
                    if int(args.exceed/args.generate_size*index_gene) > args.max_num_node:
                        print('ceed有用，进入large生成')
                        G_pred_step = test_rnn_epoch_largeScale(epoch, args, rnn, output,
                                                     gene_num=int(args.exceed / args.generate_size * index_gene),
                                                     test_batch_size=-1, union_graph_ori=union_graph_type)[2]
                    else:
                        print('ceed有用，小号生成')
                        G_pred_step = test_rnn_epoch(epoch, args, rnn, output,
                                                     gene_num=int(args.exceed / args.generate_size * index_gene),
                                                     test_batch_size=-1, union_graph_ori=union_graph_type)[2]
                    fname = args.graph_save_path+'big_gene_stable_v1/' + args.fname_pred + str(epoch) + 'epoch_' + str(index_gene) + 'time_' + str(int(args.exceed/args.generate_size*index_gene)) + 'size.pkl'

                else:
                    print('常规生成')
                    G_pred_step = test_rnn_epoch(epoch, args, rnn, output, gene_num=int(args.max_num_node/args.generate_size*index_gene), test_batch_size=-1, union_graph_ori=union_graph_type)[2]
                    torch.cuda.empty_cache()
                    fname = args.graph_save_path+'common_gene_stable_v1/' + args.fname_pred + str(epoch) + 'epoch_' + str(index_gene) + 'time_' + str(int(args.max_num_node/args.generate_size*index_gene)) + 'size.pkl'
                index_gene = index_gene + 1

                with open(fname, "wb") as f:
                    pickle.dump(G_pred_step, f)
                
                G_pred.append(G_pred_step)

            print('test done, graphs saved')
           
            if epoch % args.epochs_NMD == 0 and epoch >= args.epochs_NMD_start:
                NMD_score = calculate_mmd_tqdm(graphs_list[50:80], G_pred, sigma=1)
                print('NMD_score: ', NMD_score)
  
        if args.save:
            if epoch % args.epochs_save_model == 0:
                fname = args.model_save_path + args.fname + 'GRU_VAE_' + str(epoch) + '.dat'
                torch.save(rnn.state_dict(), fname)
                fname = args.model_save_path + args.fname + 'edgeAttention_' + str(epoch) + '.dat'
                torch.save(output.state_dict(), fname)
        epoch += 1
    np.save(args.timing_save_path+args.fname, time_all)
