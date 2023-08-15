
import numpy as np
import sklearn
from os.path import join as pjoin
import os
import networkx as nx
import sys
import pickle

class DataReader():
    def __init__(self,
                 data_dir,  

                 rnd_state=None):
        self.data_dir = data_dir
        files = os.listdir(self.data_dir)
        

        self.data = {}  

        self.data['reindex_from0'] = []   

        self.data['event_id']=[]
        self.data['source'] = []

        self.tree_index = 1
        self.data['unionGraph'] = nx.Graph()
        self.data['largest_component'] = nx.Graph()
        self.data['propagation'] = []
        self.data['propagation_DAG'] = []
        self.data['Fake_or_True'] = []
        self.data['user']={}

        self.get_user_att() 

        

        file_count = 1
        for file_name in files:
            self.data['event_id'].append(file_name.replace(".txt", ""))
            self.unionGraphCons(list(filter(lambda f: f.find(file_name) >= 0, files))[0])
            propag_tree = self.propagation_tree_processing(list(filter(lambda f: f.find(file_name) >= 0, files))[0])
            self.data['propagation'].append(propag_tree)

            unique_tree = self.convert_to_directed(propag_tree, self.data['source'][-1])
            self.data['propagation_DAG'].append(unique_tree)
        

        self.get_max_G()

        self.get_labels()

    def convert_to_directed(slef, graph, root):
        directed_graph = nx.DiGraph()
        visited = set()

        def dfs(node):
            visited.add(node)
            for neighbor in graph.neighbors(node):
                if neighbor not in visited:
                    directed_graph.add_edge(node, neighbor)
                    dfs(neighbor)

        dfs(root)
        for node in directed_graph.nodes:
            if 'time' not in directed_graph.nodes[node]:
                directed_graph.nodes[node]['time'] = graph.nodes[node]['time']
            if 'att' not in directed_graph.nodes[node]:
                directed_graph.nodes[node]['att'] = graph.nodes[node]['att']

        return directed_graph

    def get_user_att(self):
        with open(pjoin('complete_result_t16.txt'), 'r', encoding='utf-8') as f:
            lines = [line.strip().split(':') for line in f.readlines()]
        for user_row in lines:
            user_row=user_row[0].replace('[', '').replace(']', '').split(', ')
            user_id = user_row[0]
            if int(user_id) in self.data['user']:
                print(user_id)
            del (user_row[0])
            self.data['user'][int(user_id)] = user_row

    def get_max_G(self):
        largest = max(nx.connected_components(self.data['unionGraph']), key=len)
        self.data['largest_component'] = self.data['unionGraph'].subgraph(largest)

    def get_labels(self):
        with open(pjoin('label.txt'), 'r') as f:
            lines = [line.strip().split(':') for line in f.readlines()]
        for id in self.data['event_id']:
            for label_id in lines:
                if label_id[1] == id:
                    self.data['Fake_or_True'].append(label_id[0])
                    break

    def remove_error_edge(self):
        with open(pjoin('../rumdetect2017/rumor_detection_acl2017/twitter16/remove_v2.txt'), 'r') as f:
            lines = f.readlines()
        for line in lines:
            eles = line.split('\n')[0].split('  ')
            parent = eles[3]
            child = eles[5]  

            w = self.data['unionGraph'][int(parent)][int(child)]['weight']
            self.data['unionGraph'].add_edge(int(parent), int(child), weight=w - 1)  

    def propagation_tree_processing(self, fpath):
        print(fpath)
        with open(pjoin(self.data_dir, fpath), 'r') as f:
            lines = f.readlines()
        g = nx.Graph()
        related_event = set()
        related_event.add(int(self.data['event_id'][-1]))
        first_row = True
        for line in lines:
            if first_row is True:
                first_row = False
                continue
            parent_to_child = line.split('\n')
            parent_to_child = parent_to_child[0].split('->')
            parent_list = eval(parent_to_child[0])
            child_list = eval(parent_to_child[1])
            if float(parent_list[2]) > float(child_list[2]):
                continue
            if int(parent_list[0]) == int(child_list[0]):
                continue
            if int(parent_list[1]) not in related_event and int(child_list[1]) not in related_event:
                continue
            else:
                related_event.add(int(parent_list[1]))
                related_event.add(int(child_list[1]))
            if g.has_node(int(parent_list[0])) and g.has_node(int(child_list[0])):
                g.add_edge(int(parent_list[0]), int(child_list[0]))
                print(int(parent_list[0]), int(child_list[0]))
                cycle = nx.algorithms.cycles.find_cycle(g, orientation='original')
                node_set = set()
                for node_tuple in cycle:
                    node_set.add(node_tuple[0])
                    node_set.add(node_tuple[1])

                node_time_max = -9999999.9
                node_time_min = 99999999.9
                node_source = -1
                node_end = -1
                for n in node_set:
                    if g.nodes[n]['time'] > node_time_max:
                        node_end = n
                        node_time_max = g.nodes[n]['time']
                    if g.nodes[n]['time'] < node_time_min:
                        node_source = n
                        node_time_min = g.nodes[n]['time']
                if len(node_set) > 3:
                    if g.has_edge(node_source, node_end):
                        g.remove_edge(node_source, node_end)
                        print(node_source, "source_end-4remove: ", node_end)
                    else:
                        g.remove_edge(int(parent_list[0]), int(child_list[0]))
                        print(int(parent_list[0]), "凑合当前删-4remove: ",int(child_list[0]))
                else:
                    print(int(parent_list[0]), "三角形的-remove: ", int(child_list[0]))
                    g.remove_edge(node_source, node_end)
                continue

            if g.has_edge(int(parent_list[0]), int(child_list[0])):
                print('错误: ', fpath, " parent: ", int(parent_list[0]), " child: ", int(child_list[0]))
                

                

            g.add_node(int(parent_list[0]))
            g.add_node(int(child_list[0]))
            g.add_edge(int(parent_list[0]), int(child_list[0]))

            if 'time' not in g.nodes[int(parent_list[0])]:
                g.nodes[int(parent_list[0])]['time'] = float(parent_list[2])
            if 'time' not in g.nodes[int(child_list[0])]:
                g.nodes[int(child_list[0])]['time'] = float(child_list[2])
            if 'att' not in g.nodes[int(parent_list[0])]:
                g.nodes[int(parent_list[0])]['att'] = self.data['user'].get(int(parent_list[0]))
            if 'att' not in g.nodes[int(child_list[0])]:
                g.nodes[int(child_list[0])]['att'] = self.data['user'].get(int(child_list[0]))

            

            

        return g

    def unionGraphCons(self, fpath):
        with open(pjoin(self.data_dir, fpath), 'r') as f:
            lines = f.readlines()
        print(fpath)
        first_row = True
        for line in lines:
            parent_to_child = line.split('\n')
            parent_to_child = parent_to_child[0].split('->')
            if first_row is True:
                first_row = False
                self.data['source'].append(int(eval(parent_to_child[1])[0]))
                continue
            parent_list = eval(parent_to_child[0])
            child_list = eval(parent_to_child[1])
            if float(parent_list[2]) > float(child_list[2]):
                continue
            self.data['unionGraph'].add_node(int(parent_list[0]))
            self.data['unionGraph'].add_node(int(child_list[0]))

            if 'att' not in self.data['unionGraph'].nodes[int(parent_list[0])]:
                self.data['unionGraph'].nodes[int(parent_list[0])]['att'] = self.data['user'].get(int(parent_list[0]))
            if 'att' not in self.data['unionGraph'].nodes[int(child_list[0])]:
                self.data['unionGraph'].nodes[int(child_list[0])]['att'] = self.data['user'].get(int(child_list[0]))

            if self.data['unionGraph'].has_edge(int(parent_list[0]), int(child_list[0])):
                w = self.data['unionGraph'][int(parent_list[0])][int(child_list[0])]['weight']
                self.data['unionGraph'].add_edge(int(parent_list[0]), int(child_list[0]), weight=w+1)  

            else:
                self.data['unionGraph'].add_edge(int(parent_list[0]), int(child_list[0]), weight=1)

if __name__ == '__main__':
    twitter16 = DataReader(data_dir='./tree')

    print("1")
