import networkx as nx
import numpy as np
from tqdm import tqdm
import processing16
import scipy

from utils import *
from data import *
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from processing16 import DataReader

def create01(args):
    graphs=[]
    with open('twitter16.pkl', 'rb') as f:
        twitter16 = pickle.load(f)

    if args.graph_type=='Twitter':
        graphs = []
        def convert_date_string(date_string):
            

            pattern = r"(\d{4})年(\d{1,2})月"
            match = re.match(pattern, date_string)
            if match:
                year = int(match.group(1))
                month = int(match.group(2))

                

                result = year + month * 0.01
                if type(result) != float:
                    print(date_string)
                return result

        def five_att_top10percent(users: dict):
            tweets = []
            reg = []
            following = []
            fans = []
            for _, atts in users.items():
                if (atts[0] is None) or float(atts[0]) < 0 or (atts[2] is None) or float(atts[2]) < 0 or (
                        atts[3] is None) or float(atts[3]) < 0 or (atts[1] is None) or (atts[4] is None):
                    continue
                tweets.append(int(atts[0]))
                reg.append(convert_date_string(atts[1]))
                following.append(int(atts[2]))
                fans.append(int(atts[3]))

            tweets = [item for item in tweets if item is not None]
            tweets.sort()
            reg = [item for item in reg if item is not None]
            reg.sort()
            following = [item for item in following if item is not None]
            following.sort()
            fans = [item for item in fans if item is not None]
            fans.sort()

            return tweets[int(0.9 * len(tweets))], reg[int(0.1 * len(reg))], following[int(0.9 * len(following))], \
                   fans[int(0.9 * len(fans))]

        def covert_N_type(one_DAG, big_tweet, early_reg, big_following, big_fans):  


            for node_ID in one_DAG.nodes:
                hot_encoder = []
                att_list = one_DAG.nodes.get(node_ID)['att']

                if att_list[1] == -1.111111111 or att_list[1] == -999 or (att_list[1] is None):
                    hot_encoder.append(0)
                else:
                    f = convert_date_string(att_list[1])
                    if f is None:
                        hot_encoder.append(0)
                    elif f < early_reg:
                        hot_encoder.append(1)
                    else:
                        hot_encoder.append(0)

                if att_list[3] is None:
                    hot_encoder.append(0)
                elif float(att_list[3]) > big_fans:
                    hot_encoder.append(1)
                else:
                    hot_encoder.append(0)

                if att_list[4] == "True":
                    hot_encoder.append(1)
                else:
                    hot_encoder.append(0)

                if att_list[2] is None:
                    hot_encoder.append(0)
                elif float(att_list[2]) > big_following:
                    hot_encoder.append(1)
                else:
                    hot_encoder.append(0)

                if att_list[0] is None:
                    hot_encoder.append(0)
                elif float(att_list[0]) > big_tweet:
                    hot_encoder.append(1)
                else:
                    hot_encoder.append(0)

                one_DAG.nodes.get(node_ID)['type_list'] = hot_encoder
                one_DAG.nodes.get(node_ID)['type'] = 16 * hot_encoder[0] + 8 * hot_encoder[1] + 4 * hot_encoder[2] + 2 * \
                                                     hot_encoder[3] + hot_encoder[4]

            return one_DAG

        def processing_DAG(one_DAG, dropout_rate=False):
            if dropout_rate is not False:
                nodes_to_remove = []
                root = [node for node, in_degree in one_DAG.in_degree() if in_degree == 0][0]
                for neighbor in one_DAG.neighbors(root):
                    if one_DAG.out_degree(neighbor) == 0:
                        if random.random() < dropout_rate:
                            nodes_to_remove.append(neighbor)
                for node in nodes_to_remove:
                    one_DAG.remove_node(node)
            tweets_l = []
            reg_l = []
            followings_l = []
            fans_l = []
            ren_l = []
            for one_user in one_DAG.nodes:
                att_list = one_DAG.nodes.get(one_user)['att']
                if float(att_list[0]) != -1.111111111 and float(att_list[0]) != -999:
                    tweets_l.append(float(att_list[0]))
                    reg_l.append(convert_date_string(att_list[1]))
                    reg_l = [item for item in reg_l if item is not None]
                    followings_l.append(float(att_list[2]))
                    fans_l.append(float(att_list[3]))
                    ren_l.append(1 if att_list[4] == 'True' else 0)

            

            return sum(tweets_l)/len(tweets_l), sum(reg_l)/len(reg_l), sum(followings_l)/len(followings_l), sum(fans_l)/len(fans_l), sum(ren_l)/len(ren_l)

        rumor_y = []
        common_y = []

        tweets_L_R = []
        reg_L_R = []
        followings_L_R = []
        fans_L_R = []
        ren_L_R = []

        tweets_L_C = []
        reg_L_C = []
        followings_L_C = []
        fans_L_C = []
        ren_L_C = []
        for i, one_DAG in enumerate(tqdm(twitter16.data['propagation_DAG'])):
            y = twitter16.data['Fake_or_True'][i]
            if y == 'unverified':
                continue

            if y == 'true': 

                rumor_y.append(y)
                tweets, reg_date, followings, fans, is_varify = processing_DAG(one_DAG, dropout_rate=args.dropout_rate)
                tweets_L_R.append(tweets)
                reg_L_R.append(reg_date)
                followings_L_R.append(followings)
                fans_L_R.append(fans)
                ren_L_R.append(is_varify)
            else: 

                common_y.append(y)
                tweets, reg_date, followings, fans, is_varify = processing_DAG(one_DAG, dropout_rate=args.dropout_rate)
                tweets_L_C.append(tweets)
                reg_L_C.append(reg_date)
                followings_L_C.append(followings)
                fans_L_C.append(fans)
                ren_L_C.append(is_varify)

        combined_array_R = np.column_stack((np.array(tweets_L_R), np.array(reg_L_R), np.array(followings_L_R),
                                               np.array(fans_L_R), np.array(ren_L_R)))
        combined_array_C = np.column_stack((np.array(tweets_L_C), np.array(reg_L_C), np.array(followings_L_C),
                                               np.array(fans_L_C), np.array(ren_L_C)))
        combined_array = np.concatenate((combined_array_R, combined_array_C), axis=0)
        y_array = np.concatenate((np.array(rumor_y), np.array(common_y)))
        scaler = MinMaxScaler()
        standardized_data = scaler.fit_transform(combined_array)

        k = 5  

        selector = SelectKBest(chi2, k=k)
        selector.fit(standardized_data, y_array)

        

        feature_scores = selector.scores_
        feature_ranks = np.argsort(-feature_scores)
        print("Feature scores:", feature_scores)
        print("Feature importance ranking (from most to least important):", feature_ranks)

        

        



        big_tweet, early_reg, big_following, big_fans = five_att_top10percent(twitter16.data['user'])
        for i, one_DAG in enumerate(tqdm(twitter16.data['propagation_DAG'])):
            y = twitter16.data['Fake_or_True'][i]
            if y == 'true':
                one_DAG_type = covert_N_type(one_DAG, big_tweet, early_reg, big_following, big_fans)
                graphs.append(one_DAG_type)

        union_graph = twitter16.data['unionGraph']
        union_graph_type = covert_N_type(union_graph, big_tweet, early_reg, big_following, big_fans)

    return graphs, union_graph_type

def create(args):


    graphs=[]
    with open('twitter16.pkl', 'rb') as f:
        twitter16 = pickle.load(f)

    if args.graph_type=='Twitter':
        graphs = []
        def convert_date_string(date_string):
            

            pattern = r"(\d{4})年(\d{1,2})月"
            match = re.match(pattern, date_string)
            if match:
                year = int(match.group(1))
                month = int(match.group(2))

                

                result = year + month * 0.01
                if type(result) != float:
                    print(date_string)
                return result

        def five_att_top10percent(users: dict):
            tweets = []
            reg = []
            following = []
            fans = []
            for _, atts in users.items():
                if (atts[0] is None) or float(atts[0]) < 0 or (atts[2] is None) or float(atts[2]) < 0 or (
                        atts[3] is None) or float(atts[3]) < 0 or (atts[1] is None) or (atts[4] is None):
                    continue
                tweets.append(int(atts[0]))
                reg.append(convert_date_string(atts[1]))
                following.append(int(atts[2]))
                fans.append(int(atts[3]))

            tweets = [item for item in tweets if item is not None]
            tweets.sort()
            reg = [item for item in reg if item is not None]
            reg.sort()
            following = [item for item in following if item is not None]
            following.sort()
            fans = [item for item in fans if item is not None]
            fans.sort()

            return [tweets[-1], tweets[int(0.9 * len(tweets))],tweets[int(0.8 * len(tweets))],tweets[int(0.7* len(tweets))],tweets[int(0.6 * len(tweets))],
                    tweets[int(0.5 * len(tweets))],tweets[int(0.4 * len(tweets))],tweets[int(0.3 * len(tweets))],tweets[int(0.2 * len(tweets))],
                    tweets[int(0.1 * len(tweets))]], \
                   [reg[0], reg[int(0.1 * len(reg))], reg[int(0.2 * len(reg))], reg[int(0.3 * len(reg))],reg[int(0.4 * len(reg))],
                    reg[int(0.5 * len(reg))], reg[int(0.6 * len(reg))], reg[int(0.7 * len(reg))],reg[int(0.8 * len(reg))],
                    reg[int(0.9 * len(reg))]], \
                   [following[-1], following[int(0.9 * len(following))], following[int(0.8 * len(following))],
                    following[int(0.7 * len(following))], following[int(0.6 * len(following))],
                    following[int(0.5 * len(following))], following[int(0.4 * len(following))],
                    following[int(0.3 * len(following))], following[int(0.2 * len(following))],
                    following[int(0.1 * len(following))]], \
                   [fans[-1], fans[int(0.9 * len(fans))], fans[int(0.8 * len(fans))], fans[int(0.7 * len(fans))],fans[int(0.6 * len(fans))],
                    fans[int(0.5 * len(fans))], fans[int(0.4 * len(fans))], fans[int(0.3 * len(fans))],fans[int(0.2 * len(fans))],
                    fans[int(0.1 * len(fans))]]

        def covert_N_type(one_DAG, big_tweet, early_reg, big_following, big_fans):  


            for node_ID in one_DAG.nodes:
                hot_encoder = []
                att_list = one_DAG.nodes.get(node_ID)['att']

                if att_list[1] == -1.111111111 or att_list[1] == -999 or (att_list[1] is None):
                    hot_encoder.append(0)
                else:
                    f = convert_date_string(att_list[1])
                    if f is None:
                        hot_encoder.append(0)
                    elif f <= early_reg[0]:
                        hot_encoder.append(1)
                    elif f <= early_reg[1]:
                        hot_encoder.append(0.9)
                    elif f <= early_reg[2]:
                        hot_encoder.append(0.8)
                    elif f <= early_reg[3]:
                        hot_encoder.append(0.7)
                    elif f <= early_reg[4]:
                        hot_encoder.append(0.6)
                    elif f <= early_reg[5]:
                        hot_encoder.append(0.5)
                    elif f <= early_reg[6]:
                        hot_encoder.append(0.4)
                    elif f <= early_reg[7]:
                        hot_encoder.append(0.3)
                    elif f <= early_reg[8]:
                        hot_encoder.append(0.2)
                    elif f <= early_reg[9]:
                        hot_encoder.append(0.1)
                    else:
                        hot_encoder.append(0)

                if att_list[3] is None:
                    hot_encoder.append(0)
                elif float(att_list[3]) >= big_fans[0]:
                    hot_encoder.append(1)
                elif float(att_list[3]) >= big_fans[1]:
                    hot_encoder.append(0.9)
                elif float(att_list[3]) >= big_fans[2]:
                    hot_encoder.append(0.8)
                elif float(att_list[3]) >= big_fans[3]:
                    hot_encoder.append(0.7)
                elif float(att_list[3]) >= big_fans[4]:
                    hot_encoder.append(0.6)
                elif float(att_list[3]) >= big_fans[5]:
                    hot_encoder.append(0.5)
                elif float(att_list[3]) >= big_fans[6]:
                    hot_encoder.append(0.4)
                elif float(att_list[3]) >= big_fans[7]:
                    hot_encoder.append(0.3)
                elif float(att_list[3]) >= big_fans[8]:
                    hot_encoder.append(0.2)
                elif float(att_list[3]) >= big_fans[9]:
                    hot_encoder.append(0.1)
                else:
                    hot_encoder.append(0)

                if att_list[4] == "True":
                    hot_encoder.append(1)
                    hot_encoder.append(0)
                else:
                    hot_encoder.append(0)
                    hot_encoder.append(1)

                if att_list[2] is None:
                    hot_encoder.append(0)
                elif float(att_list[2]) >= big_following[0]:
                    hot_encoder.append(1)
                elif float(att_list[2]) >= big_following[1]:
                    hot_encoder.append(0.9)
                elif float(att_list[2]) >= big_following[2]:
                    hot_encoder.append(0.8)
                elif float(att_list[2]) >= big_following[3]:
                    hot_encoder.append(0.7)
                elif float(att_list[2]) >= big_following[4]:
                    hot_encoder.append(0.6)
                elif float(att_list[2]) >= big_following[5]:
                    hot_encoder.append(0.5)
                elif float(att_list[2]) >= big_following[6]:
                    hot_encoder.append(0.4)
                elif float(att_list[2]) >= big_following[7]:
                    hot_encoder.append(0.3)
                elif float(att_list[2]) >= big_following[8]:
                    hot_encoder.append(0.2)
                elif float(att_list[2]) >= big_following[9]:
                    hot_encoder.append(0.1)
                else:
                    hot_encoder.append(0)

                if att_list[0] is None:
                    hot_encoder.append(0)
                elif float(att_list[0]) >= big_tweet[0]:
                    hot_encoder.append(1)
                elif float(att_list[0]) >= big_tweet[1]:
                    hot_encoder.append(0.9)
                elif float(att_list[0]) >= big_tweet[2]:
                    hot_encoder.append(0.8)
                elif float(att_list[0]) >= big_tweet[3]:
                    hot_encoder.append(0.7)
                elif float(att_list[0]) >= big_tweet[4]:
                    hot_encoder.append(0.6)
                elif float(att_list[0]) >= big_tweet[5]:
                    hot_encoder.append(0.5)
                elif float(att_list[0]) >= big_tweet[6]:
                    hot_encoder.append(0.4)
                elif float(att_list[0]) >= big_tweet[7]:
                    hot_encoder.append(0.3)
                elif float(att_list[0]) >= big_tweet[8]:
                    hot_encoder.append(0.2)
                elif float(att_list[0]) >= big_tweet[9]:
                    hot_encoder.append(0.1)
                else:
                    hot_encoder.append(0)

                one_DAG.nodes.get(node_ID)['type_list'] = hot_encoder
                one_DAG.nodes.get(node_ID)['type'] = int(100000 * hot_encoder[0] + 10000 * hot_encoder[1] + 1000 * hot_encoder[2] + 100 * \
                                                     hot_encoder[4] + 10*hot_encoder[5])

            return one_DAG

        def processing_DAG(one_DAG, dropout_rate=False):
            if dropout_rate is not False:
                nodes_to_remove = []
                root = [node for node, in_degree in one_DAG.in_degree() if in_degree == 0][0]
                for neighbor in one_DAG.neighbors(root):
                    if one_DAG.out_degree(neighbor) == 0:
                        if random.random() < dropout_rate:
                            nodes_to_remove.append(neighbor)
                for node in nodes_to_remove:
                    one_DAG.remove_node(node)
            tweets_l = []
            reg_l = []
            followings_l = []
            fans_l = []
            ren_l = []
            for one_user in one_DAG.nodes:
                att_list = one_DAG.nodes.get(one_user)['att']
                if float(att_list[0]) != -1.111111111 and float(att_list[0]) != -999:
                    tweets_l.append(float(att_list[0]))
                    reg_l.append(convert_date_string(att_list[1]))
                    reg_l = [item for item in reg_l if item is not None]
                    followings_l.append(float(att_list[2]))
                    fans_l.append(float(att_list[3]))
                    ren_l.append(1 if att_list[4] == 'True' else 0)

            

            return sum(tweets_l)/len(tweets_l), sum(reg_l)/len(reg_l), sum(followings_l)/len(followings_l), sum(fans_l)/len(fans_l), sum(ren_l)/len(ren_l)

        rumor_y = []
        common_y = []

        tweets_L_R = []
        reg_L_R = []
        followings_L_R = []
        fans_L_R = []
        ren_L_R = []

        tweets_L_C = []
        reg_L_C = []
        followings_L_C = []
        fans_L_C = []
        ren_L_C = []
        for i, one_DAG in enumerate(tqdm(twitter16.data['propagation_DAG'])):
            y = twitter16.data['Fake_or_True'][i]
            if y == 'unverified':
                continue

            if y == 'true': 

                rumor_y.append(y)
                tweets, reg_date, followings, fans, is_varify = processing_DAG(one_DAG, dropout_rate=args.dropout_rate)
                tweets_L_R.append(tweets)
                reg_L_R.append(reg_date)
                followings_L_R.append(followings)
                fans_L_R.append(fans)
                ren_L_R.append(is_varify)
            else: 

                common_y.append(y)
                tweets, reg_date, followings, fans, is_varify = processing_DAG(one_DAG, dropout_rate=args.dropout_rate)
                tweets_L_C.append(tweets)
                reg_L_C.append(reg_date)
                followings_L_C.append(followings)
                fans_L_C.append(fans)
                ren_L_C.append(is_varify)

        combined_array_R = np.column_stack((np.array(tweets_L_R), np.array(reg_L_R), np.array(followings_L_R),
                                               np.array(fans_L_R), np.array(ren_L_R)))
        combined_array_C = np.column_stack((np.array(tweets_L_C), np.array(reg_L_C), np.array(followings_L_C),
                                               np.array(fans_L_C), np.array(ren_L_C)))
        combined_array = np.concatenate((combined_array_R, combined_array_C), axis=0)
        y_array = np.concatenate((np.array(rumor_y), np.array(common_y)))
        scaler = MinMaxScaler()
        standardized_data = scaler.fit_transform(combined_array)

        k = 5  

        selector = SelectKBest(chi2, k=k)
        selector.fit(standardized_data, y_array)

        

        feature_scores = selector.scores_
        feature_ranks = np.argsort(-feature_scores)
        print("Feature scores:", feature_scores)
        print("Feature importance ranking (from most to least important):", feature_ranks)

        

        



        big_tweet, early_reg, big_following, big_fans = five_att_top10percent(twitter16.data['user'])
        for i, one_DAG in enumerate(tqdm(twitter16.data['propagation_DAG'])):
            y = twitter16.data['Fake_or_True'][i]
            if y == 'true':
                one_DAG_type = covert_N_type(one_DAG, big_tweet, early_reg, big_following, big_fans)
                graphs.append(one_DAG_type)

        union_graph = twitter16.data['unionGraph']
        union_graph_type = covert_N_type(union_graph, big_tweet, early_reg, big_following, big_fans)

    return graphs, union_graph_type

def create_big(args):


    graphs=[]
    with open('twitter16.pkl', 'rb') as f:
        twitter16 = pickle.load(f)

    if args.graph_type=='Twitter':
        graphs = []
        def convert_date_string(date_string):
            

            pattern = r"(\d{4})年(\d{1,2})月"
            match = re.match(pattern, date_string)
            if match:
                year = int(match.group(1))
                month = int(match.group(2))

                

                result = year + month * 0.01
                if type(result) != float:
                    print(date_string)
                return result

        def five_att_top10percent(users: dict):
            tweets = []
            reg = []
            following = []
            fans = []
            for _, atts in users.items():
                if (atts[0] is None) or float(atts[0]) < 0 or (atts[2] is None) or float(atts[2]) < 0 or (
                        atts[3] is None) or float(atts[3]) < 0 or (atts[1] is None) or (atts[4] is None):
                    continue
                tweets.append(int(atts[0]))
                reg.append(convert_date_string(atts[1]))
                following.append(int(atts[2]))
                fans.append(int(atts[3]))

            tweets = [item for item in tweets if item is not None]
            tweets.sort()
            reg = [item for item in reg if item is not None]
            reg.sort()
            following = [item for item in following if item is not None]
            following.sort()
            fans = [item for item in fans if item is not None]
            fans.sort()

            return [tweets[-1], tweets[int(0.9 * len(tweets))],tweets[int(0.8 * len(tweets))],tweets[int(0.7* len(tweets))],tweets[int(0.6 * len(tweets))],
                    tweets[int(0.5 * len(tweets))],tweets[int(0.4 * len(tweets))],tweets[int(0.3 * len(tweets))],tweets[int(0.2 * len(tweets))],
                    tweets[int(0.1 * len(tweets))]], \
                   [reg[0], reg[int(0.1 * len(reg))], reg[int(0.2 * len(reg))], reg[int(0.3 * len(reg))],reg[int(0.4 * len(reg))],
                    reg[int(0.5 * len(reg))], reg[int(0.6 * len(reg))], reg[int(0.7 * len(reg))],reg[int(0.8 * len(reg))],
                    reg[int(0.9 * len(reg))]], \
                   [following[-1], following[int(0.9 * len(following))], following[int(0.8 * len(following))],
                    following[int(0.7 * len(following))], following[int(0.6 * len(following))],
                    following[int(0.5 * len(following))], following[int(0.4 * len(following))],
                    following[int(0.3 * len(following))], following[int(0.2 * len(following))],
                    following[int(0.1 * len(following))]], \
                   [fans[-1], fans[int(0.9 * len(fans))], fans[int(0.8 * len(fans))], fans[int(0.7 * len(fans))],fans[int(0.6 * len(fans))],
                    fans[int(0.5 * len(fans))], fans[int(0.4 * len(fans))], fans[int(0.3 * len(fans))],fans[int(0.2 * len(fans))],
                    fans[int(0.1 * len(fans))]]

        def covert_N_type(one_DAG, big_tweet, early_reg, big_following, big_fans):  


            for node_ID in one_DAG.nodes:
                hot_encoder = []
                att_list = one_DAG.nodes.get(node_ID)['att']

                if att_list[1] == -1.111111111 or att_list[1] == -999 or (att_list[1] is None):
                    hot_encoder.append(0)
                else:
                    f = convert_date_string(att_list[1])
                    if f is None:
                        hot_encoder.append(0)
                    elif f <= early_reg[0]:
                        hot_encoder.append(10)
                    elif f <= early_reg[1]:
                        hot_encoder.append(9)
                    elif f <= early_reg[2]:
                        hot_encoder.append(8)
                    elif f <= early_reg[3]:
                        hot_encoder.append(7)
                    elif f <= early_reg[4]:
                        hot_encoder.append(6)
                    elif f <= early_reg[5]:
                        hot_encoder.append(5)
                    elif f <= early_reg[6]:
                        hot_encoder.append(4)
                    elif f <= early_reg[7]:
                        hot_encoder.append(3)
                    elif f <= early_reg[8]:
                        hot_encoder.append(2)
                    elif f <= early_reg[9]:
                        hot_encoder.append(1)
                    else:
                        hot_encoder.append(0)

                if att_list[3] is None:
                    hot_encoder.append(0)
                elif float(att_list[3]) >= big_fans[0]:
                    hot_encoder.append(10)
                elif float(att_list[3]) >= big_fans[1]:
                    hot_encoder.append(9)
                elif float(att_list[3]) >= big_fans[2]:
                    hot_encoder.append(8)
                elif float(att_list[3]) >= big_fans[3]:
                    hot_encoder.append(7)
                elif float(att_list[3]) >= big_fans[4]:
                    hot_encoder.append(6)
                elif float(att_list[3]) >= big_fans[5]:
                    hot_encoder.append(5)
                elif float(att_list[3]) >= big_fans[6]:
                    hot_encoder.append(4)
                elif float(att_list[3]) >= big_fans[7]:
                    hot_encoder.append(3)
                elif float(att_list[3]) >= big_fans[8]:
                    hot_encoder.append(2)
                elif float(att_list[3]) >= big_fans[9]:
                    hot_encoder.append(1)
                else:
                    hot_encoder.append(0)

                if att_list[4] == "True":
                    hot_encoder.append(1)
                    hot_encoder.append(0)
                else:
                    hot_encoder.append(0)
                    hot_encoder.append(1)

                if att_list[2] is None:
                    hot_encoder.append(0)
                elif float(att_list[2]) >= big_following[0]:
                    hot_encoder.append(10)
                elif float(att_list[2]) >= big_following[1]:
                    hot_encoder.append(9)
                elif float(att_list[2]) >= big_following[2]:
                    hot_encoder.append(8)
                elif float(att_list[2]) >= big_following[3]:
                    hot_encoder.append(7)
                elif float(att_list[2]) >= big_following[4]:
                    hot_encoder.append(6)
                elif float(att_list[2]) >= big_following[5]:
                    hot_encoder.append(5)
                elif float(att_list[2]) >= big_following[6]:
                    hot_encoder.append(4)
                elif float(att_list[2]) >= big_following[7]:
                    hot_encoder.append(3)
                elif float(att_list[2]) >= big_following[8]:
                    hot_encoder.append(2)
                elif float(att_list[2]) >= big_following[9]:
                    hot_encoder.append(1)
                else:
                    hot_encoder.append(0)

                if att_list[0] is None:
                    hot_encoder.append(0)
                elif float(att_list[0]) >= big_tweet[0]:
                    hot_encoder.append(10)
                elif float(att_list[0]) >= big_tweet[1]:
                    hot_encoder.append(9)
                elif float(att_list[0]) >= big_tweet[2]:
                    hot_encoder.append(8)
                elif float(att_list[0]) >= big_tweet[3]:
                    hot_encoder.append(7)
                elif float(att_list[0]) >= big_tweet[4]:
                    hot_encoder.append(6)
                elif float(att_list[0]) >= big_tweet[5]:
                    hot_encoder.append(5)
                elif float(att_list[0]) >= big_tweet[6]:
                    hot_encoder.append(4)
                elif float(att_list[0]) >= big_tweet[7]:
                    hot_encoder.append(3)
                elif float(att_list[0]) >= big_tweet[8]:
                    hot_encoder.append(2)
                elif float(att_list[0]) >= big_tweet[9]:
                    hot_encoder.append(1)
                else:
                    hot_encoder.append(0)

                one_DAG.nodes.get(node_ID)['type_list'] = hot_encoder
                one_DAG.nodes.get(node_ID)['type'] = int(10000 * hot_encoder[0] + 9999 * hot_encoder[1] + 999 * hot_encoder[2] + 99 * \
                                                     hot_encoder[4] + 9*hot_encoder[5])

            return one_DAG

        def processing_DAG(one_DAG, dropout_rate=False):
            if dropout_rate is not False:
                nodes_to_remove = []
                root = [node for node, in_degree in one_DAG.in_degree() if in_degree == 0][0]
                for neighbor in one_DAG.neighbors(root):
                    if one_DAG.out_degree(neighbor) == 0:
                        if random.random() < dropout_rate:
                            nodes_to_remove.append(neighbor)
                for node in nodes_to_remove:
                    one_DAG.remove_node(node)
            tweets_l = []
            reg_l = []
            followings_l = []
            fans_l = []
            ren_l = []
            for one_user in one_DAG.nodes:
                att_list = one_DAG.nodes.get(one_user)['att']
                if float(att_list[0]) != -1.111111111 and float(att_list[0]) != -999:
                    tweets_l.append(float(att_list[0]))
                    reg_l.append(convert_date_string(att_list[1]))
                    reg_l = [item for item in reg_l if item is not None]
                    followings_l.append(float(att_list[2]))
                    fans_l.append(float(att_list[3]))
                    ren_l.append(1 if att_list[4] == 'True' else 0)

            

            return sum(tweets_l)/len(tweets_l), sum(reg_l)/len(reg_l), sum(followings_l)/len(followings_l), sum(fans_l)/len(fans_l), sum(ren_l)/len(ren_l)

        rumor_y = []
        common_y = []

        tweets_L_R = []
        reg_L_R = []
        followings_L_R = []
        fans_L_R = []
        ren_L_R = []

        tweets_L_C = []
        reg_L_C = []
        followings_L_C = []
        fans_L_C = []
        ren_L_C = []
        for i, one_DAG in enumerate(tqdm(twitter16.data['propagation_DAG'])):
            y = twitter16.data['Fake_or_True'][i]
            if y == 'unverified':
                continue

            if y == 'true': 

                rumor_y.append(y)
                tweets, reg_date, followings, fans, is_varify = processing_DAG(one_DAG, dropout_rate=args.dropout_rate)
                tweets_L_R.append(tweets)
                reg_L_R.append(reg_date)
                followings_L_R.append(followings)
                fans_L_R.append(fans)
                ren_L_R.append(is_varify)
            else: 

                common_y.append(y)
                tweets, reg_date, followings, fans, is_varify = processing_DAG(one_DAG, dropout_rate=args.dropout_rate)
                tweets_L_C.append(tweets)
                reg_L_C.append(reg_date)
                followings_L_C.append(followings)
                fans_L_C.append(fans)
                ren_L_C.append(is_varify)

        combined_array_R = np.column_stack((np.array(tweets_L_R), np.array(reg_L_R), np.array(followings_L_R),
                                               np.array(fans_L_R), np.array(ren_L_R)))
        combined_array_C = np.column_stack((np.array(tweets_L_C), np.array(reg_L_C), np.array(followings_L_C),
                                               np.array(fans_L_C), np.array(ren_L_C)))
        combined_array = np.concatenate((combined_array_R, combined_array_C), axis=0)
        y_array = np.concatenate((np.array(rumor_y), np.array(common_y)))
        scaler = MinMaxScaler()
        standardized_data = scaler.fit_transform(combined_array)

        k = 5  

        selector = SelectKBest(chi2, k=k)
        selector.fit(standardized_data, y_array)

        feature_scores = selector.scores_
        feature_ranks = np.argsort(-feature_scores)
        print("Feature scores:", feature_scores)
        print("Feature importance ranking (from most to least important):", feature_ranks)

        big_tweet, early_reg, big_following, big_fans = five_att_top10percent(twitter16.data['user'])
        for i, one_DAG in enumerate(tqdm(twitter16.data['propagation_DAG'])):
            y = twitter16.data['Fake_or_True'][i]
            if y == 'true':
                one_DAG_type = covert_N_type(one_DAG, big_tweet, early_reg, big_following, big_fans)
                graphs.append(one_DAG_type)

        union_graph = twitter16.data['unionGraph']
        union_graph_type = covert_N_type(union_graph, big_tweet, early_reg, big_following, big_fans)

    return graphs, union_graph_type