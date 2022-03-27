# -*- coding: utf-8 -*-
import time,datetime
import pandas as pd
import math
import numpy as np
import random
import geohash
from tqdm import tqdm


# -*- coding: utf-8 -*-
import time, datetime
import pandas as pd
import math
import numpy as np
import tools

#the path of the input file
filepath='../dataset/cic-darknet-2020/Darknet.csv'

#the path of the output traning file
outfile='../midfile/S2_train_testlist_8Fe_darknet.csv'
#the path of the output testing file
outfile_test='../midfile/S2_test_testlist_8Fe_darknet.csv'
#the path of the edgelist file
edge_outfile='..midfile/S2_train_edgelist_darknet.csv'

data = pd.read_csv(filepath,sep=',',dtype=str)
selected_features=['Idle Max','Fwd Seg Size Min','Bwd Packet Length Min','Protocol','Idle Mean',
                   'FWD Init Win Bytes','FIN Flag Count','Subflow Bwd Bytes',#'Packet Length Std',
                   'Bwd Packet Length Max',
                   'Label']
data = data[selected_features]
data = data.rename(columns={'Label': 'sign'})
data = data.sample(frac=1)
label = data['sign'].map(lambda x: 1 if x in ['Tor','VPN'] else 0)
label_test = label[int(len(data)*0.7):]
label = label[:int(len(data)*0.7)]
data['time']='2022'+'03'+'08'+'000000'

time_d=data['time']


data = data[selected_features[:-1]]



data_train= data[:int(len(data)*0.7)]
data_test= data[int(len(data)*0.7):]

##Clean the data and change it to the desensitization data of node_id

node2id = {}
id2node = {}
id = 0
for value in data_train.values:
    for i, v in enumerate(value):
        key = str(data_train.columns[i]) + '_' + str(v)

        if key not in node2id:
            node2id[key] = id
            id2node[str(id)] = key
            id += 1
print('node count:' + str(len(node2id)))

tools.savedict('../midfile/node2id_darknet.txt', node2id)
tools.savedict('../midfile/id2node_darknet.txt', id2node)
##generating training data
for i, colu in enumerate(data_train.columns):
    data_train[colu] = data_train[colu].map(lambda x: node2id[str(colu) + '_' + str(x)])

(data_train.join(label)).to_csv(outfile, sep=',', index=False)
data_train = (data_train.join(label[:int(len(data)*0.7)])).join(time_d[:int(len(data)*0.7)])
##generating testing data
for i, colu in enumerate(data_test.columns):
    data_test[colu] = data_test[colu].map(
        lambda x: node2id[str(colu) + '_' + str(x)] if str(colu) + '_' + str(x) in node2id else '')
(data_test.join(label_test)).to_csv(outfile_test, sep=',', index=False)
data_test = (data_test.join(label[int(len(data)*0.7):])).join(time_d[int(len(data)*0.7):])
# %%
def pro_edgelist(data):
    '''
    Convert the event into multiple edges
    '''
    edgelist = pd.DataFrame(columns=['node1', 'type1', 'node2', 'type2', 'type','time'])
    colus = data.columns[:-2].values.tolist()
    for i in range(len(colus)):
        for j in range(i + 1, len(colus)):
            temp = data[[colus[i], colus[j]]]
            temp = temp.rename(columns={colus[i]: 'node1', colus[j]: 'node2'})
            temp['type1'] = colus[i]
            temp['type2'] = colus[j]
            temp['type'] = str(colus[i]) + '-' + str(colus[j])
            temp['time'] = data['time']
            edgelist = edgelist.append(temp)
    edgelist = edgelist.dropna(axis=0, how='any')
    edgelist = edgelist.sort_values(by=['node1', 'node2', 'time', 'type'])
    return edgelist


def kappa_function(t, th, b=0.5, timedecay_unit=60 * 60 * 24 * 7):
    '''
    :param t: Current time
    :param th: Historical time when the event occurred
    :param b: Hyperparameter
    :param timedecay_unit: Unit
    :return: Time attenuation weight
    '''
    t = time.strptime(t, "%Y%m%d%H%M%S")
    th = time.strptime(th, "%Y%m%d%H%M%S")
    time_diff = int(time.mktime(t)) - int(time.mktime(th))
    return math.exp(-b * (time_diff / timedecay_unit))


def delta_function(th):
    return 1


def pro_time_weight(data):
    lasttime = '20220308000000'
    timeperiod = 60 * 60 * 24
    timedecay_unit = 60 * 60 * 24 * 7
    timeperiod_unit = 60 * 60
    data['time_weight'] = data['time'].apply(
        lambda x: kappa_function(lasttime, x, 0.5, 60 * 60 * 24) * delta_function(x))
    return data


def u_x_y(x, y, embdict):
    '''
    :param x: node x
    :param y: node y
    :return: cosine similarity of word vector
    '''
    x, y = int(x), int(y)
    x, y = id2node[str(x)][2:], id2node[str(y)][2:]
    if True:
        vecx, vecy = np.array(embdict[str(x)[0]], dtype=float), np.array(embdict[str(y)[0]], dtype=float)
        for i in str(x)[1:]:
            if i not in embdict: continue
            vecx += np.array(embdict[i], dtype=float)
        for j in str(y)[1:]:
            if j not in embdict: continue
            vecy += np.array(embdict[j], dtype=float)

        return tools.compute_cos(vecx, vecy)
    else:
        return 0.5


def sigmoid(x):
    if x==0: return -float('inf')
    x = math.log(1.8 * x)
    return round(1 / (1 + math.exp(-x + 5)), 2)


def pro_sumweight(data):
    data = data[['node1', 'type1', 'node2', 'type2', 'type', 'time_weight']].groupby(
        by=['node1', 'type1', 'node2', 'type2', 'type']).agg({'time_weight': sum}).reset_index()
    #pre-trained word emb file
    embdict = tools.readdict("../dataset/pretrain_vector/crawl-300d-2M-10.vec")
    data['u'] = data[['node1', 'node2']].apply(lambda row: u_x_y(row[0], row[1], embdict), axis=1)
    data['weight'] = data[['u', 'time_weight']].apply(lambda row: row[0] * (1 + sigmoid(row[1])), axis=1)
    data = data.sort_values(by=['weight', 'node1', 'node2'])
    return data[['node1', 'type1', 'node2', 'type2', 'type', 'weight']]


def undirect(data):
    da = data[['node2', 'type2', 'node1', 'type1', 'type', 'weight']]
    da = da.rename(columns={'node2': 'node1', 'type2': 'type1', 'node1': 'node2', 'type1': 'type2'})
    data = data.append(da, ignore_index=True)
    return data
edges = pro_edgelist(data_train)
edges = pro_time_weight(edges)
edges = pro_sumweight(edges)
edges = undirect(edges)
edges.to_csv(edge_outfile, index=False, sep=',')
