import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from model import GTN
import pdb
import pickle
import argparse
from utils import f1_score
from tqdm import tqdm

class Analyzer(torch.utils.data.Dataset):

    def __init__(self, Q, eventFile):
        super(Analyzer, self).__init__()
        self.Q = Q
        self.classes={}
        self.sttics_dict={}
        self.id2class={}
        self.node_c={}
        self.we_c={}
        self.timesx_c={}
        self.readFiles(eventFile)
    def readFiles(self,  eventFile):

        def get_event_statics(fname):
            import pandas as pd
            data = pd.read_csv(fname, low_memory=False, dtype=str )
            #data=data.sample(frac=0.01)##测试使用，正常请注释
            id2class={}#判断node为什么类型的字段，以对应We
            for row in data.values:
                for i in range(len(row[:-1])):
                    id_node=int(row[i])
                    if id_node in id2class: continue
                    id2class[id_node]=i

            sign=data.columns[-1]
            classes = {}
            for i in set(data[sign]):
                classes[i] = len(data.loc[data[sign] == i])

            statics_dict = {}
            for c in classes:
                statics_dict[c] = {}
                for colu in data.columns[:-1]:
                    datacount=(data.loc[data[sign] == c])[colu].value_counts()
                    for ind in datacount.index:
                            statics_dict[c][int(ind)] = datacount.loc[ind]
            if len(classes)>2:
                lam = 2 / (len(classes) * (len(classes) - 1))
            else:
                lam=1

            node_c = {}
            we_c = {}
            timesx_c = {}
            for c in classes:
                nodex = []
                wex = []
                timesx = []
                for x in statics_dict[c]:
                    if x < 0:
                        continue
                    nodex.append(x)
                    wex.append(id2class[x])
                    timesx.append([statics_dict[c][x]])
                nodex = torch.tensor(nodex)
                wex = torch.tensor(wex)
                timesx = torch.tensor(timesx)
                node_c[c] = nodex
                we_c[c] = wex
                timesx_c[c] = timesx

            return classes, statics_dict, id2class, lam, node_c,we_c,timesx_c
        self.classes, self.statics_dict, self.id2class, self.lam, self.node_c, self.we_c, self.timesx_c=get_event_statics(eventFile)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='darknet',
                        help='Dataset')
    parser.add_argument('--epoch', type=int, default=100,
                        help='Training Epochs')
    parser.add_argument('--node_dim', type=int, default=128,
                        help='Node dimension')
    parser.add_argument('--num_channels', type=int, default=2,
                        help='number of channels')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001,
                        help='l2 reg')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layer')
    parser.add_argument('--norm', type=str, default='true',
                        help='normalization')
    parser.add_argument('--adaptive_lr', type=str, default='false',
                        help='adaptive learning rate')
    parser.add_argument( '--path',
                        action='store', default='../midfile/result_emb/nrl_gtn_', type=str,
                        help="output path")
    args = parser.parse_args()
    print(args)
    epochs = args.epoch
    node_dim = args.node_dim
    num_channels = args.num_channels
    lr = args.lr
    weight_decay = args.weight_decay
    num_layers = args.num_layers
    norm = args.norm
    adaptive_lr = args.adaptive_lr

    import tools
    import scipy
    import numpy as np
    import pandas as pd

    eventFile = '../midfile/S2_train_testlist_8Fe_gtn_darknet.csv'
    analyzer = Analyzer(5, eventFile)
    handict = tools.readdict('../midfile/iddict_gtn_darknet.txt')
    hanrdict = tools.readdict('../midfile/rdict_gtn_darknet.txt')
    # generate features
    embdict = tools.readdict("../dataset/pretrain_vector/crawl-300d-2M-10.vec")

    infile = '../midfile/S2_train_edgelist_gtn_darknet.csv'
    data = pd.read_csv(infile, sep=' ', dtype=str, nrows=None)
    data = data[:]
    # 将任意两种元路径都保留
    data.type = data[['type1', 'type2']].apply(lambda row: row[0] + '-' + row[1], axis=1)
    metalist = list(set(data.type))
    ## 从networkX构造csc_matrix
    import networkx as nx
    all_nodes=list(set(list(data.node1) + list(data.node2)))
    edges = []
    num_nodes = len(all_nodes)
    i=0
    for meta in tqdm(metalist):
        data1 = data.loc[data.type == meta]
        nodes = list(set(list(data1.node1) + list(data1.node2)))
        nodes0 = list(set(list(data1.node1)))
        nodes1 = list(set(list(data1.node2)))
        print(len(data1.node1),len(data1.node2))
        edge = [(i, j) for i, j in zip(data1.node1, data1.node2)]
        graph1 = nx.DiGraph()
        graph1.add_nodes_from(all_nodes, bipartite=0)
        graph1.add_nodes_from(nodes0, bipartite=0)
        graph1.add_nodes_from(nodes1, bipartite=1)
        graph1.add_edges_from(edge)
        edgeA = nx.adjacency_matrix(graph1).todense()
        edgeA = np.array(edgeA, dtype=np.int8)
        # A=scipy.sparse.csc_matrix(A)
        edgeA = scipy.sparse.csr_matrix(edgeA)

        if i == 0:
            A = torch.from_numpy(edgeA.todense()).type(torch.FloatTensor).unsqueeze(-1)
        else:
            A = torch.cat([A, torch.from_numpy(edgeA.todense()).type(torch.FloatTensor).unsqueeze(-1)], dim=-1)
        i+=1
    A = torch.cat([A, torch.eye(num_nodes).type(torch.FloatTensor).unsqueeze(-1)], dim=-1)




    node_features=[]
    for i in range(num_nodes):
        n=handict[i]
        vecx=np.array(embdict['0'],dtype=float)
        for j in n:
            if j in embdict:
                vecx+=np.array(embdict[str(j)],dtype=float)
        node_features.append(vecx)
    node_features=np.array(node_features)
    #node_features = torch.FloatTensor(node_features)

    #generate labels from node type
    node2label={}
    for row in data.values:
        if row[0] not in node2label: node2label[row[0]] = row[1]
        if row[2] not in node2label: node2label[row[2]] = row[3]
    label2id={}
    id=0
    for l in list(set(node2label.values())):
        label2id[l]=id
        id+=1
    labels = np.zeros(num_nodes, dtype=np.int64)
    for i in range(num_nodes):
        labels[i] = label2id[node2label[str(i)]]
    labels = torch.LongTensor(labels)


    node_features = torch.from_numpy(node_features).type(torch.FloatTensor)


    num_classes = len(set(node2label.values()))

    float_mask = np.zeros(num_nodes)
    for label_ in label2id:
        pc_c_mask = (labels==label2id[label_])
        float_mask[pc_c_mask] = np.random.permutation(np.linspace(0, 1, pc_c_mask.sum()))
    train_node = np.where(float_mask <= 0.2)[0]
    train_target=np.array([int(label2id[node2label[str(i)]]) for i in train_node])
    train_target = torch.from_numpy(train_target).type(torch.LongTensor)

    valid_node = np.where((float_mask > 0.2) & (float_mask <= 0.3))[0]
    valid_target = np.array([int(label2id[node2label[str(i)]]) for i in valid_node])
    valid_target = torch.from_numpy(valid_target).type(torch.LongTensor)
    test_node = np.where(float_mask > 0.3)[0]
    test_target = np.array([int(label2id[node2label[str(i)]]) for i in test_node])
    test_target = torch.from_numpy(test_target).type(torch.LongTensor)

    final_f1 = 0
    for l in range(1):
        model = GTN(num_edge=A.shape[-1],
                            num_channels=num_channels,
                            w_in = node_features.shape[1],
                            w_out = node_dim,
                            num_class=num_classes,
                            num_layers=num_layers,
                            norm=norm,
                            num_nodes=num_nodes)
        model.device= torch.device("cpu")
        if adaptive_lr == 'false':
            optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)
        else:
            optimizer = torch.optim.Adam([{'params':model.weight},
                                        {'params':model.linear1.parameters()},
                                        {'params':model.linear2.parameters()},
                                        {"params":model.layers.parameters(), "lr":0.5}
                                        ], lr=0.005, weight_decay=0.001)




        loss = nn.CrossEntropyLoss()
        # Train & Valid & Test
        best_val_loss = 10000
        best_test_loss = 10000
        best_train_loss = 10000
        best_train_f1 = 0
        best_val_f1 = 0
        best_test_f1 = 0

        for i in tqdm(range(epochs)):
            for param_group in optimizer.param_groups:
                if param_group['lr'] > 0.005:
                    param_group['lr'] = param_group['lr'] * 0.9
            print('Epoch:  ', i + 1)
            model.zero_grad()
            model.train()
            loss, y_train, Ws = model(A, node_features, train_node, train_target)

            los_d = 0
            cluster_centre = {}
            for c in analyzer.classes:
                timesx = (analyzer.timesx_c[c])
                wex = model.weEmbedding((analyzer.we_c[c]))
                nodex = model.nodeEmbedding((analyzer.node_c[c]))
                centres = timesx * wex * nodex
                centres = (centres.T.sum(dim=1)).T / len(centres)
                cluster_centre[c] = centres
            for c in analyzer.classes:
                cluster_centre[c] = cluster_centre[c] / analyzer.classes[c]
            for c1 in analyzer.classes:
                for c2 in analyzer.classes:
                    if c1 == c2: continue
                    dis = 0.5 * (
                        sum((cluster_centre[c1] - cluster_centre[c2]) * (cluster_centre[c1] - cluster_centre[c2]))) ** (
                              0.5)
                    los_d += dis

            l1 = torch.norm(model.weEmbedding.weight, p=1)
            loss = loss - los_d + l1



            train_f1 = torch.mean(
                f1_score(torch.argmax(y_train.detach(), dim=1), train_target, num_classes=num_classes)).cpu().numpy()
            print('Train - Loss: {}, Macro_F1: {}, Dis - Loss: {}, L1 - Loss: {}'.format(loss.detach().cpu().numpy(), train_f1,los_d,l1))
            loss.backward()
            optimizer.step()
            model.eval()
            # Valid
            with torch.no_grad():
                val_loss, y_valid, _ = model.forward(A, node_features, valid_node, valid_target)
                val_f1 = torch.mean(
                    f1_score(torch.argmax(y_valid, dim=1), valid_target, num_classes=num_classes)).cpu().numpy()
                print('Valid - Loss: {}, Macro_F1: {}'.format(val_loss.detach().cpu().numpy(), val_f1))
                test_loss, y_test, W = model.forward(A, node_features, test_node, test_target)
                test_f1 = torch.mean(
                    f1_score(torch.argmax(y_test, dim=1), test_target, num_classes=num_classes)).cpu().numpy()
                print('Test - Loss: {}, Macro_F1: {}\n'.format(test_loss.detach().cpu().numpy(), test_f1))
            if val_f1 > best_val_f1:
                best_val_loss = val_loss.detach().cpu().numpy()
                best_test_loss = test_loss.detach().cpu().numpy()
                best_train_loss = loss.detach().cpu().numpy()
                best_train_f1 = train_f1
                best_val_f1 = val_f1
                best_test_f1 = test_f1

            model.saveWeights(path=args.path, dataset=args.dataset)
        print('---------------Best Results--------------------')
        print('Train - Loss: {}, Macro_F1: {}'.format(best_train_loss, best_train_f1))
        print('Valid - Loss: {}, Macro_F1: {}'.format(best_val_loss, best_val_f1))
        print('Test - Loss: {}, Macro_F1: {}'.format(best_test_loss, best_test_f1))
        final_f1 += best_test_f1

        model.saveWeights(path=args.path, dataset=args.dataset)
