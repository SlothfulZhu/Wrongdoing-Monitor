import numpy as np
import torch
import torch.nn.functional
import torch.utils.data
from tqdm import tqdm
import math
import sys

class DotProductAttention(torch.nn.Module):
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = torch.nn.Dropout(dropout)


    def SequenceMask(self, X, X_len, value=-1e6):
        maxlen = X.size(1)
        # print(X.size(),torch.arange((maxlen),dtype=torch.float)[None, :],'\n',X_len[:, None] )
        mask = torch.arange((maxlen), dtype=torch.float)[None, :] >= X_len[:, None]
        # print(mask)
        X[mask] = value
        return X

    def masked_softmax(self,X, valid_length):
        # X: 3-D tensor, valid_length: 1-D or 2-D tensor
        softmax = torch.nn.Softmax(dim=-1)
        if valid_length is None:
            return softmax(X)
        else:
            shape = X.shape
            if valid_length.dim() == 1:
                try:
                    valid_length = torch.FloatTensor(valid_length.numpy().repeat(shape[1], axis=0))  # [2,2,3,3]
                except:
                    valid_length = torch.FloatTensor(valid_length.cpu().numpy().repeat(shape[1], axis=0))  # [2,2,3,3]
            else:
                valid_length = valid_length.reshape((-1,))
            # fill masked elements with a large negative, whose exp is 0
            X = self.SequenceMask(X.reshape((-1, shape[-1])), valid_length)

            return softmax(X).reshape(shape)
    def forward(self, query, key, value, valid_length=None):
        d = query.shape[-1]
        # set transpose_b=True to swap the last two dimensions of key

        scores = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(d)
        attention_weights = self.dropout(self.masked_softmax(scores, valid_length))
        print("attention_weight\n", attention_weights)
        return torch.bmm(attention_weights, value)

class Analyzer(torch.utils.data.Dataset):

    def __init__(self, Q):
        super(Analyzer, self).__init__()
        self.Q = Q
        self.edgeList = []
        self.relaObserved = {}
        self.nodeCount = 0
        self.relaCount = 0
        self.attribute = None
        self.attri_dim = 0
        self.classes={}
        self.sttics_dict={}
        self.id2class={}
        self.lam=0
        self.node_c={}
        self.we_c={}
        self.timesx_c={}

    def readFiles(self, undirected, edgeFile, eventFile, attriFile):
        with open(edgeFile) as f:
            nodeCount, relaCount = (int(x) for x in f.readline().split())

            for line in f:
                index_i, index_j, index_k = (int(x) for x in line.split())
                if index_i == index_j:
                    continue
                observedSet = self.relaObserved.setdefault(index_k, set())
                observedSet.add((index_i, index_j))
                self.edgeList.append((index_i, index_j, index_k))
                if not undirected:
                    continue
                observedSet.add((index_j, index_i))
                self.edgeList.append((index_j, index_i, index_k))

        self.nodeCount = nodeCount
        self.relaCount = relaCount
        print("edges: {}".format(len(self.edgeList)))

        def get_event_statics(fname):
            import pandas as pd
            data = pd.read_csv(fname, low_memory=False, dtype=str )
            
            id2class={}#
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


        if not attriFile:
            return self

        with open(attriFile) as f:
            count, dimension = (int(x) for x in f.readline().split())
            assert count == nodeCount
            embedding = np.empty((nodeCount, dimension))

            for index, line in enumerate(f, start=0):
                embedding[index,:] = np.array(line.split(), dtype=float)

        self.attri_dim = dimension
        self.attribute = torch.tensor(embedding, dtype=torch.float)

        return self

    def __len__(self):
        return len(self.edgeList) * self.Q

    def __getitem__(self, index):
        i, j, k = self.edgeList[index // self.Q]
        inputVector = [k, i, j]
        observedSet = self.relaObserved[k]

        if np.random.random() < 0.5:  # corrupt tail
            while True:
                corrupt = np.random.randint(self.nodeCount)
                if corrupt != i and (i, corrupt) not in observedSet:
                    break
            inputVector.extend([i, corrupt])
        else:                         # corrupt head
            while True:
                corrupt = np.random.randint(self.nodeCount)
                if corrupt != j and (corrupt, j) not in observedSet:
                    break
            inputVector.extend([corrupt, j])

        return torch.tensor(inputVector, dtype=torch.int64, requires_grad=False)


class Marine(torch.nn.Module):

    def __init__(self, edgeFile,eventFile, Q=5, dimension=128, undirected=False,
                 alpha=0.0, attriFile=None):
        super(Marine, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #self.device = torch.device('cpu')
        print("device: {}".format(self.device))

        self.analyzer = Analyzer(Q).readFiles(undirected, edgeFile,eventFile, attriFile)
        self.nodeEmbedding = torch.nn.Embedding(self.analyzer.nodeCount, dimension)
        self.relaEmbedding = torch.nn.Embedding(self.analyzer.relaCount, dimension)
        self.linkEmbedding = torch.nn.Embedding(self.analyzer.relaCount, dimension)
        self.weEmbedding = torch.nn.Embedding(len(set(self.analyzer.id2class.values())), 1)
        sub=torch.from_numpy(np.array([0.5]))
        self.weEmbedding.weight.data.copy_(sub)
        #print(list(self.weEmbedding.weight))

        self.alpha = alpha
        self.gen_num=0
        if alpha:
            print("alpha: {} ({})".format(alpha, attriFile))
            self.attribute = torch.nn.Embedding.from_pretrained(
                self.analyzer.attribute, freeze=True)
            self.transform = torch.nn.Linear(dimension, self.analyzer.attri_dim)



    def forward(self, batchVector):
        idx_k = batchVector[:, 0]
        link_k = self.linkEmbedding(idx_k)
        rela_k = self.relaEmbedding(idx_k)
        idx_i = batchVector[:, 1]
        idx_j = batchVector[:, 2]
        pos_i = self.nodeEmbedding(idx_i)  ##节点的向量
        pos_j = self.nodeEmbedding(idx_j)  ##节点的向量
        neg_i = self.nodeEmbedding(batchVector[:, 3])
        neg_j = self.nodeEmbedding(batchVector[:, 4])

        # softplus(corrupt - correct)
        relaError = torch.sum((neg_j - neg_i - pos_j + pos_i) * rela_k, dim=1)
        linkError = torch.sum((neg_i * neg_j - pos_i * pos_j) * link_k, dim=1)
        loss = torch.nn.functional.softplus(relaError + linkError)
        loss = loss.sum()
        #loss = torch.sigmoid(loss)

        #return loss

        # event loss
        los_d = 0
        cluster_centre = {}
        #计算簇新改为矩阵算法，大幅提速待解决
        if str(self.device)=='cpu':
            for c in self.analyzer.classes:
                timesx = self.analyzer.timesx_c[c]
                wex = self.weEmbedding(self.analyzer.we_c[c])
                nodex = self.nodeEmbedding(self.analyzer.node_c[c])
                centres = timesx * wex * nodex
                centres = (centres.T.sum(dim=1)).T / len(centres)
                cluster_centre[c] = centres
            for c in self.analyzer.classes:
                cluster_centre[c] = cluster_centre[c] / self.analyzer.classes[c]
            for c1 in self.analyzer.classes:
                for c2 in self.analyzer.classes:
                    if c1 == c2: continue
                    dis = 0.5 * (
                        sum((cluster_centre[c1] - cluster_centre[c2]) * (cluster_centre[c1] - cluster_centre[c2]))) ** (
                              0.5)
                    los_d += dis
        else:
            for c in self.analyzer.classes:

                timesx = (self.analyzer.timesx_c[c]).cuda()
                wex=self.weEmbedding((self.analyzer.we_c[c]).cuda())
                nodex=self.nodeEmbedding((self.analyzer.node_c[c]).cuda())
                centres = timesx * wex * nodex
                centres= (centres.T.sum(dim=1)).T /len(centres)
                cluster_centre[c] = centres
            for c in self.analyzer.classes:
                cluster_centre[c] = cluster_centre[c] / self.analyzer.classes[c]
            for c1 in self.analyzer.classes:
                for c2 in self.analyzer.classes:
                    if c1 == c2: continue
                    dis = 0.5 * (
                        sum((cluster_centre[c1] - cluster_centre[c2]) * (cluster_centre[c1] - cluster_centre[c2]))) ** (
                              0.5)
                    los_d += dis
        #los_d = torch.tensor([los_d/(batchVector.shape[0]*self.gen_num)] *batchVector.shape[0], dtype=torch.float)
        #los_d=torch.sigmoid(los_d)
        l1 = torch.norm(self.weEmbedding.weight, p=1)
        #sys.stdout.write(str(loss.item())+' '+str(los_d.item())+' '+str(l1.item())+' ')
        return loss-los_d+ l1
        #return l1
        if not self.alpha:
            return loss

        diff_i = self.transform(pos_i) - self.attribute(idx_i)
        diff_j = self.transform(pos_j) - self.attribute(idx_j)
        return loss + self.alpha * (torch.norm(diff_i, p=2, dim=1) +
                                    torch.norm(diff_j, p=2, dim=1))




    def train(self, epoches=100,path='.'):
        self.to(device=self.device)
        optimizer = torch.optim.Adam(self.parameters())
        generator = torch.utils.data.DataLoader(
            self.analyzer, batch_size=64, shuffle=True, num_workers=1)
        self.gen_num=generator.sampler.num_samples
        print('self.gen_num:'+str(self.gen_num))
        loss = 0.0
        for epoch in range(1, epoches + 1):
            loss_old = loss
            loss = 0.0
            i=0
            for batchData in tqdm(generator):
                i+=1
                optimizer.zero_grad()
                batchData = batchData.to(device=self.device)
                batchLoss = self(batchData)
                loss += float(batchLoss)
                batchLoss.backward()  # loss 求导
                optimizer.step()  # 更新参数
                if i%100==0:
                    sys.stdout.write(str([round(float(i),4) for i in self.weEmbedding.weight]))
                    sys.stdout.flush()

            #print("Epoch{:4d}/{}   Loss: {:e}".format(epoch, epoches, loss))
            print("Epoch{:4d}/{}   Loss: {:e}  diffLoss: {:e}".format(epoch, epoches, loss,loss_old-loss))
            #sys.stdout.flush()
            if epoch%1==0:
                self.saveWeights(path)

    @staticmethod
    def dumpTensor(tensor, filePath):
        matrix = tensor.weight
        size = matrix.size()
        with open(filePath, 'w') as f:
            print("{} {}".format(size[0], size[1]), file=f)
            for vec in tqdm(matrix):
                print(' '.join(['{:e}'.format(x) for x in vec]), file=f)

    def saveWeights(self, path='.'):
        import os
        self.dumpTensor(self.weEmbedding, path + 'we' + '_vec_8Fe_dim' + str(128) + '_w_darknet.emb')
        self.dumpTensor(self.nodeEmbedding, path+'node'+'_vec_8Fe_dim' + str(128) + '_w_darknet.emb')

        def writeemb(out):
            def readdict(dictfile):
                f = open(dictfile, 'r')
                a = f.read()
                dict = eval(a)
                f.close()
                return dict

            outfile = open(out, 'w')
            infile = open(path+'node'+'_vec_8Fe_dim' + str(128) + '_w_darknet.emb', 'r')
            iddict = readdict('../midfile/iddict_darknet_marine.txt')  # id:node
            from itertools import islice
            id = 0
            outfile.writelines('1880 128\n')
            for line in tqdm(islice(infile, 1, None)):
                outfile.write(str(iddict[id]) + ' ')
                outfile.writelines(line)
                id += 1
            infile.close()
            outfile.close()
            pass

        writeemb(path + 'nodenid' + '_vec_8Fe_dim' + str(128) + '_w_darknet.emb')
        self.dumpTensor(self.relaEmbedding, path + 'rela' + '_vec_8Fe_dim' + str(128) + '_w_darknet.emb')
        self.dumpTensor(self.linkEmbedding, path + 'link' + '_vec_8Fe_dim' + str(128) + '_w_darknet.emb')

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Marine')


    parser.add_argument('-g', '--edgeFile',
                        action='store', default='../midfile/S2_train_edgelist_darknet_marine.csv', type=str,
                        help="edgeFile file")
    parser.add_argument('-t', '--eventFile',
                        action='store', default='../midfile/S2_train_testlist_8Fe_darknet_marine.csv', type=str,
                        help="edgeFile file")
    #_part1000
    parser.add_argument('-q', '--Q',
                        action='store', default=2, type=int,
                        help="negative samples per instance")
    parser.add_argument('-d', '--dimension',
                        action='store', default=128, type=int,
                        help="output embeddings' dimension")
    parser.add_argument('-u', '--undirected',
                        action='store_true',
                        help="whether the graph is undirected")
    parser.add_argument('-a', '--alpha',
                        action='store', default=0, type=float,
                        help="loss ratio of attributes")
    parser.add_argument('-A', '--attribute',
                        action='store', default=None, type=str,
                        help="attribute file (with a positive alpha)")
    parser.add_argument('-e', '--epoches',
                        action='store', default=20, type=int,
                        help="training epoches")
    parser.add_argument('-p', '--path',
                        action='store', default='../midfile/result_emb/nrl_marine_', type=str,
                        help="output path")

    args = parser.parse_args()
    if args.alpha <= 0.0 or not args.attribute:
        args.alpha = 0.0
        args.attribute = None

    module = Marine(edgeFile=args.edgeFile,
                    eventFile=args.eventFile,
                    Q=args.Q,
                    dimension=args.dimension,
                    undirected=args.undirected,
                    alpha=args.alpha,
                    attriFile=args.attribute)
    module.train(epoches=args.epoches,path=args.path)
    module.saveWeights(path=args.path)


if __name__ == "__main__":
    main()

