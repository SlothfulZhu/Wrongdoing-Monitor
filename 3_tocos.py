# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
from itertools import islice
import datetime
from tqdm import tqdm
def cos(vec1, vec2,norm=True):
    '''计算两个向量之间余弦相似度，输入向量为list格式'''
    if vec1==None or vec2==None:
        return None  
    dot_product = 0.0
    normA = 0.0
    normB = 0.0
    for a,b in zip(vec1,vec2):
        a=float(a)
        b=float(b)
        dot_product += a*b
        normA += a**2
        normB += b**2
    if normA == 0.0 or normB==0.0:
        return None
    else:
        return dot_product / ((normA*normB)**0.5)
def loaddict(fname):
    '''从学习到的向量中加载为字典'''
    nodevec_dict={}
    file = open(fname,'r')
    for line in islice(file,1,None):
            line=line.strip().split(' ')
            nodevec_dict[str(line[0])]=line[1:]
    file.close()
    return nodevec_dict


dictfile='../midfile/result_emb/nrl_hin2vec_node_vec_8Fe_dim128_walklen10_window4_event_w_darknet.emb'



Train=0 #1 train   0test
if Train==1:
    file_t='train'
    infile = '../midfile/S2_'+file_t+'_testlist_8Fe_darknet.csv'
    outfile = '../midfile_noevent/S2_'+file_t+'_testlist_8Fe_cos_quanbian_'+dictfile[33:-4]+'.csv'

else:
    file_t='test'
    infile = '../midfile/S2_'+file_t+'_testlist_8Fe_darknet.csv'
    outfile = '../midfile_noevent/S2_'+file_t+'_testlist_8Fe_cos_quanbian_'+dictfile[33:-4]+'.csv'


dict=loaddict(dictfile)
dict['nan']=None
data=pd.read_csv(infile, dtype=str)
cosdict={}
def cal_cos(r0,r1):
    if str(r0)+'_'+str(r1) in cosdict:
        return cosdict[str(r0)+'_'+str(r1)]
    c=cos(dict[str(r0)], dict[str(r1)])
    cosdict[str(r0)+'_'+str(r1)]=c
    cosdict[str(r1) + '_' + str(r0)] = c
    return c

columns=data.columns[:-1]
for i in range(len(columns)):
    for j in tqdm(range(i+1,len(columns))):
        colu_name=str(columns[i])+'--'+str(columns[j])
        data[colu_name]=data[[columns[i], columns[j]]].apply(lambda row: cal_cos(row[0],row[1]), axis=1)


colus=list(data.columns[-28:])

data=data[['sign']+colus]

def findavg(row):
    list=[]
    for i in row:
        if i==i:
            list.append(i)
    narray = np.array(list)
    sum1 = narray.sum()
    avg = sum1 / len(list)
    return avg

def findvar(row):
    list = []
    for i in row:
        if  i==i:
            list.append(i)
    narray = np.array(list)
    var = narray.var()
    return var

datadist=data[colus]

data['avg']=datadist.apply(lambda row:findavg(row),axis=1)
data['var']=datadist.apply(lambda row:findvar(row),axis=1)

data.to_csv(outfile,index=False)
