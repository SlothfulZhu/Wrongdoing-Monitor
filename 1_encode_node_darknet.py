import pandas as pd


def readdict(dictfile):
    f = open(dictfile, 'r')
    a = f.read()
    dict = eval(a)
    f.close()
    return dict
def savedict(dictfile,dict={}):
    f = open(dictfile, 'w')
    f.write(str(dict))
    f.close()



infile= '../../midfile/S2_train_edgelist_darknet.csv'
outfile= '../../midfile/S2_train_edgelist_darknet_marine.csv'

infile_event= '../../midfile/S2_test_testlist_8Fe_darknet.csv'
outfile_event= '../../midfile/S2_test_testlist_8Fe_darknet_marine.csv'

data=pd.read_csv(infile,dtype=str)
data_event=pd.read_csv(infile_event,dtype=str)

data=data[['node1','node2','type']]

rdict={}
index=0
for v in data['type'].values:
    if v in rdict:
        continue
    else:
        rdict[v]=index
        index+=1

rcount=index
ndict={}
index=0
iddict={}
for v in data[['node1','node2']].values:
    if v[0] not in ndict:
        iddict[index] = str(v[0])
        ndict[str(v[0])]=index
        index+=1
    if v[1] not in ndict:
        ndict[str(v[1])]=index
        iddict[index]=v[1]
        index+=1
ncount=index

print(ncount,rcount)
data['node1']=data['node1'].map(lambda  x: ndict[x])
data['node2']=data['node2'].map(lambda  x: ndict[x])
data['type']=data['type'].map(lambda  x: rdict[x])

def transfer(x):
    if str(x) in ndict:
        return ndict[str(x)]
    else:
        return -1
for c in data_event.columns[:-1]:
    data_event[c]=data_event[c].map(lambda  x: transfer(x))

savedict('../../midfile/iddict_darknet_marine.txt',iddict)


df2 = pd.DataFrame([len(ndict),len(rdict),'']).T
df2.columns = data.columns
data = pd.concat([df2,data], axis=0 ,ignore_index=True) 

data.to_csv(outfile,sep=' ',header=None,index=False)
data_event.to_csv(outfile_event,sep=',',index=False)