# encoding=utf-8
from itertools import islice
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
def compute_cos(vec1, vec2):
    '''计算两个向量之间余弦相似度，输入向量为list格式'''
    if type(vec1)==list:
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
def load_embdict(fname):
    '''从学习到的向量中加载为字典'''
    nodevec_dict={}
    file = open(fname,'r')
    for line in islice(file,1,None):
            line=line.strip().split(' ')
            nodevec_dict[str(line[0])]=line[1:]
    file.close()
    return nodevec_dict

def load_vectors(fname):
    #官方给定的向量加载
    import io
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data

def load_vectors_gensim(fname):
    from gensim.models import KeyedVectors
    #fname="./fastText/wiki-news-300d-1M.vec"
    data=KeyedVectors.load_word2vec_format(fname)
    return data
def load_vectors_gensim_bin(fname):
    from gensim.models import FastText
    #fname= "./fastText/cc.en.300.bin"
    data=FastText.load_fasttext_format(fname)
    return data

#%%


if __name__=='__main__':
    data=load_vectors("../dataset/pretrain_vector/wiki-news-300d-1M.vec")

    #data=load_vectors_gensim("../dataset/pretrain_vector/wiki-news-300d-1M.vec")