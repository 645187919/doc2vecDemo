# coding:utf-8
import os
import sys
import gensim
import jieba
import numpy as np
from jieba import analyse

from gensim.models.doc2vec import Doc2Vec, LabeledSentence

TaggededDocument = gensim.models.doc2vec.TaggedDocument

def get_datasest():
    fin = open("questions.txt",encoding='utf8').read().strip(' ')   #strip()取出首位空格
    # print(fin)
    # print(type(fin))
    # 添加自定义的词库用于分割或重组模块不能处理的词组。
    jieba.load_userdict("userdict.txt")
    # 添加自定义的停用词库，去除句子中的停用词。
    stopwords = set(open('stopwords.txt',encoding='utf8').read().strip('\n').split('\n'))   #读入停用词
    text = ' '.join([x for x in jieba.lcut(fin) if x not in stopwords])  #去掉停用词中的词
    # print(text)
    print (type(text),len(text))

    x_train = []

    word_list = text.split('\n')
    print(word_list[0])

    for i,sub_list in enumerate(word_list):
        document = TaggededDocument(sub_list, tags=[i])
        # document是一个Tupple,形式为：TaggedDocument( 杨千嬅 现在 教育 变成 一种 生意 , [42732])
        # print(document)
        x_train.append(document)
    return x_train

def getVecs(model, corpus, size):
    vecs = [np.array(model.docvecs[z.tags[0]].reshape(1, size)) for z in corpus]
    return np.concatenate(vecs)

def train(x_train, size=200, epoch_num=1):
    # D2V参数解释：
    # min_count：忽略所有单词中单词频率小于这个值的单词。
    # window：窗口的尺寸。（句子中当前和预测单词之间的最大距离）
    # size:特征向量的维度
    # sample：高频词汇的随机降采样的配置阈值，默认为1e-3，范围是(0,1e-5)。
    # negative: 如果>0,则会采用negativesampling，用于设置多少个noise words（一般是5-20）。默认值是5。
    # workers：用于控制训练的并行数。
    model_dm = Doc2Vec(x_train,min_count=1, window = 5, size = size, sample=1e-3, negative=5, workers=4,hs=1,iter=6)
    # total_examples：统计句子数
    # epochs：在语料库上的迭代次数(epochs)。
    model_dm.train(x_train, total_examples=model_dm.corpus_count, epochs=70)
    model_dm.save('model_test')

    return model_dm

def test():
    model_dm = Doc2Vec.load("model_test")
    test_ = '申请贷款需要什么条件？'
    #读入停用词
    stopwords = set(open('stopwords.txt',encoding='utf8').read().strip('\n').split('\n'))
    #去掉停用词中的词
    test_text = ' '.join([x for x in jieba.lcut(test_) if x not in stopwords])
    print(test_text)
    #获得对应的输入句子的向量
    inferred_vector_dm = model_dm.infer_vector(doc_words=test_text)
    # print(inferred_vector_dm)
    #返回相似的句子
    sims = model_dm.docvecs.most_similar([inferred_vector_dm], topn=10)
    return sims

if __name__ == '__main__':
    x_train = get_datasest()
    # print(x_train)
    # model_dm = train(x_train)
    sims = test()
    # sims:[(89, 0.730167031288147), (6919, 0.6993225812911987), (6856, 0.6860911250114441), (40892, 0.6508388519287109), (40977, 0.6465731859207153), (30707, 0.6388640403747559), (40160, 0.6366203427314758), (11672, 0.6353889107704163), (16752, 0.6346361637115479), (40937, 0.6337493062019348)]
    # sim是一个Tuple,内部包含两个元素，一个是对应的句子的索引号（之前自定义的tag）一个是对应的相似度
    # print(type(sims))
    # print('sims:'+str(sims))
    for count, sim in sims:
        sentence = str(x_train[count])
        # sentence = x_train[count]
        # print('sentence:'+sentence)
        # print('sim:'+str(sim))
        print(sentence, sim, len(sentence)
              )