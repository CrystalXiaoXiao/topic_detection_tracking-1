import os
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from time import time
from gensim.models import word2vec

# load corpus
def load_corpus(corpus_path):
    documents = word2vec.Text8Corpus(fname=corpus_path)
    return documents


# train
def train(corpus_path, sg=0, min_count=20, size=256, seed=1, iter=8, workers=15):
    corpus_path_list = corpus_path.split('/')
    model_path = './results/model_local/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model_file_path = model_path + corpus_path_list[-1].split('.')[0] + '_word2vec.model'
    # './results/model_local/corpusseg_merge.model'

    ##训练word2vec模型
    # 获取日志信息
    begin_time = time()
    logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)
    # model exists?
    if not os.path.exists(model_file_path):
        # 加载分词后的文本，使用的是Text8Corpus类
        documents = word2vec.Text8Corpus(fname=corpus_path)
        # 训练模型，部分参数如下
        model = word2vec.Word2Vec(documents, sg=sg, min_count=min_count, size=size, seed=seed, iter=iter,
                                  workers=workers, window=7)
        # 保留模型，方便重用
        model.save(model_file_path)
    else:
        model = word2vec.Word2Vec.load(model_file_path)
    end_time = time()
    print('Total processing time: %d seconds' % (end_time - begin_time))
    return model

# similarity(word1, word2):
def similarity(md, word1, word2):
    y1 = md.similarity(word1, word2)  # 算两个词的相似度/相关程度
    print('两个词的相似度: ', y1)
    return y1 # float

# TODO: plot
# most_similar, top_list
def most_similar(md, word, topn=20):
    try:
        y2 = md.most_similar(word, topn=topn) #计算某个词的最相关词列表
        print('与"{}"相关词有：{}\n'.format(word, y2))
    except:
        print("词典中无此词，请换一个试试？")
        y2 = []
    return y2

def doesnt_match(md, words_list):
    y3 = md.doesnt_match('刘凡 周昌 任笑 推出 日票'.split()) #寻找不合群的词
    print('不合群的词有：', y3)

# save dataframe
def save_datafram(dataframe, path_or_buf='./word2vec_gensim/cache/data_frame.csv'):
    path_list = path_or_buf.split('/')
    if not os.path.exists(path_list[:-1]):
        os.mkdir(path_list[:-1])
    pd.DataFrame.to_csv(path_or_buf=path_or_buf)
    print("[SAVE] Saving the dataframe successfully. PATH:{}".format(path_or_buf))
# visualize
def getpoint_visualize(md, size=300):
    # 模型可视化
    # 使用t-SNE可视化学习的嵌入。t-SNE是一种数据可视化工具，可将数据的维度降至2或3维，从而可以轻松进行绘制。
    # 由于t-SNE算法的空间复杂度是二次的，因此在本教程中，我们将仅查看模型的一部分。
    # 我们使用下面的代码从我们的词汇中选择10,000个单词
    if not os.path.exists('./word2vec_gensim/cache'):
        os.makedirs('./word2vec_gensim/cache')
    path_or_buf = './word2vec_gensim/cache/points_dataframe.csv'
    if not os.path.exists(path_or_buf):
        count = 50000
        word_vectors_matrix = np.ndarray(shape=(count, size), dtype='float32')
        word_list = []
        i = 0
        for word in md.wv.vocab:
            word_vectors_matrix[i] = md[word]
            word_list.append(word)
            i = i+1
            if i == count:
                break
        print("word_vectors_matrix shape is: ", word_vectors_matrix.shape)

        # 由于模型是一个size=256维向量，利用Scikit-Learn 中的降维算法t-SNE
        # 初始化模型并将我们的单词向量压缩到二维空间
        import sklearn.manifold as ts
        tsne = ts.TSNE(n_components=2, random_state=0)
        word_vectors_matrix_2d = tsne.fit_transform(word_vectors_matrix)
        print("word_vectors_matrix_2d shape is: ", word_vectors_matrix_2d.shape)
        # 数据框，其中包含所选单词和每个单词的x和y坐标
        points = pd.DataFrame(
            [(word, coords[0], coords[1]) for word, coords in [(word, word_vectors_matrix_2d[word_list.index(word)])
                                                               for word in word_list] ], columns=["word", "x", "y"])
        points.to_csv(path_or_buf=path_or_buf)
    else:
        points = pd.read_csv(path_or_buf, index_col=0)
    print("Points DataFrame built")
    print(points.head(10))
    return points # Datafram

#TODO:
# plot point:
def plot_point(data_frame):
	 # 指定默认字体：解决plot不能显示中文问题,否则会显示成方块
	plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 更新字体格式
	plt.rcParams['font.size'] = 12
	plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
	plt.title('Word Points')
	sns.set_context('paper') #四种预设，按相对尺寸的顺序(线条越来越粗)，分别是paper，notebook, talk, and poster
	points.plot.scatter("x", "y", s=10, figsize=(16, 9))
	plt.savefig('./figures/point_{}.png'.format(int(len(data_frame))))
	plt.show()

#TODO:
# 将相似词转化为坐标显示出来，一个model一幅图，共计24

if __name__ == '__main__':

    corpus_path = '../results/corpusseg_merge.txt'
    md = train(corpus_path, size=256)
    points = getpoint_visualize(md, size=256)
    #visualize(train('../results/test_corpus_seg.txt'))
    # train(corpus_path)
    plot_point(points)
    while True:
        print(most_similar(md, input('>>> word:')))