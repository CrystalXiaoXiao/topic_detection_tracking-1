#!--encoding=utf-8
from __future__ import print_function

import os
import matplotlib.pyplot as plt
import multiprocessing as mp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.externals import joblib
from sklearn.manifold import TSNE
from random import shuffle

# find_best_ncluster, mutiple-processing
'''
    这里设置了range(3，25)的循环，目的是为了训练出不同簇的K-Means.model。
    目的在于之后的在不同簇下model之间做性能对比。
    如果你不想，或者电脑资源有限，建议设置小一些，依据数据量的大小去选择适当的循环范围；
    当然，你也可以只循环一次，指定一个你认为合适的簇数值。
    test_set在项目最后并没有用到，是因为电脑算力问题导致之后的sse值无法算出，出现高维危机。
    你可以尝试把特征维度选择低一些，再去改写代码即可
'''
def find_best_ncluster(corpus_seg_path):
    '''测试选择最优参数, 保存中间结果'''
    print("[FIND] Start select the best n for k-means's n_cluster...")
    train_set, test_set= loadDataset(corpus_seg_path)
    print("[FIND] %d documents" % len(train_set))
    tf_idf, vectorizer = transform(train_set, n_features=500)
    test_tf_idf, _ = transform(test_set, n_features=300)
    for i in range(3, 25, 1):
        process = mp.Process(target=train, args=(tf_idf, vectorizer, test_tf_idf,  i))
        process.start()
    # print('[FIND] Finished operating the n_cluster from {} to {}.\n\n'.format(3, 24))
        print('[FIND] Be computing...\n')
#@ called by: main()

# load struct data.
def loadDataset(corpus_seg_path='../results/corpusseg_merge.txt'):
    '''导入文本数据集'''
    print('[LOAD] Loading the dataset.')
    f = open(corpus_seg_path, 'r', encoding='utf-8')
    dataset = []
    for line in f.readlines():
        if not int(len(line)) < 10:
            dataset.append(line)
    f.close()
    shuffle(dataset)
    dataset_count = int(len(dataset))
    set_seg_rate = int(dataset_count / 10)
    # train set
    train_set = dataset[:set_seg_rate * 8]
    train_set_count = int(len(train_set))
    # test set
    test_set = dataset[set_seg_rate * 8:]
    test_set_count = int(len(test_set))
    # valid_test
    
    print('[LOAD] Finished. SIZE: train_set:{0:>7}/{1:<7};;;test_set:{2:>7}/{1:<7}'
          .format(train_set_count, dataset_count, test_set_count))
    return train_set, test_set
# @called by: find_best_ncluster() return: train_set, test_set

# text transform to matrix.
def transform(data, n_features=300):
    vectorizer = TfidfVectorizer(max_df=0.5, max_features=n_features, min_df=2, use_idf=True)
    tf_idf = vectorizer.fit_transform(data)   # tf-idf weight
    return tf_idf, vectorizer
# @ called by: def find_best_ncluster()

# cluster_visualize (time: too long to stop).
def cluster_visualize(i, tf_idf, kmeans_model, n_components=2):
    print('[VISUAL] Cluster_visualizing...')
    if not os.path.exists('./figures'):
        os.mkdir('./figures')
    local_file_name = './figures/cluster_visualize_decomposition{}_model{}.save'\
        .format(n_components, i)
    print('[VISUAL] decomposition_file_name:{}'.format(local_file_name))
    if not os.path.exists(local_file_name):
        print('[VISUAL] Fitting decomposition_data...')
        tf_idf_weight = tf_idf.toarray()
        t = TSNE(n_components=n_components)
        decomposition_data = t.fit_transform(tf_idf_weight)
        print('[VISUAL] Saving decomposition_data...')
        joblib.dump(decomposition_data, local_file_name)
        print('[VISUAL] Saved successfully.\n\n')
    else:
        print('[VISUAL] Loading decomposition_data...')
        decomposition_data = joblib.load(local_file_name)
        print('[VISUAL] Loaded successfully.\n\n')

    print('[VISUAL]  Plot...')
    x = []
    y = []
    for i in decomposition_data:
        x.append(i[0])
        y.append(i[1])

    fig = plt.figure(figsize=(100, 100))
    ax = plt.axes()
    plt.scatter(x, y, c=kmeans_model.labels_, marker="x")
    plt.xticks(())
    plt.yticks(())
    plt.show()
    print('[VISUAL]  Plot finished. Saving...')
    figure_name = './figures/cluster_visualize_decomposition{}_model{}.png'\
        .format(n_components, i)
    plt.savefig(figure_name, aspect=1)
    print('[VISUAL]  Saving finished. PATH:{}'.format(figure_name))
# @ called by: def train()

# compute rss # train model, and return the loss of model (time: so long(first time))
def train(tf_idf, vectorizer, test_tf_idf, true_k=10, minibatch=False, showLable=True):
    # process_i: the identity of the process
    # model file
    print('[TRAIN] Starting training...                                     '
          ': k-means_ncluster_{}.model'.format(true_k))

    if not os.path.exists('./results/model_local'):
        os.makedirs('./results/model_local')
    model_name = './results/model_local/k-means_ncluster_{}.model'.format(true_k)

    if not os.path.exists(model_name):
        # 使用采样数据还是原始数据训练k-means，
        if minibatch:
            km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
                                 init_size=1000, batch_size=1000, verbose=True)
        else:
            km = KMeans(n_clusters=true_k, init='k-means++', max_iter=300, n_init=1,
                        verbose=True)
        km.fit(tf_idf)

        print('[TRAIN] Finished. Starting saving the model                      '
              ': k-means_ncluster_{}.model'.format(true_k))
        # save model
        joblib.dump(km, '{}'.format(model_name))
    else:
        print('[TRAIN] The model is arealy existed, loading...                  '
              ': k-means_ncluster_{}.model'.format(true_k))
        km = joblib.load(model_name)
        print('[TRAIN] Loading successfully.                                    '
              ': k-means_ncluster_{}.model'.format(true_k))

    print('[TRAIN] Operating successfully. Starting saving the info about model '
          ': k-means_ncluster_{}.model\n\n'.format(true_k))
    # info file
    if not os.path.exists('./results/model_info'):
        os.mkdir('./results/model_info')
    info_name_file = './results/model_info/k-means_ncluster_{}.info'.format(true_k)

    info_file = open(info_name_file, 'w', encoding='utf-8')
    if showLable:
        print('[INFO] Top terms per cluster   :   k-means_ncluster_{}.model\n'
              .format(true_k))
        info_file.writelines("[0] Top terms per cluster:\n")

        order_centroids = km.cluster_centers_.argsort()[:, ::-1]
        terms = vectorizer.get_feature_names()
        # print(vectorizer.get_stop_words())
        for i in range(true_k):
            print("[INFO] Cluster %d:" % i, end='')
            info_file.writelines("Cluster {:<2}:".format(i))
            words = ''
            for ind in order_centroids[i, :]:
                print(' %s' % terms[ind], end='')
                words += terms[ind] + ' '
            print()
            info_file.writelines(words + '\n')

        # the time is so long
        '''
        result = list(km.predict(tf_idf))
        print('[INFO] Cluster distribution   :   k-means_ncluster_{}.model\n'
              .format(true_k) + dict([(i, result.count(i)) for i in result]))

        info_file.writelines('[INFO] Cluster distribution   :   k-means_ncluster_{}.model\n'
                             .format(true_k))
        for i, count in iter((i, result.count(i)) for i in result):
            info_file.writelines('{:<2d}::{:>8d}\n'.format(i, count))
        sse = 0
        #TODO:
        # compute and save the RSS
        # sse = -km.score(test_tf_idf)
        print('[INFO] SSR: k-means_ncluster_{}.model :: {:.4f}'.format(true_k, sse))
        info_file.writelines('[INFO] SSR: k-means_ncluster_{}.model :: {:.4f}'
                             .format(true_k, sse))
        '''
        info_file.close()
    #TODO:
    # cluster_visualize
    # cluster_visualize(true_k, tf_idf, km)
    # return -km.score(X)
# @ called by: def find_best_ncluster()



# load rss info of each model, to plot (true_ks, scores)
def load_info_sse():
    info_path = './results/model_info'
    ncluster_scores = []
    for k in range(3, int(len(os.listdir(info_path))) + 3):
        info_name_file = './results/model_info/k-means_ncluster_{}.info'.format(k)
        with open(info_name_file, 'r', encoding='utf-8') as info_file:
            for line in info_file:
                # info_file.writelines('
                # [INFO] SSR: k-means_ncluster_{}.model :: {:.4f}'.format(true_k, rss))
                if '[INFO] SSR: k-means_ncluster_{}.model'.format(k) in line:
                    score = float(line.split('::')[1])
                    ncluster_scores.append((k, score))
    return ncluster_scores
#@ called by: ncluster_scores_plot()


# ncluster_scores_plot: (X, Y)
def ncluster_scores_plot(nclusters_scorcs):
    if not os.path.exists('./results/figures'):
        os.makedirs('./results/figures')
    print('[PLOT] Start ploting (n_cluster[i], RSS[i])')
    true_ks = []
    scores = []
    true_ks.append(i[0] for i in nclusters_scorcs)
    scores.append(i[1] for i in nclusters_scorcs)
    print('[PLOT] x: {}'.format(true_ks))
    print('[PLOT] y: {}'.format(scores))
    plt.figure(figsize=(14, 4))
    plt.plot(true_ks, scores, label="RSS", color="blue", linewidth=1)
    plt.xlabel("n_clusters")
    plt.ylabel("RSS")
    plt.legend()
    plt.show()
    save_name = './results/figures/ncluster_scores_plot.png'
    plt.savefig(save_name, aspect=1)
    print('[PLOT] Finished. PATH:{}'.format(save_name))
#@ called by: main()

# TODO:  # predict：
def predict():
	return

# read info
# TODO: 返回每个簇的特征词构成的列表
def get_feature_ncluster(n_model):
    with open('./results/model_info/k-means_ncluster_{}.info'.format(n_model)) as info:
        line_words_list = []
        for line in info:
            if 'Cluster ' in line:
                wordi_list = line.split(':')[1].split(' ')
                line_words_list.append(wordi_list)
    return line_words_list


if __name__ == '__main__':
    # test1
    # 24 processing, output: (n_cluster, rss) or (true_k, score) | count:24
    # process = mp.Process(target=find_best_ncluster)
    # process.start()
    # process.join()
    # print('[FIND] Finished operating the n_cluster from {} to {}.\n\n'.format(3, 24))
    # # plot (n_clusters, scores)
    # ncluster_scores_plot(load_info_sse())

    # test2
    print(get_feature_ncluster(3)[1])