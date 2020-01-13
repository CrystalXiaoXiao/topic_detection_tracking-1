import os
import multiprocessing as mp
import threading
import sys
# sys.path.append('k_means_sklearn/k_means_alg.py')
# sys.path.append('utils/word_segment_multi_processing.py')
# sys.path.append('word2vec_gensim/word2vec.py')


import utils.word_segment_multi_processing as data_processing
import k_means_sklearn.k_means_alg as kg
import word2vec_gensim.word2vec as wg
import word2vec_gensim.painter as painter
import word_cloud.word_cloud as word_cloud
#TODO:
# find similar word for each word in every cluster for the models
def similar_words():
    all_cluster_top50 = []
    md = wg.train(corpus_path ='results/corpusseg_merge.txt')
    for i in range(3, int(len(os.listdir('results/model_info')) + 3)):
        cluster_features = kg.get_feature_ncluster(i)
        i_cluster_top50 = [] #
        for n_cluster in cluster_features:
            word_top20 = []
            for word in n_cluster:
                # 列表元祖
                word_top20.extend(wg.most_similar(md, word, topn=20))
            sorted(word_top20, key=lambda x: (x[1], x[0]), reverse=True)
            i_cluster_top50.append(word_top20[:50]) # 有i个簇的model的，第n簇的， top50词
        # 保存top50
        if not os.path.exists('results/top_50'):
            os.mkdir('results/top_50')
        with open('results/top_50/k-means_icluster_{}.top50'.format(i), 'w', encoding='utf-8') as save_file:
            count = 0  #
            for n_cluster in i_cluster_top50:
                for word_similar_tuple in n_cluster:
                    save_file.writelines('[{}]'.format(count) + str(word_similar_tuple[0])
                                         + ':' + str(word_similar_tuple[1]) + '\n')
                count += 1
        all_cluster_top50.append(i_cluster_top50) #((('word_name';similar_value),... ), ...)
    return all_cluster_top50

#TODO:
# 每一个模型model的所有簇的top词分布显示在一张图上，24个模型一共生成24张图。
# visualize
def visulize():
    for i in range(3, int(len(os.listdir('results/model_info'))) + 3):
        with open('results/top_50/k-means_icluster_{}.top50'.format(i), 'r') as save_file:
            n_cluster = 0
            lines = save_file.readlines()
            lines_count = int(len(lines))
            cluster_count = int(lines_count / 50)
            for n_cluster in range(cluster_count):
                words = []
                similar_values = []
                for line in lines:
                    if '[{}]'.format(n_cluster) in line:
                        line = line.split(']')[1].split(':')
                        word, similar_value = line[0], float(line[1].strip('\n'))
                        words.append(word)
                        with open('./temp/words_{}.cache'.format(i), 'w', encoding='utf-8') as words_file:
                            words_file.writelines(' '.join(words))
                        similar_values.append(similar_value)
                # 直方图
                # painter_threading = threading.Thread(target=painter.painter,
                #                                      args=(words,
                #                                      similar_values,
                #                                      'Cluster{}_k-means_{}_top50.png'
                #                                      .format(n_cluster, i)))
                # painter_threading.start()
                # painter_threading.join()
                painter.painter(words, similar_values, title='k-means_{}_cluster{}_top50.png'.format(i, n_cluster))
                # 词云图
                word_cloud.take_word_cloud('./temp/words_{}.cache'.format(i),
                           cloud_name='k-means_{}_cluster{}_top50.png'.format(i, n_cluster),
                           back_graph_path='./word_cloud/images/beijing.jpg',
                            font_path='./utils/font/SimHei.ttf')

if __name__ == '__main__':
    '''
        1st: you should using "cat news_tensite_xml.dat | iconv -f gbk -t utf-8 -c | grep "<content>"  > corpus.txt"
        2nd: next work: data processing
    '''
    # 该部分文件就是通过命令行实现数据初步处理所生成的文件，执行命令可参考utils/getContent.sh
    # file_path = ('./results/corpus_url.txt', './results/corpus.txt')
    # print('[MAIN] [Loading info...]')
    # # output: ./results/{corpusseg | urlseg}_merge.txt
    # data_processing.segment_api(file_path=file_path, process_num=10)
    # # train: k-means.model and word2vec.model
    # print('[MAIN] [Training:[w2v and kmeans]...]')
    # corpus_seg_path = './results/corpusseg_merge.txt'
    # word2vec_train_process = mp.Process(target=wg.train, args=(corpus_seg_path, ))
    # kmeans_train_process = mp.Process(target=kg.find_best_ncluster, args=(corpus_seg_path, ))
    # word2vec_train_process.start()
    # kmeans_train_process.start()
    # word2vec_train_process.join()
    # kmeans_train_process.join()
    # # to find the similar word of each cluster
    # print('[Computing similar...]')
    # all_cluster_top50 = similar_words()
    # print(all_cluster_top50)
    # print('[Visualizing...]')
    visulize()
    print('[Finished...]')



