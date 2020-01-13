import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# MacOS
# paint
def painter(word_list, value_list, title, xlable_name= 'Frequency', ylabel_name='Words'):
    y_pos = np.arange(int(len(word_list)))
    plt.figure()
    plt.barh(y_pos, value_list, color="blue", align='center')
    # matplotlib.rcParams['font.family'] = 'Microsoft YaHei'
    # SimHei = fm.FontProperties('../utils/font/SimHei.ttf')
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 更新字体格式
    plt.rcParams['font.size'] = 12
    plt.rcParams['figure.figsize'] = (20, 9)
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    plt.yticks(y_pos, word_list)
    plt.ylabel(ylabel_name)
    plt.xlabel(xlable_name)
    plt.title("Topic: " + title)
    if not os.path.exists('./results/figures'):
        os.mkdir('./results/figures')
    plt.savefig('./results/figures/{}'.format(title))
    # plt.show()


if __name__ == '__main__':
    word_list = ['狮子', '2', '3', '4']
    value_list = [0.001, 0.002, 0.005, 0.01]
    title = 'test_cavus'
    painter(word_list, value_list, title)

