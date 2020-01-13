import matplotlib.pyplot as plt                          #数学绘图库
import numpy as np                                      #科学数值计算包，可用来存储和处理大型矩阵
import os
from PIL import Image
from wordcloud import WordCloud, ImageColorGenerator   #词云库
 
#1、读入txt文本数据
#text = open('./results/corpusseg_merge.txt',"r").read()
def load_corpus(corpus_seg_path='../results/test_corpus_seg.txt'):
    if not os.path.exists('./results/save_cloud'):
        os.mkdir('./results/save_cloud')
    corpus_seg = ''
    try:
        with open(corpus_seg_path, "r", encoding='utf-8') as file:
            for line in file:
                line = line.replace(' ', '/')  #必须给个符号分隔开分词结果,否则不能绘制词云
                corpus_seg += line
    except FileNotFoundError:
        print('Not found the file:{}'.format(corpus_seg_path))
    return corpus_seg

#3、初始化自定义背景图片
def set_back_groud(image_path):
    image = Image.open(image_path)
    back_graph = np.array(image)
    return back_graph
    
#4、产生词云图
#有自定义背景图：生成词云图由自定义背景图像素大小决定
def take_word_cloud(corpus_path, back_graph_path='./images/beijing.jpg', backgroud_color = 'black', cloud_name='ABC'
				, font_path='../utils/font/SimHei.ttf', background_color='black'):
    print('[WORD CLOUD] Starting...')
    corpus_seg = load_corpus(corpus_path)
    back_graph = set_back_groud(back_graph_path)
    print('[WORD CLOUD] WordCloud...')
    wc = WordCloud(font_path=font_path, background_color = backgroud_color
    				, max_font_size = 50, mask = back_graph)
    wc.generate(corpus_seg)
    #5、绘制文字的颜色以背景图颜色为参考
    image_color = ImageColorGenerator(back_graph)#从背景图片生成颜色值
    wc.recolor(color_func=image_color)
    print('[WORD CLOUD] Saving...')
    if not os.path.exists('./results/save_cloud'):
        os.makedirs('./results/save_cloud')
    wc.to_file("./results/save_cloud/word_cloud_{}"
    			.format(cloud_name))
    			#按照背景图大小保存绘制好的词云图，比下面程序显示更清晰         
    # 6、显示图片
    plt.rcParams['figure.figsize'] = (20, 9)
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    plt.figure("wcloud_{}".format(cloud_name))   #指定所绘图名称
    plt.imshow(wc)         # 以图片的形式显示词云
    plt.axis("off")       # 关闭图像坐标系
    # plt.show()
    print('[WORD CLOUD] Finished...')


if __name__ == '__main__':
    corpus_path = '../results/corpusseg_merge.txt'
    back_graph_path = './images/beijing.jpg'
    take_word_cloud(corpus_path, back_graph_path)
