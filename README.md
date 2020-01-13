# topic_detection_tracking

#### 介绍
    @ 起因：
      燕山大学2016届2019-2020第一学期综合课设
    @ 目的：
      运用大数据分析及机器学习算法挖掘和发现潜藏的数据价值。
    @ 本组信息：
      计算机科学与技术16-2班
      组长：石佳琪 组员：乔健、易方峻、仲生全
    @ 本组项目介绍：
      数据集：Sougou新闻数据<https://www.sogou.com/labs/resource/cs.php>
      算法：K-Means、word2vec
      可视化：水平直方图、词云
      流程：数据预处理 --> 训练模型（K-Means、word2vec）--> 对每个簇的特征预测Top50相似词 --> 数据可视化（相似词概率h直方图）
    @ 项目思想:
      本项目使用了两个简单算法，并将K-Means所得到的模型中每个簇的特征作为word2vec的输入，这是从整体数据流动上的大概认识。
      目的在于直观的观察每个K-Means模型的每个簇的主题分布情况，以此来挖掘数据信息。
    @ 配置环境：
      操作系统: macOS Catalina10.15.2
      配置: CPU@2.3GHz 8-Cores Intel Core i9 | 内存@32GB 2667MHz DDR4
      开发环境: python3.7 | pycharm社区版
    如果有什么不懂得地方，请移步问答区，在那里可能会有你的答案。
      
          
#### 安装教程

#####第三方库


1.  pip3 install jieba
2.  pip3 install numpy
3.  pip3 install pandas
4.  pip3 install matplotlib
5.  pip3 install seaborn
6.  pip3 install gensim
7.  pip3 install sklearn

#### 使用说明
    
    
    所有依赖安装完成之后，之后准备对应语料即可。注意语料准备请按照**本地化数据说明**中的数据格式准备数据。
    
    如果是第一次运行时间会比较长，请耐心等待。（大约2h以上，届时两个模型由两个进程同时训练）
    
    如果电脑性能不佳，建议修改参数`process_num=#建议10以内#`
    ---> ./main.py:data_processing.segment_api(file_path=file_path, process_num=10)
    
    ☞准备好之后，建议在terminal中运行(cd至工程目录下)python3 main.py
    ps: 由于是多进程运行，所以控制台的输出信息会比较混乱，注意观察每条信息最后标识。（若只担心结果可忽略终端提示信息）
    
    所有的运行结果都会进行本地化以达到复用目的，之后的执行时间少去训练模型的时间以及部分文件记录时间
    ps：如果你对本项目的大致结构有了了解可参考这段建议：
    训练过程中你可以中断训练过程，程序在下一次运行会自动加载以有model，接着上一次训练节点继续训练
    
    
#####工程目录说明
    
    
    ps: '|-' 中'-'代表一级，以此类推，可借助'目录截图.png'查看
    这部分与该项目无太大关系，只是暂时存放以url为标签进行数据分类的数据，分类后的效果非常不理想。分类结果可见./data_set/log/*
    文本格式说明：url标签:该类对应报道数
    |-./data_set
    |--./log
    |---./log_count.txt
    |---./topic_count_list.txt
    该模块为k-Means算法相关的所有实现，详情见代码注释
    |-./k_means_sklearn
    |--./__init__.py
    |--./k_means_alg.py
    该部分为所有数据本地化仓库(详情请见本地化数据说明)
    |-./results
    |--./figures
    |--./model_info
    |--./model_local
    |--./save_cloud
    |--./seg_part_cache
    |--./top_50
    |--./*.txt
    该部分用于存放部分cache文件，无其他用处
    |-./temp
    该模块用于实现一些小工具以及字体文件
    |-./utils
    |--./font: 用于保存字体文件（本字体文件适用于mac os中文显示）
    |--./__init__.py
    |--./classify_story.py: 以url为标签进行分类，分类结果存于data_set(弃用)
    |--./clear_file.py: (弃用)
    |--./getContent.sh: 终端命令
    |--./line_to_list.py: (弃用)
    |--./word_segment_multi_processing.py: 用于多进程加速预处理数据 
    该模块实现与word2vec相关的所有功能
    |-./word2vec_gensim
    |--./__init__.py
    |--./painter.py: 水平直方图绘制
    |--./word2vec.py: 算法API调用及部分功能实现，详情请移步源码文件
    |--./Word2VecDocument.pages: API参数说明，From: 库文件
    该模块实现词云功能，详情请移步word_cloud/word_cloud.py
    |-./word_cloud
    |--./images: 用于存储词云图的背景图片
    |--./save_cloud.py: 示例图片
    
    main.py: 集成所有功能（环境配置好后即可运行main.py）
    
    
#####本地化数据说明
    
    
    所有本地化数据存储与./results(所有目录均相对于工程根目录)
    |-./results: 存储语料及url源文件(terminal command)，处理之后的文件: '*_merge.txt', 其他非主要文件
    |--./seg_part_cache: 用于保存分词时的中间结果（目录中文件数量取决于进程数）
    |--./model_local: 用于存储kmeans模型和word2vec模型
    |--./model_info: 用于存储模型信息，如，K-Means模型每个簇的特征
    |--./save_cloud: 用于保存每个K-Means模型的每个簇的top50预测词
    |--./top_50: 用于存储每个K-Means模型的每个簇的预测词Top50（由word2vec进行预测）（为本实验最终结果）
    |--./figures: 用于保存对最终结果可视化后的结果，包括词云图，水平直方图
    |--./save_cloud: 用于保存词云图
    |--./corpus.txt: (自备, 可参考utils/getContent.sh) 内容示例: <content>南都讯*****出双倍车资买坐票费用有点高。</content>
    |--./corpus_url.txt: (自备, 同上) 内容示例: <url>http://gongyi.sohu.com/20120706/n347457739.shtml</url>
    |--./corpusseg_merge.txt: (运行程序后自动生成)
    |--./urlseg_merge.txt: (同上)
    **如果你想自己重新从零开始训练模型，建议清空当前文件夹。**
    
    
####问答区
      
      
      问: 为什么要对每个K-Means模型的每个簇进行词相似预测并选出Top50？
      答: 开始的约定: true_k: 簇的数量
         本项目中一共训练true_k=3到true_k=24，步长为1，共计22个K-Means模型。有最终可视化效果比对，即每个模型每个簇的Top50的可视化，
         以此查看所挖掘信息大概的分布情况。
      问: 为什么会有这样的想法？
      答: 目的很简单，80%为了好玩儿。这样说可能会有小伙伴觉得很不严谨，其实我觉得这才是学习的真正乐趣，这些算法只是工具而已，实现你自己
      的想法才是最让你开心的事情。
      问: 为什么代码写的这么烂？
      答: 没什么理由，我相信未来会写得更好。
      问: 有没有示例之类的？
      答: 你可以移步至results目录下自行查阅，会有你想要的答案。
      -----------------------------------------------------------------------------------------------------------------
      如果小伙伴还有什么问题可在Issue中提问，欢迎追星加关注。
      
      
####讲到最后
    
    
    本组也是第一次接触大数据这门学科，也是第一次使用python进行大数据分析，难免会有很多瑕疵，所以不论你是特别厉害的大佬，还是和我一样的
    入门小白，在这里我想请各位爱学习的小伙伴抱着谦虚的态度去对待每个人的作品，很期待各位小伙伴可以commit大家一起学习，成长。
    
