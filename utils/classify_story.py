import os
# operate the conten using the index

# read file, return list
import time

# corpusseg_merge.txt
def load_file(file_path):
    temp_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if int(len(line)) < 1:
                continue
            temp_list.append(line.strip('\n'))
        return temp_list

# get the topic from url_seg_list
def get_topic_in_url(url_seg_list):
    temp_list = [] # save the uni identify of each url
    for line in url_seg_list:
        if(int(len(line)) < 1):
            continue
        temp_list.append(line.split(' ')[1])
    return temp_list


# dict {'topic_name': [the list of the index]}
def topic_content2index(topic_list):
    print('[CONVERT] Starting convert to index...')
    topic_index_dict = {}
    for i, topic in enumerate(topic_list):
        topic_index_dict.setdefault(topic.strip('\n'), []).append(i)
    # print(list(topic_index_dict.keys())[0])
    # print(list(topic_index_dict.values())[0])
    print('[CONVERT] Finished.')
    return  topic_index_dict

# locality file, transform the index to the file contacts all of story about the topic
def classify_story_local(param_tuple):
    print('[CLASSIFY] Starting...')
    topic_conten_index_dict = param_tuple[0]
    content_list = param_tuple[1]
    log_path = '../data_set/log'
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    log_file = open(log_path + '/log_count.txt', 'w', encoding='utf-8')
    log_file.writelines(time.asctime(time.localtime()) + '\n')
    # input:  topic_conten_index_dict:dict, content_list: list
    # output: locality
    print('[CURREN WORK PATH] {}'.format(os.getcwd()))
    log_file.writelines('[CURREN WORK PATH] {}\n'.format(os.getcwd()))
    count_dict = {} # {'topic' : the count of content about this topic}
    for topic in iter(topic_conten_index_dict):
        index_list = topic_conten_index_dict[topic]
        count_dict[topic] = count_dict.get(topic, 0) + int(len(index_list))
        file_path = '../data_set/' + topic + '.txt'
        with open(file_path, 'w', encoding='utf-8') as f:
            print('[CLASSIFY] {:<50}: {:>10} L'.format(topic, int(len(index_list))))
            log_file.writelines('[CLASSIFY] {:<50}: {:>10} L\n'.format(topic, int(len(index_list))))
            for i in index_list:
                f.writelines(content_list[i]+'\n')
    print('[CLASSIFY] Finished. | TOPIC_COUNT:{:>10}'.format(int(len(topic_conten_index_dict))))
    log_file.writelines('[CLASSIFY] Finished. | TOPIC_COUNT:{:>10}\n'.format(int(len(topic_conten_index_dict))))
    count_dict_list = sorted(count_dict.items(), key=lambda count_dict: count_dict[1], reverse=True)
    print(count_dict)
    log_file.writelines(str(count_dict_list) + '\n')
    with open(log_path + '/topic_count_list.txt', 'w', encoding='utf-8') as f:
        for i, item in enumerate(count_dict_list):
            log_file.writelines('{:<40}:{:>8}\n'.format(item[0], item[1]))
            f.writelines('{:<40}:{:>8}\n'.format(item[0], item[1]))
    log_file.close()

def get_params(file_path):
    return topic_content2index(get_topic_in_url(load_file(file_path[0]))), load_file(file_path[1])

# test for topic_content2index
if __name__ == '__main__':
    url_file_path = '../results/urlseg_merge.txt'
    content_file_path = '../results/corpusseg_merge.txt'
    file_path = (url_file_path, content_file_path)
    classify_story_local(get_params(file_path=file_path))