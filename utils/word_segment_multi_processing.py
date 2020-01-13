# word_segment.py用于语料分词
## coding=utf-8
import os
import jieba
import multiprocessing as mp

#API: called by topic_detection_tracking/main.py
# tuple
def segment_api(file_path=('../results/corpus_url.txt', '../results/corpus.txt')
				, process_num=4):
	# filePath = ['../results/test_url', '../results/test_content']
	job_p = mp.Process(target=job, args=(file_path, process_num))
	job_p.start()
	job_p.join()
	# merge
	corpus_cache_path = "results/seg_part_cache/corpus_seg_cache"
	url_cache_path = "results/seg_part_cache/url_seg_cache"
	# mkdir
	if not os.path.exists(corpus_cache_path):
		os.makedirs(corpus_cache_path)
	if not os.path.exists(url_cache_path):
		os.makedirs(url_cache_path)
	# merge cache file
	merge_cache(corpus_cache_path)
	merge_cache(url_cache_path)


# read the file by line
def load_data(filePath):
	print('Reading data...')
	fileTrainRead = []
	with open(filePath) as fileTrainRaw:
		for line in fileTrainRaw:
			fileTrainRead.append(line)
	print('Reading data finished.')
	# list
	return fileTrainRead

# define this function to print a list with Chinese
def PrintListChinese(list):
	for i in range(len(list)):
		print(list[i])

# get_block_size()
def get_block_size(lenghth_of_fileTrain, cpu_number=8):
	print('Getting the block size...')
	# the number of threading is 8
	block_size = int(lenghth_of_fileTrain / cpu_number)
	print('The block number:{};;;The block size:{}'.format(cpu_number, block_size))
	return block_size

# save the result
def save_result(topic_list, blockTrainSeg, which_block):
	print('Saving block_num:{}...'.format(which_block))
	with open('../seg_part_cache/url_seg_cache/url_subseg_' + str(which_block) + '.cache', 'w', encoding='utf-8') as fw:
		for i in topic_list:
			fw.writelines(i + '\n')
	with open('../seg_part_cache/corpus_seg_cache/corpus_subseg_part_' + str(which_block) + '.cache','wb') as fW:
		for i in range(len(blockTrainSeg)):
			fW.write(blockTrainSeg[i][0].encode('utf-8'))
			fW.write(b'\n')
	print('Saving done.')

# segment word with jieba
def segmentword(url_list, content_list, block_size, which_block):
	#@ topic_list: list ,be constructed of lines
	#@ content_list: list
	#@ block_size: tuple(),# (start_line, end_line)
	print('[BEGIN:Task:{}] Start segmenting task...'.format(which_block))
	url_list = url_list[which_block * block_size:(which_block + 1) * block_size]
	content_list = content_list[which_block * block_size:(which_block + 1) * block_size]
	block_url_result = []
	block_train_seg = [] # save the sub_result
	for i in range(len(content_list)):
		story = content_list[i][9:-11]
		url_temp = list(url_list[i][5:-7].split('/'))
		jietemp = list(jieba.cut(story, cut_all=False))
		len_start = len(jietemp)
		if(len_start == 0):
			continue
		for item in url_temp:
			if item == '':
				url_temp.pop(url_temp.index(item))
		block_url_result.append(' '.join(url_temp))
		while True:
			len_start = len(jietemp)
			for item in jietemp:
				j = item.strip(' /qwertyuiopasdfghjklzxcvbnm<>~!@#$%^&*()_ \
				+-={}[]|\/?:;.,`0123456789')
				# print(type(j))
				if int(len(j)) < 2:
					# print(item + ':' + str(len(j)))
					jietemp.pop(jietemp.index(item))
			len_end = len(jietemp)
			# print("len_start:{};;;len_end:{}".format(len_start,len_end))
			if(len_start == len_end):
				break

		#print(jietemp)
		block_train_seg.append([' '.join(jietemp)])
		#if i % 2000 == 0 :
			#print('[Task:{:02d}]: {:>6d}'.format(which_block,i))
	print('[SAVE:Task{:02}] [block_url_result] {};; [block_train_seg]{}'
		  .format(which_block, (block_url_result), len(block_train_seg)))
	save_result(block_url_result, block_train_seg, which_block)
	# print('[Task:{:02d}] Finished.'.format(which_block))

# collect all the blockcache of task
def job(filePath, process_num = 4):
	print('[JOB] Starting job....')
	topic_list = load_data(filePath[0])   # topic in url
	content_list = load_data(filePath[1]) # all contents
	if not os.path.exists('./results/corpusseg_merge.txt'):
		# get the size of each block
		block_size = get_block_size(len(content_list), process_num) # int
		for i in range(process_num):
			# sub_t = threading.Thread(target=segmentword, args=(content_list, block_size, i))
			sub_mp = mp.Process(target=segmentword, args=(topic_list, content_list
														  , block_size, i))
			sub_mp.start()
			# sub_mp.join()
	else:
		print('[JOB] The segment file is existed. PATH:{}'.format('./results/corpusseg_merge.txt'))
		pass

# merge cache file, input: the path of the cache file
def merge_cache(cache_path):
	print('[MERGE] Starting...')
	print('Current Work Dir: ' + os.getcwd())
	# /{}_merge.txt' =================================../seg_part_cache/corpus_seg_cache
	out_name = './results/{}_merge.txt'.format(''.join(cache_path.split('/')[-1].split('_')[:-1]))
	if(os.path.exists(out_name)):
		os.remove(out_name)
	out_file = open(out_name, 'x', encoding='utf-8')
	cache_list = os.listdir(cache_path) # count the cache file
	print(cache_list)
	file_name_prex = cache_list[0].split('_')[0] + '_'
	cache_file_count = int(len(cache_list))
	print('[MERGE]{} include {} files'.format(cache_path, cache_file_count))
	lines_count = 0
	for i in range(cache_file_count):
		# print(file_name)
		try:
			file_name = cache_path+ '/' + file_name_prex + str(i) +'.cache'
			print('[MERGE] {}'.format(file_name))
			with open(file_name, 'r' ,encoding='utf-8') as temp:
				content = temp.readlines()
				lines_count += int(len(content))
				out_file.writelines(content)
				out_file.writelines('\n')
		except FileNotFoundError:
			print('[{}] {}'.format(FileNotFoundError, file_name))
	out_file.close()
	print('[MERGE] Finished. OUT_PATH:{} | Lines:{}'.format(out_name, lines_count))

# test
if __name__ == '__main__':
	filePath = ['../results/corpus_url.txt', '../results/corpus.txt']
	# filePath = ['../results/test_url', '../results/test_content']
	job(filePath, process_num = 20)
	# merge
	merge_cache("../results/seg_part_cache/corpus_seg_cache")
	merge_cache("../results/seg_part_cache/url_seg_cache")
	# output: merge_file  path: ../results/...
