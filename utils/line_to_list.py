def toList(filePath):
	print('Reading data:{}...'.format(filePath))
	file_list = []
	with open(filePath) as file:
		for line in file:
			line = line.strip('\n')
			file_list.append(line)
	print('Reading data finished.')
	# list
	return file_list

# get the list of url
def urlList(filePath):
	return(toList(filePath))

# get the list of content
def contentList(filePath):
	return(toList(filePath))

def getUrlContentList(urlFilePath, contentFilePath):
	url_list = urlList(urlFilePath)
	return url_list, list(set(url_list)), contentList(contentFilePath)

if __name__ == '__main__':
	url_file = '../temp/topicfile.txt'
	content_file = '../results/seg_part_cache/corpusSegMerge.txt'
	url_list, url_set_list, content_list = getUrlContentList(url_file, content_file)
	print(url_list[:2])
	print(url_set_list)
	print(content_list[1])
	


