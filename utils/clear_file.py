
with open('../word2vec_gensim/cache/points_dataframe_1.csv') as file:
    line_temp = []
    for line in file:
        line = line.split(',')[2:]
        line_temp.append(','.join(line))
    print(line_temp)
    with open('../word2vec_gensim/cache/points_dataframe.csv', 'w') as file:
        file.writelines(line_temp)