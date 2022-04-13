import numpy as np
from feature import get_feature

def load_data(data_file1, data_file2):
    '''导入训练数据
    input:  data_file(string):训练数据所在文件
    output: data(mat):训练样本的特征
            label(mat):训练样本的标签
    '''
    x = []
    y = []
    with open(data_file1, encoding='UTF-8') as f:
        for line in f:
            temp = get_feature(line)
            x.append(temp)
            y.append(1)
    with open(data_file2, encoding='UTF-8') as f:
        for line in f:
            temp = get_feature(line)
            x.append(temp)
            y.append(0)
    return np.array(x), np.array(y).T