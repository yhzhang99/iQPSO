import urllib.parse

# url解码文件，r为需解码文件路径，w为解码后文件路径
def dec(filename_r, filename_w):
    with open(filename_r, encoding='UTF-8') as fr:
        data = []
        for line in fr:
            data.append(urllib.parse.unquote(line))
    with open(filename_w, mode='w', encoding='UTF-8') as fw:
        fw.writelines(data)
    return data


r = 'data/dataset/normal_examples.csv'
w = 'normal_examples_d.txt'

dec(r, w)
