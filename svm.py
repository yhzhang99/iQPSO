import re
from sklearn import svm
from sklearn import model_selection
from sklearn import metrics
from feature import get_feature

x = []
y = []


# def get_len(url):
#     return len(url)
#
#
# def get_url_count(url):
#     if re.search('(http://)|(https://)', url, re.IGNORECASE):
#         return 1
#     else:
#         return 0
#
#
# def get_evil_char(url):
#     return len(re.findall("[<>,\'\"/]", url, re.IGNORECASE))
#
#
# def get_evil_word(url):
#     return len(
#         re.findall("(alert)|(script=)(%3c)|(%3e)|(%20)|(onerror)|(onload)|(eval)|(src=)|(prompt)", url, re.IGNORECASE))
#
#
# def get_last_char(url):
#     if re.search('/$', url, re.IGNORECASE):
#         return 1
#     else:
#         return 0
#
#
# def get_feature(url):
#     return [get_len(url), get_url_count(url), get_evil_char(url), get_evil_word(url), get_last_char(url)]


def etl(filename, data, isxss):
    with open(filename, encoding='UTF-8') as f:
        for line in f:
            temp = get_feature(line)
            data.append(temp)
            if isxss:
                y.append(1)
            else:
                y.append(0)
    return data


etl('data/xss.txt', x, 1)
etl('data/normal.txt', x, 0)
# etl('xss-200000.txt',x,1)
# etl('good-xss-200000.txt',x,0)


rbf_svm = svm.SVC(kernel='rbf')
cv_scores = model_selection.cross_val_score(rbf_svm, x, y, cv=3, scoring='accuracy')

print(cv_scores.mean())

# joblib.dump(clf,"xss-svm-200000-module.m")

'''
with open("good-xss-200000.txt") as f:
    for line in f:
#clf.predict([[2., 2.]])
        predict=clf.predict(get_feature(line))
        if predict == 1:
            print("maybe guest error xss %s") % (line)
'''
