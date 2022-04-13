import re
import numpy as np
from sklearn import model_selection
from sklearn import datasets
from sklearn import svm
import joblib
from sklearn.metrics import classification_report
from sklearn import metrics

x = []
y = []


def get_len(url):
    return len(url)


def get_url_count(url):
    if re.search('(http://)|(https://)', url, re.IGNORECASE):
        return 1
    else:
        return 0


def get_evil_char(url):
    return len(re.findall("[<>,\'\"/]", url, re.IGNORECASE))


def get_evil_word(url):
    return len(
        re.findall("(alert)|(script=)(%3c)|(%3e)|(%20)|(onerror)|(onload)|(eval)|(src=)|(prompt)", url, re.IGNORECASE))


def get_last_char(url):
    if re.search('/$', url, re.IGNORECASE):
        return 1
    else:
        return 0


def get_feature(url):
    return [get_len(url), get_url_count(url), get_evil_char(url), get_evil_word(url), get_last_char(url)]


def do_metrics(y_test, y_pred):
    print("metrics.accuracy_score:")
    print(metrics.accuracy_score(y_test, y_pred))
    print("metrics.confusion_matrix:")
    print(metrics.confusion_matrix(y_test, y_pred))
    print("metrics.precision_score:")
    print(metrics.precision_score(y_test, y_pred))
    print("metrics.recall_score:")
    print(metrics.recall_score(y_test, y_pred))
    print("metrics.f1_score:")
    print(metrics.f1_score(y_test, y_pred))


def etl(filename, data, isxss):
    with open(filename, encoding='UTF-8') as f:
        for line in f:
            f1 = get_len(line)
            f2 = get_url_count(line)
            f3 = get_evil_char(line)
            f4 = get_evil_word(line)
            data.append([f1, f2, f3, f4])
            if isxss:
                y.append(1)
            else:
                y.append(0)
    return data


etl('data/xss-200000.txt', x, 1)
etl('data/good-xss-200000.txt', x, 0)
# etl('xss-200000.txt',x,1)
# etl('good-xss-200000.txt',x,0)


x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.4, random_state=0)

clf = svm.SVC(kernel='linear', C=1).fit(x_train, y_train)

y_pred = clf.predict(x_test)
print(y_train)
print(y_pred)
print(y_test)
do_metrics(y_test, y_pred)

print(clf.score(x_test, y_test))

# joblib.dump(clf,"xss-svm-200000-module.m")

'''
with open("good-xss-200000.txt") as f:
    for line in f:
#clf.predict([[2., 2.]])
        predict=clf.predict(get_feature(line))
        if predict == 1:
            print("maybe guest error xss %s") % (line)
'''
