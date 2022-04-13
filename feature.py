import re

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

def get_char_1(url):
    return len(re.findall('script', url, re.IGNORECASE))


def get_char_2(url):
    return len(re.findall('<', url, re.IGNORECASE))


def get_char_3(url):
    return len(re.findall('>', url, re.IGNORECASE))


def get_char_4(url):
    return len(re.findall('/', url, re.IGNORECASE))


def get_char_5(url):
    return len(re.findall('iframe', url, re.IGNORECASE))

def get_char_6(url):
    return len(re.findall('onerror', url, re.IGNORECASE))


def get_char_7(url):
    return len(re.findall('onload', url, re.IGNORECASE))


def get_char_8(url):
    return len(re.findall('src=', url, re.IGNORECASE))


def get_char_9(url):
    return len(re.findall('prompt', url, re.IGNORECASE))

def get_char_10(url):
    return len(re.findall('alert', url, re.IGNORECASE))

#对于data2
# 目前1，2，3，4，5，6，9有优化效果
# 1，4，5，6，9有优化效果
# def get_feature(url):
#     return [get_char_2(url), get_char_3(url), get_char_6(url), get_char_5(url)]
