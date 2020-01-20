import re
import string
import numpy as np
import pandas as pd

import operator
from itertools import chain
from collections import Counter

def text_clean(s):
    s = s.lower()
    s = re.sub(r'[^\w\s]', '', s)
    s = re.sub(r'[0-9]', '', s)
    s = re.sub(r' [a-z]{1,2} ', '', s)
    return s

def word_dict(text, length):
    freq_dict = Counter(chain.from_iterable(map(str.split, text.values)))
    swd = dict(sorted(freq_dict.items(), key=operator.itemgetter(1), reverse=True)[20:1000*length+20])
    return dict(zip(swd.keys(), range(len(swd))))

def gen_clean(word_index, length):
    def _text2array(text):
        text = text_clean(text)
        array = np.zeros(length)
        for s in str.split(text):
            if s in word_index:
                array[word_index[s]] += 1
        return list(array)
    return _text2array

def text2array(text, word_index):
    length = len(word_index)
    return np.array(list(map(gen_clean(word_index, length), text)))