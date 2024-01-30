from ltp import LTP
import torch
import pandas as pd
import jieba

ltp_model_path = 'ltp_base'
# 初始化LTP模型
ltp = LTP(ltp_model_path)

def avg_sentence_length(text):
    words = jieba.lcut(text)
    return len(words)/ len(text)
    # 平均句长

data = pd.read_excel('clean_cut.xlsx')
text = data['text']

a = text.apply(avg_sentence_length)
print(a)

# print(type(m))