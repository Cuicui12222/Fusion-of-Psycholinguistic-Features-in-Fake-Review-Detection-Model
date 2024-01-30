# 编码处理
import codecs
import jieba
import ltp
from ltp import LTP
from ltp import StnSplit
import pandas as pd


ltp_model_path = 'ltp_base'
# 初始化LTP模型
ltp = LTP(ltp_model_path)

# 一些处理工具
# 读自定义的词库
def read_txtfile(file_path):
    with open(file_path, 'r', encoding="utf-8") as file:
        return [line.strip() for line in file]

# 分词


# 读取LIWC词典并计算目标词数量
dict = pd.read_csv("LIWC2015 Dictionary - Chinese (Simplified).dicx",low_memory=False)

dicterm_set = set(dict['DicTerm'])

def count_liwc(words, categories):
    counts = {cat: 0 for cat in categories}
    word_aim = []

    for word in words:
        if word in dicterm_set:  # Use the set for lookup
            row = dict.loc[dict['DicTerm'] == word]
            for cat in categories:
                if not pd.isna(row[cat].values[0]):
                    counts[cat] += 1
                    word_aim.append(word)

    total = sum(counts.values())

    return total, counts, word_aim
# def count_liwc(words, categories):
#     counts = {cat: 0 for cat in categories}
#     word_aim = []
#
#     for word in words:
#         if word in dict['DicTerm'].values:
#             row = dict.loc[dict['DicTerm'] == word]
#             for cat in categories:
#                 if not pd.isna(row[cat].values[0]):
#                     counts[cat] += 1
#                     word_aim.append(word)
#
#     total = sum(counts.values())
#
#     return total, counts, word_aim

# 一、认知负荷相关特征
def word_count(words):
    return len(words)
    # 词数量

def word_diversity(words):
    return len(set(words))
    # 词汇多样性

def type_token_ratio(words):
    return len(set(words)) / len(words)
    # 不重复词个数%


def verb_count(words):
    output = ltp.pipeline(words, tasks=["pos"])
    flat_output = [item for sublist in output.pos for item in sublist]
    print(flat_output)
    verb_count = len([pos for pos in flat_output if pos == 'v'])
    return verb_count
    # 动词个数

def sentence_count(text):
    return len(text)
    # 句子数量


def avg_sentence_length(text):
    if isinstance(text, list):
        text = ' '.join(text)
    words = jieba.lcut(text)
    return len(words)/ len(text)
    # 平均句长


def causation_ratio(words):
    cause_count = count_liwc(words, ['cause'])[0]
    return cause_count / len(words)
    # 因果关系词比例


def exclusive_ratio(words):
    exclusives = read_txtfile('self-made dictionary/exclusive words.txt')
    exclusive_count = sum([words.count(w) for w in exclusives])
    print([word for word in words if word in exclusives])
    print(exclusive_count)
    return exclusive_count / len(words)
    # 排除关系词比例


# 二、确信度相关特征
def modality_ratio(words):
    modal_words = read_txtfile('self-made dictionary/uncertain.txt')
    modal_count = sum([words.count(w) for w in modal_words])
    print([word for word in words if word in modal_words])
    print(modal_count)
    return modal_count / len(words)
    # 不确定词比例


def certainty_ratio(words):
    certain_count = count_liwc(words, ['certain'])[0]
    return certain_count / len(words)
    # 确信度词比例


# 三、负面情感相关特征
def negation_ratio(words):
    negation_count = count_liwc(words,['negate'])[0]
    return negation_count / len(words)
    # 否定词比例


def negative_emotion_ratio(words):
    neg_count = count_liwc(words,['negemo'])[0]
    return neg_count / len(words)
    # 负面情感词比例

def negative_emotions_only(words):
    neg_only_count = count_liwc(words, ['anx','anger','sad'])[0]
    return neg_only_count / len(words)

def anger(words):
    anger_count = count_liwc(words, ['anger'])[0]
    return anger_count / len(words)

def anxiety(words):
    anx_count = count_liwc(words, ['anx'])[0]
    return anx_count / len(words)

def sadness(words):
    sad_count = count_liwc(words, ['sad'])[0]
    return sad_count / len(words)


# 四、积极情感相关特征
def positive_emotion_ratio(words):
    pos_count = count_liwc(words, ['posemo'])[0]
    return pos_count / len(words)
    # 正面情感词比例

#def positive_1(text):
#not yet

#def positive_2(text):
#not yet


# 五、未指明情感的特征
def all_emotion_ratio(words):
    allemo_count = count_liwc(words, ['posemo', 'negemo'])[0]
    return allemo_count / len(words)
    # 情感词比例

#not yet
def pleasantness_ratio(words):
    pleasant_words = read_txtfile('self-made dictionary/pleasantness.txt')
    pleasant_count = sum([words.count(w) for w in pleasant_words])
    return pleasant_count / len(words)
    # 愉快度词比例


# 六、距离感相关特征
def pronoun_ratio(words):
    pronoun_count = count_liwc(words, ['pronoun'])[0]
    return pronoun_count / len(words)
    # 所有代词比例

def fp_sp_ratio(words):
    fp_sp_count = count_liwc(words, ['i'])[0]
    return fp_sp_count / len(words)
    # 第一人称单数代词比例 我 比例

def fp_pp_ratio(words):
    fp_pp = count_liwc(words, ['we'])[0]
    return fp_pp/ len(words)
    # 第一人称复数代词比例 我们 比例

def all_fp_ratio(words):
    all_fp_count = count_liwc(words, ['i', 'we'])[0]
    return all_fp_count / len(words)
    # 所有第一人称代词比例

def all_sp_ratio(words):
    all_sp_count = count_liwc(words, ['you'])[0]
    return all_sp_count / len(words)
    # 所有第二人称代词比例

def all_tp_ratio(words):
    all_tp_count = count_liwc(words, ['shehe', 'they'])[0]
    return all_tp_count / len(words)
    # 所有第三人称代词比例

# not yet
def passive_ratio(words):
    passive_words = read_txtfile('self-made dictionary/passive.txt')
    passive_count = sum([words.count(w) for w in passive_words])
    return passive_count / len(words)


#not yet
def generalization_ratio(words):
    general_words = read_txtfile('self-made dictionary/generalization.txt')
    general_count = sum([words.count(w) for w in general_words])
    return general_count / len(words)
    # 泛化词比例


# 七、感知和情境细节相关特征
def perceptual_ratio(words):
    perceptual_count = count_liwc(words, ['percept'])[0]
    return perceptual_count / len(words)
    # 感知词+视听触觉词汇 比例

#not yet
# def perceptual_only_ratio(text):
#     temporal_words = ['前天', '今天', '明天', '昨天', '时间']
#     words = jieba.lcut(text)
#     temporal_count = sum([words.count(w) for w in temporal_words])
#     return temporal_count / len(words)
    # 感知觉词汇


def seeing_ratio(words):
    seeing_count = count_liwc(words, ['see'])[0]
    return seeing_count / len(words)
    # 视觉词频

def feeling_ratio(words):
    feeling_count = count_liwc(words, ['feel'])[0]
    return feeling_count / len(words)
    # 触觉词频

def hearing_ratio(words):
    hearing_count = count_liwc(words, ['hear'])[0]
    return hearing_count / len(words)
    # 听觉词频

def time_ratio(words):
    time_count = count_liwc(words, ['time'])[0]
    return time_count / len(words)
    # time词频

def space_ratio(words):
    space_count = count_liwc(words, ['space'])[0]
    return space_count / len(words)
    # space词频

def timespace_ratio(words):
    timespace_count = count_liwc(words, ['time','space'])[0]
    return timespace_count / len(words)
    # time+space词频

def prep_ratio(words):
    prep_count = count_liwc(words, ['prep'])[0]
    return prep_count / len(words)
    # prep词频

def num_ratio(words):
    num_count = count_liwc(words, ['number'])[0]
    return num_count / len(words)
    # number词频

def quant_ratio(words):
    quant_count = count_liwc(words, ['quant'])[0]
    return quant_count / len(words)
    # quant词频

#not yet
def modifier_ratio(words):
    output = ltp.pipeline(words, tasks=["pos"])
    flat_output = [item for sublist in output.pos for item in sublist]
    modifier_count = len([pos for pos in flat_output if pos == 'a'])
    return modifier_count / len(words)
    # adj+adv词频

def motion_ratio(words):
    motion_count = count_liwc(words, ['motion'])[0]
    return motion_count / len(words)
    # motion词频

# 八、认知过程相关特征
def cogproc_ratio(words):
    cogproc_count = count_liwc(words, ['cogproc'])[0]
    return cogproc_count / len(words)
    # 认知词比例


def insight_ratio(words):
    insight_count = count_liwc(words, ['insight'])[0]
    return insight_count / len(words)
    # 洞察词比例



