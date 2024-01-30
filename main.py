# 数据预处理
import feature_engineering as fe
from multiprocessing import Pool
import pandas as pd
import numpy as np
import pickle

def process(self, words,text=None):
    features = []
    features.append(fe.word_count(words))

    features.append(fe.word_diversity(words))
    features.append(fe.type_token_ratio(words))
    # features.append(fe.verb_count(words))
    features.append(fe.sentence_count(text))
    features.append(fe.avg_sentence_length(words,text))
    features.append(fe.causation_ratio(words))
    # features.append(fe.exclusive_ratio(words))
    # features.append(fe.modality_ratio(words))
    features.append(fe.certainty_ratio(words))
    features.append(fe.negation_ratio(words))
    features.append(fe.negative_emotion_ratio(words))
    features.append(fe.negative_emotions_only(words))
    features.append(fe.anger(words))
    features.append(fe.anxiety(words))
    features.append(fe.sadness(words))
    features.append(fe.positive_emotion_ratio(words))
    features.append(fe.all_emotion_ratio(words))
    features.append(fe.pronoun_ratio(words))
    features.append(fe.pronoun_ratio(words))
    features.append(fe.pronoun_ratio(words))
    features.append(fe.all2pron_ratio(words))
    features.append(fe.all3pron_ratio(words))
    # features.append(fe.passive_voice_ratio(words))
    features.append(fe.perceptual_ratio(words))
    features.append(fe.seeing_ratio(words))
    features.append(fe.feeling_ratio(words))
    features.append(fe.hearing_ratio(words))
    features.append(fe.time_ratio(words))
    features.append(fe.space_ratio(words))
    features.append(fe.timespace_ratio(words))
    features.append(fe.prep_ratio(words))
    features.append(fe.num_ratio(words))
    features.append(fe.quant_ratio(words))
    features.append(fe.motion_ratio(words))
    features.append(fe.cogproc_ratio(words))
    features.append(fe.insight_ratio(words))
    # ...调用feature_engineering中的特征提取函数

    return features

if __name__ == '__main__':
    df = pd.read_excel('clean_cut.xlsx', engine="openpyxl")
    words = df['words'].values.tolist()
    text = df['sents'].values.tolist()

    features = []
    for t, w in zip(text, words):
        feat = process(t, w)
        features.append(feat)

    feature_names = [
        'word_count',
        'word_diversity',
        'type_token_ratio',
        # 'verb_count',
        'sentence_count',
        'avg_sentence_length',
        'causation_ratio',
        # 'exclusive_ratio',
        # 'modality_ratio',
        'certainty_ratio',
        'negation_ratio',
        'negative_emotion_ratio',
        'negative_emotions_only',
        'anger',
        'anxiety',
        'sadness',
        'positive_emotion_ratio',
        'all_emotion_ratio',
        'pronoun_ratio',
        'pronoun_ratio',
        'pronoun_ratio',
        'all2pron_ratio',
        'all3pron_ratio',
        # 'passive_voice_ratio',
        'perceptual_ratio',
        'seeing_ratio',
        'feeling_ratio',
        'hearing_ratio',
        'time_ratio',
        'space_ratio',
        'timespace_ratio',
        'prep_ratio',
        'num_ratio',
        'quant_ratio',
        'motion_ratio',
        'cogproc_ratio',
        'insight_ratio'
    ]  # 定义特征名称

    # 构造DataFrame
    df_features = pd.DataFrame(features, columns=feature_names)

    # 合并特征
    df_all = pd.concat([df, df_features], axis=1)

    # 输出结果
    df_all.to_excel('out1.xlsx', index=False)