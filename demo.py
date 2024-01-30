import pandas as pd
import feature_engineering as fe
import numpy as np


def parallel_extract(df_split, features):
    df_feats = pd.DataFrame()

    for feature in features:

        if feature in ['sentence_count', 'avg_sentence_len']:
            df_split[feature] = df_split['text'].apply(getattr(fe, feature))
            print(f"Applying function {feature} to data of type {type(df_split['text'].iloc[0])}")
        else:
            df_split[feature] = df_split['words'].apply(getattr(fe, feature))
            print(f"Applying function {feature} to data of type {type(df_split['words'].iloc[0])}")

        df_feats = pd.concat([df_feats, df_split[[feature]]], axis=1)

    return df_feats

def convert_str_to_list(row):
    row['words'] = eval(row['words'])
    return row

if __name__ == '__main__':

    df = pd.read_excel('clean_cut.xlsx')
    df = df.apply(convert_str_to_list, axis=1)

    # 提取文本和词语列
    text = df['text']
    words = df['words']

    # 定义要提取的特征
    features = [
    'word_count',
    'word_diversity',
    'type_token_ratio',
    'verb_count',
    'sentence_count',
    'avg_sentence_length',
    'causation_ratio',
    'exclusive_ratio',
    'modality_ratio',
    'certainty_ratio',
    'negation_ratio',
    'negative_emotion_ratio',
    'negative_emotions_only',
    'anger',
    'anxiety',
    'sadness',
    'positive_emotion_ratio',
    'all_emotion_ratio',
    'pleasantness_ratio',
    'pronoun_ratio',
    'fp_sp_ratio',
    'fp_pp_ratio',
    'all_sp_ratio',
    'all_tp_ratio',
    'passive_ratio',
    'generalization_ratio',
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
    'modifier_ratio',
    'motion_ratio',
    'cogproc_ratio',
    'insight_ratio']

df_f = parallel_extract(df_split=df, features=features)
print(df_f)
df_f.to_excel('demo.xlsx')