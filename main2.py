import pandas as pd
import feature_engineering as fe
import numpy as np
from multiprocessing import Pool
import logging

logger = logging.getLogger()


def parallel_extract(df_split, features):
        df_feats = pd.DataFrame()

        for feature in features:
                logger.info(f"Extracting {feature}")

                if feature in ['sentence_count', 'avg_sentence_len']:
                        df_split[feature] = df_split['text'].apply(getattr(fe, feature))
                else:
                        df_split[feature] = df_split['words'].apply(getattr(fe, feature))

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
        # 使用map批量应用特征函数
        # df[features] = df[features].applymap(fe.func_map)

        # 如果有大量特征,可考虑先分批并行提取特征列
        # 定义分批函数

        # 分批并行提取特征
        df_list = np.array_split(df, 4)
        pool = Pool(4)
        df_features = pd.concat(pool.starmap(parallel_extract, [(df_split, features) for df_split in df_list]))
        pool.close()
        pool.join()

        # 合并到原DataFrame
        df_final = df.join(df_features)

        # 输出结果
        df_final.to_excel('out_5.xlsx', index=False)