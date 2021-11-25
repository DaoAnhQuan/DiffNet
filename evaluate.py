'''
author: Peijie Sun
e-mail: sun.hfut@gmail.com
released date: 04/18/2019
'''

import math
import numpy as np

import conf


def get_idcg(length):
    idcg = 0.0
    for i in range(length):
        idcg = idcg + math.log(2) / math.log(i + 2)
    return idcg


def get_dcg(value):
    dcg = math.log(2) / math.log(value + 2)
    return dcg


def get_hr():
    hit = 1.0
    return hit


def evaluate_ranking_performance(evaluate_index_dict, evaluate_real_rating_matrix,
                                 evaluate_predict_rating_matrix):
    user_list = list(evaluate_index_dict.keys())
    batch_size = math.ceil(len(user_list)/conf.num_eval_procs)
    top_k = conf.topK

    hr_list, ndcg_list = [], []
    index = 0
    for _ in range(conf.num_eval_procs):
        if index + batch_size < len(user_list):
            batch_user_list = user_list[index:index + batch_size]
            index = index + batch_size
        else:
            batch_user_list = user_list[index:len(user_list)]
        tmp_hr_list, tmp_ndcg_list = get_hr_ndcg(evaluate_index_dict, evaluate_real_rating_matrix,
                                                 evaluate_predict_rating_matrix, top_k, batch_user_list)
        hr_list.extend(tmp_hr_list)
        ndcg_list.extend(tmp_ndcg_list)
    return np.mean(hr_list), np.mean(ndcg_list)


def get_hr_ndcg(evaluate_index_dict, evaluate_real_rating_matrix, evaluate_predict_rating_matrix, top_k,
                user_list):
    tmp_hr_list, tmp_ndcg_list = [], []
    for u in user_list:
        real_item_index_list = evaluate_index_dict[u]
        # print(evaluate_real_rating_matrix.shape)
        real_item_rating_list = list(np.concatenate(evaluate_real_rating_matrix[real_item_index_list]))
        positive_length = len(real_item_rating_list)
        target_length = min(positive_length, top_k)

        predict_rating_list = evaluate_predict_rating_matrix[u]
        real_item_rating_list.extend(predict_rating_list)
        sort_index = np.argsort(real_item_rating_list)
        sort_index = sort_index[::-1]

        user_hr_list = []
        user_ndcg_list = []
        hits_num = 0
        for idx in range(top_k):
            ranking = sort_index[idx]
            if ranking < positive_length:
                hits_num += 1
                user_hr_list.append(get_hr())
                user_ndcg_list.append(get_dcg(idx))

        idcg = get_idcg(target_length)
        # print(target_length)
        tmp_hr = np.sum(user_hr_list) / target_length
        tmp_ndcg = np.sum(user_ndcg_list) / idcg
        tmp_hr_list.append(tmp_hr)
        tmp_ndcg_list.append(tmp_ndcg)

    return tmp_hr_list, tmp_ndcg_list
