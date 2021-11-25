import conf
import numpy as np
from collections import defaultdict
import torch


def get_user_features():
    user_features = np.load(conf.user_feature_file)
    # mean = np.reshape(np.mean(user_features, axis=1), (-1, 1))
    # var = np.reshape(np.var(user_features, axis=1), (-1, 1))
    # user_features = (user_features - mean) * 0.2 / np.sqrt(var)
    return user_features


def get_item_features():
    item_features = np.load(conf.item_feature_file)
    # mean = np.reshape(np.mean(item_features, axis=1), (-1, 1))
    # var = np.reshape(np.var(item_features, axis=1), (-1, 1))
    # item_features = (item_features - mean) * 0.2 / np.sqrt(var)
    return item_features


def get_social_network_sparse_matrix():
    edges = open(conf.social_network_file, 'r')
    social_neighbors = defaultdict(set)
    for edge in edges:
        tmp = edge.split('\t')
        u1, u2 = int(tmp[0]), int(tmp[1])
        social_neighbors[u1].add(u2)
        social_neighbors[u2].add(u1)
    social_neighbors_indices_list = []
    social_neighbors_values_list = []
    social_neighbors_dict = defaultdict(list)
    for u in social_neighbors:
        social_neighbors_dict[u] = sorted(social_neighbors[u])

    user_list = sorted(list(social_neighbors.keys()))
    for user in user_list:
        for friend in social_neighbors_dict[user]:
            social_neighbors_indices_list.append([user, friend])
            social_neighbors_values_list.append(1.0 / len(social_neighbors_dict[user]))
    social_neighbors_indices_list = torch.tensor(social_neighbors_indices_list, dtype=torch.int32)
    social_neighbors_values_list = torch.tensor(social_neighbors_values_list, dtype=torch.float32)
    return social_neighbors_indices_list, social_neighbors_values_list


class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.total_user_list, self.hash_data = self.read_data()
        self.positive_data, self.total_data = self.arrange_positive_data()
        self.cursor = 0
        self.terminal_flag = 1
        self.negative_data = {}
        self.user_list = []
        self.item_list = []
        self.label_list = []
        self.index_dict = {}
        self.eva_negative_data = {}

    def init_per_epoch(self):
        self.cursor = 0
        self.terminal_flag = 1
        self.negative_data = self.get_negative_sample()

    def init_eval(self):
        self.eva_negative_data = self.generate_eva_negative()

    def read_data(self):
        f = open(self.file_path)
        total_user_list = set()
        hash_data = defaultdict(int)
        for _, line in enumerate(f):
            arr = line.split("\t")
            hash_data[(int(arr[0]), int(arr[1]))] = 1
            total_user_list.add(int(arr[0]))
        return list(total_user_list), hash_data

    def arrange_positive_data(self):
        positive_data = defaultdict(set)
        total_data = set()
        hash_data = self.hash_data
        for (u, i) in hash_data:
            total_data.add((u, i))
            positive_data[u].add(i)
        return positive_data, len(total_data)

    def get_user_item_sparse_matrix(self):
        positive_data = self.positive_data
        consumed_items_indices_list = []
        consumed_items_values_list = []
        consumed_items_dict = defaultdict(list)
        for u in positive_data:
            consumed_items_dict[u] = sorted(positive_data[u])
        user_list = sorted(list(positive_data.keys()))
        for u in user_list:
            for i in consumed_items_dict[u]:
                consumed_items_indices_list.append([u, i])
                consumed_items_values_list.append(1.0 / len(consumed_items_dict[u]))
        consumed_items_indices_list = torch.tensor(consumed_items_indices_list, dtype=torch.int32)
        consumed_items_values_list = torch.tensor(consumed_items_values_list, dtype=torch.float32)
        return consumed_items_indices_list, consumed_items_values_list

    def get_negative_sample(self):
        num_items = conf.num_items
        num_negatives = conf.num_negative_samples
        negative_data = defaultdict(set)
        total_data = set()
        hash_data = self.hash_data
        for (u, i) in hash_data:
            total_data.add((u, i))
            for _ in range(num_negatives):
                j = np.random.randint(num_items)
                while (u, j) in hash_data:
                    j = np.random.randint(num_items)
                negative_data[u].add(j)
                total_data.add((u, j))
        return negative_data

    def get_batch_data(self):
        positive_data = self.positive_data
        negative_data = self.negative_data
        total_user_list = self.total_user_list
        index = self.cursor
        batch_size = conf.batch_size

        user_list, item_list, labels_list = [], [], []
        if index + batch_size < len(total_user_list):
            target_user_list = total_user_list[index:index + batch_size]
            self.cursor = index + batch_size
        else:
            target_user_list = total_user_list[index:len(total_user_list)]
            self.cursor = 0
            self.terminal_flag = 0

        for u in target_user_list:
            user_list.extend([u] * len(positive_data[u]))
            item_list.extend(list(positive_data[u]))
            labels_list.extend([1] * len(positive_data[u]))
            user_list.extend([u] * len(negative_data[u]))
            item_list.extend(list(negative_data[u]))
            labels_list.extend([0] * len(negative_data[u]))

        self.user_list = user_list
        self.item_list = item_list
        self.label_list = labels_list

    def get_all_data(self):
        positive_data = self.positive_data
        negative_data = self.negative_data
        total_user_list = self.total_user_list
        user_list = []
        item_list = []
        labels_list = []
        for u in total_user_list:
            user_list.extend([u] * len(positive_data[u]))
            item_list.extend(positive_data[u])
            labels_list.extend([1] * len(positive_data[u]))
            user_list.extend([u] * len(negative_data[u]))
            item_list.extend(negative_data[u])
            labels_list.extend([0] * len(negative_data[u]))
        self.user_list = user_list
        self.item_list = item_list
        self.label_list = labels_list

    def get_eva_positive_data(self):
        hash_data = self.hash_data
        user_list = []
        item_list = []
        index_dict = defaultdict(list)
        index = 0
        for (u, i) in hash_data:
            user_list.append(u)
            item_list.append(i)
            index_dict[u].append(index)
            index = index + 1
        self.user_list = user_list
        self.item_list = item_list
        self.index_dict = index_dict

    def generate_eva_negative(self):
        num_evaluate = conf.num_negative_evaluate
        total_user_list = self.total_user_list
        num_items = conf.num_items
        eva_negative_data = defaultdict(list)
        for u in total_user_list:
            for _ in range(num_evaluate):
                j = np.random.randint(num_items)
                while (u, j) in self.hash_data:
                    j = np.random.randint(num_items)
                eva_negative_data[u].append(j)
        return eva_negative_data

    def get_eva_ranking_batch(self):
        batch_size = conf.batch_size
        num_evaluate = conf.num_negative_evaluate
        eva_negative_data = self.eva_negative_data
        total_user_list = self.total_user_list
        index = self.cursor
        terminal_flag = 1
        total_users = len(total_user_list)
        user_list = []
        item_list = []
        if index + batch_size < total_users:
            batch_user_list = total_user_list[index:index + batch_size]
            self.cursor = index + batch_size
        else:
            terminal_flag = 0
            batch_user_list = total_user_list[index:total_users]
            self.cursor = 0
        for u in batch_user_list:
            user_list.extend([u] * num_evaluate)
            item_list.extend(eva_negative_data[u])
        self.user_list = user_list
        self.item_list = item_list
        return batch_user_list, terminal_flag
