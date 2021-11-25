import torch
import dataloader
import conf
from torchvision import transforms


class DiffNet(torch.nn.Module):
    def __init__(self):
        super(DiffNet, self).__init__()
        self.user_embedding = torch.normal(mean=0, std=0.01, size=(conf.num_users, 32), dtype=torch.float32,
                                           requires_grad=True)
        self.item_embedding = torch.normal(mean=0, std=0.01, size=(conf.num_items, 32), dtype=torch.float32,
                                           requires_grad=True)
        self.user_features = torch.tensor(dataloader.get_user_features(), dtype=torch.float32)
        self.user_features = self.normalize_features(self.user_features)
        self.item_features = torch.tensor(dataloader.get_item_features(), dtype=torch.float32)
        self.item_features = self.normalize_features(self.item_features)
        social_neighbors_indices_list, social_neighbors_values_list = dataloader.get_social_network_sparse_matrix()
        self.social_tensor = torch.sparse_coo_tensor(social_neighbors_indices_list.t(), social_neighbors_values_list,
                                                     (conf.num_users, conf.num_users))

        self.wr = torch.nn.Linear(conf.num_features, conf.num_dimensions, dtype=torch.float32)
        # self.wi = torch.nn.Linear(conf.num_features + conf.num_dimensions, conf.num_dimensions, dtype=torch.float32)
        # self.w1 = torch.nn.Linear(2 * conf.num_dimensions, conf.num_dimensions, dtype=torch.float32)
        # self.w2 = torch.nn.Linear(2 * conf.num_dimensions, conf.num_dimensions, dtype=torch.float32)

    def normalize_features(self, tensor):
        mean = tensor.mean()
        std = tensor.std()
        features = (tensor - mean) / std
        return features

    def forward(self, data: dataloader.DataLoader):
        consumed_items_indices_list, consumed_items_values_list = data.get_user_item_sparse_matrix()
        user_item_tensor = torch.sparse_coo_tensor(consumed_items_indices_list.t(), consumed_items_values_list,
                                                   (conf.num_users, conf.num_items))
        user_reduce = self.wr(self.user_features)
        user_reduce = torch.nn.Sigmoid()(user_reduce)
        user_reduce = self.normalize_features(user_reduce)
        user_fusion = user_reduce + self.user_embedding
        item_reduce = self.wr(self.item_features)
        item_reduce = torch.nn.Sigmoid()(item_reduce)
        item_reduce = self.normalize_features(item_reduce)
        item_fusion = self.item_embedding + item_reduce
        item_consumed = torch.matmul(user_item_tensor, item_fusion)
        h1 = torch.matmul(self.social_tensor, user_fusion)
        h2 = torch.matmul(self.social_tensor, h1)
        last_user_embedding = item_consumed + h2
        user_indices = torch.LongTensor(data.user_list)
        item_indices = torch.LongTensor(data.item_list)
        user_embedding_matrix = last_user_embedding.index_select(index=user_indices, dim=0)
        item_embedding_matrix = item_fusion.index_select(index=item_indices, dim=0)
        predict_vector = torch.multiply(user_embedding_matrix, item_embedding_matrix)
        predict_vector = torch.sum(predict_vector, dim=1)
        predictions = torch.sigmoid(predict_vector)
        return predict_vector, predictions
