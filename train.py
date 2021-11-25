import numpy as np
import torch.nn

import dataloader
from diffnet import DiffNet
import conf
import evaluate

criterion = torch.nn.MSELoss()
model = DiffNet()
optimizer = torch.optim.Adam(model.parameters(), lr=conf.learning_rate)  # Define optimizer


def train(data: dataloader.DataLoader):
    optimizer.zero_grad()
    _, predictions = model(data)
    loss = criterion(predictions, torch.tensor(data.label_list, dtype=torch.float32))
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss


if __name__ == '__main__':
    train_data = dataloader.DataLoader(conf.train_file)
    val_data = dataloader.DataLoader(conf.val_file)
    test_data = dataloader.DataLoader(conf.test_file)
    eval_data = dataloader.DataLoader(conf.test_file)
    eval_data.init_eval()
    for epoch in range(conf.num_epoches):
        train_data.init_per_epoch()
        tmp_train_loss = []
        while train_data.terminal_flag == 1:
            train_data.get_batch_data()
            # print(train_data.user_list[:5])
            # print(train_data.item_list[:5])
            # print(train_data.label_list[:5])
            tmp_loss = train(train_data)
            tmp_train_loss.append(tmp_loss.tolist())
        train_loss = float(np.mean(tmp_train_loss))
        val_data.init_per_epoch()
        val_data.get_all_data()
        _, val_predictions = model(val_data)
        val_loss = criterion(val_predictions, torch.tensor(val_data.label_list, dtype=torch.float32))
        test_data = dataloader.DataLoader(conf.test_file)
        test_data.init_per_epoch()
        test_data.get_all_data()
        _, test_predictions = model(test_data)
        test_loss = criterion(test_predictions, torch.tensor(test_data.label_list, dtype=torch.float32))
        eval_data.get_eva_positive_data()
        index_dict = eval_data.index_dict
        _, eval_positive_predictions = model(eval_data)
        eva_negative_predictions = {}
        terminal_flag = 1
        t = 0
        while terminal_flag == 1:
            batch_user_list, terminal_flag = eval_data.get_eva_ranking_batch()
            index = 0
            _, tmp_negative_predictions = model(eval_data)
            tmp_negative_predictions = np.reshape(
                np.array(tmp_negative_predictions.tolist(), dtype=np.float32)
                ,
                [-1, conf.num_negative_evaluate])
            t = tmp_negative_predictions[0][:10]
            for u in batch_user_list:
                eva_negative_predictions[u] = tmp_negative_predictions[index]
                index = index + 1

        tmp1 = np.array(eval_positive_predictions.tolist(), dtype=np.float32)
        tmp2 = np.reshape(tmp1, [-1, 1])
        # print("positive:")
        # print(tmp2[:10])
        # print('negative')
        # print(t)
        hr, ndcg = evaluate.evaluate_ranking_performance(index_dict,
                                                         tmp2,
                                                         eva_negative_predictions)

        print('Epoch: %d, train loss: %.4f, val loss: %.4f, test loss: %.4f, hr: %.4f, ndcg: %.4f' % (
            epoch, train_loss, val_loss.tolist(), test_loss.tolist(), hr, ndcg))
