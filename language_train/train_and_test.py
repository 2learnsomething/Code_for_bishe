import torch
import os
from sklearn import metrics
import sys

sys.path.append('.')
from utils import path

data_path = path.rsplit('/', 1)[0]


def train_model(model, train_iter, optimizer, criterion, n_epochs, model_name):
    """对模型进行训练并且保存模型

    Args:
        model (_type_): 模型
        train_iter (_type_): 训练集
        optimizer (_type_): 优化器
        criterion (_type_): 损失函数
        n_epochs (int): 训练轮数
        model_name (str): 模型名称
    """
    model.train()
    model = torch.nn.DataParallel(model)
    total_step = len(train_iter)
    print("            =======  Training  ======= \n")
    for epoch in range(n_epochs):
        train_loss = correct = total = 0
        for i, batch in enumerate(train_iter):
            data, labels = batch.news, batch.label
            data = data.cuda()  # torch.size([400,8])
            labels = labels.cuda()
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #train_loss += loss.item()
            train_loss += float(loss)
            total += labels.size(0)
            correct += torch.eq(outputs.argmax(dim=1), labels).sum().item()

            if (i + 1) % 25 == 0 or (i + 1) == total_step:
                print(
                    'Epoch: [{:3}/{}], Step: [{:3}/{}], Loss: {:.3f}, acc: {:6.3f}'
                    .format(
                        epoch + 1,
                        n_epochs,
                        i + 1,
                        total_step,
                        train_loss / (i + 1),
                        100.0 * correct / total,
                    ))
    print("\n            =======  Training Finished  ======= \n")
    print("\nModel saving... \n")
    PATH = os.path.join(data_path, 'model_cache', model_name + '.pt')
    if not os.path.exists(PATH):
        torch.save(model.state_dict(), PATH)
    print("\nModel saved... \n")


def test_model(model, model_path, test_iter):
    """对模型进行测试集的测试

    Args:
        model (): 模型
        model_path (str): 训练之后的模型保存的路径
        test_iter (_type_): 测试集
    """
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print("\n            =======  Testing  ======= \n")
    y_true = []
    y_pred = []
    for batch in test_iter:
        batch_xs = batch.news  # (batch_size, max_len)
        batch_ys = batch.label  # (batch_size, )
        batch_xs, batch_ys = batch_xs.cuda(), batch_ys.cuda()  #改了

        batch_out = model(batch_xs)  # (batch_size, num_classes)
        batch_pred = batch_out.argmax(dim=-1)

        y_true.extend(batch_ys.cpu().numpy())
        y_pred.extend(batch_pred.cpu().numpy())

    accuracy = metrics.accuracy_score(y_true, y_pred)
    macro_f1 = metrics.f1_score(y_true, y_pred, average="macro")
    print("test accuracy {:.3f}, test macro f1-score {:.3f}".format(
        accuracy, macro_f1))
    print("\n            =======  Testing Finished  ======= \n")
