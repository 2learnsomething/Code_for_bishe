from turtle import color
from sympy import per
import torch
import time
import os
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from tqdm import tqdm
import sys

sys.path.append('.')
from preprocessing.price_preprocess import new_left_company
from preprocessing.price_preprocess import get_company_name_code
from utils import path

#图片存储路径
figure_path = '/new_python_for_gnn/毕设code/language_model_result/figure'

data_path = path.rsplit('/', 1)[0]

##############################
# 与utils.dl差不多，只有保存路径上的不同，为了减少文件调用，以及提高模块化性质，这里采取分开的做法
##############################


def train_model(model,
                train_iter,
                train_price_iter,
                optimizer,
                criterion,
                scheduler,
                n_epochs,
                model_name,
                num_iter,
                is_clip=False):
    """对模型进行训练并且保存模型

    Args:
        model (_type_): 模型
        train_iter (_type_): 训练集
        optimizer (_type_): 优化器
        criterion (_type_): 损失函数
        n_epochs (int): 训练轮数
        model_name (str): 模型名称
    """
    device = torch.device('cuda:0')
    model.to(device)
    #model.cuda()
    model.train()
    total_step = len(train_iter)
    print("            =======  Training  ======= \n")
    train_loss_ = []  #一个epoch的训练误差
    correct_ = []  #一个epoch的精度
    accuracy_ = []
    for epoch in range(n_epochs):
        train_loss = 0
        correct = 0
        total = 0
        time_strat = time.time()
        for i, (batch, price) in enumerate(zip(train_iter, train_price_iter)):
            data, labels = batch.news, batch.label
            period = batch.news_period
            # labels = labels.long()
            # period = period.float()
            # price = price.float()
            #print(period)
            data = data.cuda()
            labels = labels.cuda().long()
            period = period.cuda().float()
            price = price.cuda().float()
            # Forward
            outputs = model(data, period, price)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            if is_clip:
                torch.nn.utils.clip_grad_value_(
                    model.parameters(), clip_value=1000)  #clip_value还需要调参.
            optimizer.step()

            #train_loss += loss.item()
            train_loss += float(loss)
            total += labels.size(0)
            correct += torch.eq(outputs.argmax(dim=1), labels).sum().item()
            accuracy = 100.0 * correct / total

            if (i + 1) % num_iter == 0 or (i + 1) == total_step:
                time_end = time.time()
                print(
                    'Epoch: [{:3}/{}], Step: [{:3}/{}], Ave_Loss: {:.3f}, acc: {:6.3f}, time: {:.3f}s'
                    .format(epoch + 1, n_epochs, i + 1, total_step,
                            train_loss / (i + 1), accuracy,
                            time_end - time_strat))

        scheduler.step(train_loss)
        train_loss_.append(train_loss)
        correct_.append(correct)
        accuracy_.append(accuracy)
        #保存断点
        if (epoch + 1) % 5 == 0:
            print('checkpoint saving...')
            save_checkpoint(model, optimizer, epoch, model_name)
    print("\n           =======  Training Finished  ======= \n")
    #先简单可视化一下准确率
    plt.figure()
    plt.rcParams["font.sans-serif"] = ["SimHei"]  #设置字体
    plt.rcParams["axes.unicode_minus"] = False  #该语句解决图像中的“-”负号的乱码问题
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.plot(accuracy_, 'b*')
    #plt.show()
    #保存损失便于进一步可视化
    print('loss saving...')
    with open(
            os.path.join(data_path, 'dgl_model_result/train',
                         model_name + '_loss.json'), 'wb') as f:
        loss_file = OrderedDict()
        loss_file['train_loss'] = train_loss_
        loss_file['correct'] = correct_
        loss_file['accuracy'] = accuracy_
        json_cont = json.dumps(loss_file, indent=4).encode(encoding='utf-8')
        f.write(json_cont)
    print('success for loss saving...')
    #保存模型方便之后进行模型测试等操作
    print("Model saving...")
    PATH = os.path.join(data_path, 'model_cache', model_name + '.pt')
    if not os.path.exists(PATH):
        torch.save(model.state_dict(), PATH)
    print("Model saved...")

    #return train_loss_,correct_,accuracy_,model_name


def test_model(model, model_path, test_iter, test_price_iter, model_name,
               num_iter):
    """对模型进行测试集的测试

    Args:
        model (): 模型
        model_path (str): 训练之后的模型保存的路径
        test_iter (_type_): 测试集
    """
    device = torch.device('cuda:0')
    model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print("\n            =======  Testing  ======= \n")
    y_true = []
    y_pred = []
    test_len = len(test_iter)
    with torch.no_grad():
        for i, (batch, price) in enumerate(zip(test_iter, test_price_iter)):
            batch_xs = batch.news
            batch_ys = batch.label
            batch_zs = batch.news_period
            batch_xs, batch_ys = batch_xs.cuda(), batch_ys.cuda().long()
            batch_zs = batch_zs.cuda().float()
            price = price.cuda().float()

            batch_out = model(batch_xs, batch_zs, price)
            batch_pred = batch_out.argmax(dim=-1)

            y_true.extend(batch_ys.cpu().numpy())
            y_pred.extend(batch_pred.cpu().numpy())

            if (i + 1) % num_iter == 0 or i == test_len:
                accuracy = accuracy_score(y_true, y_pred)
                macro_f1 = f1_score(y_true, y_pred, average="macro")
                print(
                    "test accuracy {:.3f}, test macro f1-score {:.3f}".format(
                        accuracy, macro_f1))
    print("\n          =======  Testing Finished  ======= \n")
    print('calculating classification index...')
    confusion_matrix_ = confusion_matrix(y_true, y_pred)
    cm_type = 'test_result'
    test_result = (y_true, y_pred)
    classification_result(confusion_matrix_, cm_type, test_result, model_name)
    print('finished saving index file...')


#仿utils_ml
def classification_result(confusion_matrix, cm_type, test_result, model_name):
    """对训练结果进行分析输出和保存

    Args:
        confusion_matrix (tuple): 混淆矩阵，包括训练的和测试的
        cm_type (list): 混淆矩阵的类型，是训练的结果还是测试的结果
    """
    #解释一下参数的具体内容
    # confusion_matrix = test_matrix
    # cm_type = 'test_result'
    # train_result = (train_y, pred_train,pred_prob_train)
    # test_result = (test_y,pred_test,preed_prob_test)

    cm_info = OrderedDict()
    print(cm_type + '的混淆矩阵为:\n')
    print(confusion_matrix)
    cm_info[cm_type] = confusion_matrix.tolist()
    TN = confusion_matrix[0][0]
    FN = confusion_matrix[1][0]
    TP = confusion_matrix[1][1]
    FP = confusion_matrix[0][1]
    #array类型取出具体的数值用item,我记得tensor也是
    cm_info['TN'] = TN.item()
    cm_info['FN'] = FN.item()
    cm_info['TP'] = TP.item()
    cm_info['FP'] = FP.item()
    #二级指标
    #准确率（Accuracy）(已完成)
    accuracy = accuracy_score(test_result[0], test_result[1])
    cm_info['accuracy'] = accuracy
    print('准确率为' + str(accuracy))
    #精确率（Precision）——查准率（已完成）
    precision = precision_score(test_result[0], test_result[1])
    cm_info['precision'] = precision
    print('精确率为' + str(precision))
    #查全率、召回率、反馈率（Recall）(已完成)
    recall = recall_score(test_result[0], test_result[1])
    cm_info['recall'] = recall
    print('召回率为' + str(recall))
    #特异度（Specificity）(已完成)
    TNR = TN / (TN + FP)
    cm_info['TNR'] = TNR
    print('特异度为' + str(TNR))
    #FPR（假警报率）(已完成)
    FPR = FP / (FP + TN)
    cm_info['FPR'] = FPR
    print('假报警率为' + str(FPR))
    #TPR（真正率）(已完成)
    TPR = TP / (TP + FN)
    cm_info['TPR'] = TPR
    print('真正率为' + str(TPR))
    #三级指标
    #F1_score(已完成)
    f1 = f1_score(test_result[0], test_result[1])
    cm_info['f1'] = f1
    print('f1分数为' + str(f1))
    #G-mean(在数据不平衡的时候,这个指标很有参考价值。)
    g_mean = np.sqrt(recall * TNR)
    cm_info['g_mean'] = g_mean
    print('g_mean值为' + str(g_mean))
    json_data = json.dumps(cm_info, indent=4)
    with open(
            os.path.join(data_path, 'dgl_model_result/test',
                         model_name + '.json'), 'w') as f:
        f.write(json_data)
    f.close
    print('成功保存测试结果！')


def get_parameter_number(model):
    """获取模型参数量

    Args:
        model (_type_): pytorch模型

    Returns:
        ditc: 总参数量和可训练参数量的字典
    """
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters()
                        if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def save_checkpoint(model, optimizer, epoch, model_name):
    """保存checkpoint

    Args:
        model (_type_): _description_
        optimizer (_type_): _description_
        epoch (_type_): _description_
        model_name (_type_): _description_
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch
    }
    path_checkpoint = "/new_python_for_gnn/毕设code/model_cache/{}_checkpoint.pkl".format(
        model_name)
    torch.save(checkpoint, path_checkpoint)