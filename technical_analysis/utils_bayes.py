import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_curve, auc
from collections import OrderedDict

#训练结果保存
train_path = '/new_python_for_gnn/毕设code/technical_model_result/train'
#测试结果保存
test_path = '/new_python_for_gnn/毕设code/technical_model_result/test'
#保存路径
result_path = [train_path, test_path]


def process_x(processing_type, x):
    """对x数据进行归一化

    Args:
        processing_type (str): 处理类型或者说方法
        x (ndarray): 数据
    """
    if processing_type == 'minmax':
        from sklearn.preprocessing import MinMaxScaler
        mm = MinMaxScaler()
        x = mm.fit_transform(x)
    elif processing_type == 'stand':
        from sklearn.preprocessing import StandardScaler
        ss = StandardScaler()
        x = ss.fit_transform(x)
    elif processing_type == 'maxabs':
        from sklearn.preprocessing import MaxAbsScaler
        ma = MaxAbsScaler()
        x = ma.fit_transform(x)
    return x


###开始训练与预测
def train_pre(model, train_x, test_x, train_y, test_y):
    """对模型进行训练,得到一些指标数据,这里追求全面

    Args:
        model (model): 模型
        train_x (array): 训练数据
        test_x (array): 测试数据
        train_y (array): 训练特征
        test_y (array): 测试特征
    """
    #模型训练
    model.fit(train_x, train_y)

    is_pred_prob = hasattr(model, 'predict_prob')  #判断是否有预测分类概率的函数
    #注下面的特征重要性似乎只是随机森林算法才有，其他模型没有，这个需要注意
    #print('特征的重要性:{}'.format(model.best_estimator_.feature_importance_))

    pred_train = model.predict(train_x)  # 获取标签
    accuracy1 = accuracy_score(train_y, pred_train)
    if is_pred_prob:
        pred_prob_train = model.predict_prob(train_x)  #获取概率
        pred_prob_train = np.max(pred_prob_train, axis=1)
    else:
        pred_prob_train = np.array([1, 2, 3])  #这里保证有输出，但是不会用到
    print('在训练集上的精确度: %.4f' % accuracy1)

    #m模型测试
    pred_test = model.predict(test_x)
    accuracy2 = accuracy_score(test_y, pred_test)
    if is_pred_prob:
        pred_prob_test = model.predict_prob(test_x)
        pred_prob_test = np.max(pred_prob_test, axis=1)
    else:
        pred_prob_test = np.array([4, 5, 6])  #同上解释
    print('在测试集上的精确度: %.4f' % accuracy2)

    return (confusion_matrix(train_y, pred_train),
            confusion_matrix(test_y,
                             pred_test)), (train_y, pred_train,
                                           pred_prob_train,
                                           is_pred_prob), (test_y, pred_test,
                                                           pred_prob_test,
                                                           is_pred_prob)


###可视化,参考https://blog.csdn.net/Monk_donot_know/article/details/86614558
def classification_result(confusion_matrix, cm_type, train_result, test_result,
                          model_name):
    """对训练结果进行分析输出和保存

    Args:
        confusion_matrix (tuple): 混淆矩阵，包括训练的和测试的
        cm_type (list): 混淆矩阵的类型，是训练的结果还是测试的结果
        train_result (tuple): 训练模型之后对训练集的预测结果和真实结果
        test_result (tuple): 训练模型之后对测试集的预测结果和真实结果
    """
    #解释一下参数的具体内容
    # confusion_matrix = (train_matrix,test_matrix)
    # cm_type = ['train_result','test_result']
    # train_result = (train_y, pred_train,pred_prob_train)
    # test_result = (test_y,pred_test,preed_prob_test)
    # 以下我自己知道实现的比较复杂，可以直接一个if，else解决，但是我觉得这样更加直观。
    for index, cm in enumerate(confusion_matrix):
        cm_info = OrderedDict()
        # cm才是混淆矩阵
        # 混淆矩阵
        print(cm_type[index] + '的混淆矩阵为:\n')
        print(cm)
        #print('cm的类型为:\n')
        #print(type(cm))
        #注意json不能保存numpy的array结果
        #这里采取转化为list保存
        cm_info[cm_type[index]] = cm.tolist()
        # TP:True Posirive:正确的肯定的分类数
        # TN:True Negatives:正确的否定的分类数
        # FP:False Positive:错误的肯定的分类数
        # FN:False Negatives:错误的否定
        TN = cm[0][0]
        FN = cm[1][0]
        TP = cm[1][1]
        FP = cm[0][1]
        #array类型取出具体的数值用item,我记得tensor也是
        cm_info['TN'] = TN.item()
        cm_info['FN'] = FN.item()
        cm_info['TP'] = TP.item()
        cm_info['FP'] = FP.item()
        #二级指标
        #准确率（Accuracy）(已完成)
        if index == 0:
            accuracy = accuracy_score(train_result[0], train_result[1])
            cm_info['accuracy'] = accuracy
        else:
            accuracy = accuracy_score(test_result[0], test_result[1])
            cm_info['accuracy'] = accuracy
        print('准确率为' + str(accuracy))
        #精确率（Precision）——查准率（已完成）
        if index == 0:
            precision = precision_score(train_result[0], train_result[1])
            cm_info['precision'] = precision
        else:
            precision = precision_score(test_result[0], test_result[1])
            cm_info['precision'] = precision
        print('精确率为' + str(precision))
        #查全率、召回率、反馈率（Recall）(已完成)
        if index == 0:
            recall = recall_score(train_result[0], train_result[1])
            cm_info['recall'] = recall
        else:
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
        if index == 0:
            f1 = f1_score(train_result[0], train_result[1])
            cm_info['f1'] = f1
        else:
            f1 = f1_score(test_result[0], test_result[1])
            cm_info['f1'] = f1
        print('f1分数为' + str(f1))
        #G-mean(在数据不平衡的时候,这个指标很有参考价值。)
        g_mean = np.sqrt(recall * TNR)
        cm_info['g_mean'] = g_mean
        print('g_mean值为' + str(g_mean))

        json_data = json.dumps(cm_info, indent=4)
        if index == 0:
            with open(
                    os.path.join(result_path[0],
                                 model_name + '_' + cm_type[0] + '_2tag.json'),
                    'w') as f:
                f.write(json_data)
            f.close
            print('成功保存训练结果！')
        else:
            with open(
                    os.path.join(result_path[1],
                                 model_name + '_' + cm_type[1] + '_2tag.json'),
                    'w') as f:
                f.write(json_data)
            f.close
            print('成功保存测试结果！')


def plot_ROC(labels, preds, savepath):
    """
    Args:
        labels : ground truth
        preds : model prediction
        savepath : save path 
    """
    #ROC曲线、Auc值(已完成)
    fpr, tpr, threshold = roc_curve(labels, preds, pos_label=2)  ###计算真正率和假正率

    roc_auc = auc(fpr, tpr)  ###计算auc的值，auc就是曲线包围的面积，越大越好
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr,
             tpr,
             color='darkorange',
             lw=lw,
             label='AUC = %0.2f' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    #plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.title('ROCs for Decision Tree')
    plt.legend(loc="lower right")
    plt.show()
    plt.savefig(savepath)  #保存文件