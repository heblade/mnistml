import time
import numpy as np
import pandas as pd
import sklearn.svm as svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib as mpl
import matplotlib.pyplot as plt


def startjob():
    #随机森林
    classifier_type = 'SVM'
    #SVM
    # classifier_type = 'SVM'
    print('载入训练数据...')
    t = time.time()
    data = pd.read_csv('./data/mnist_train.csv', header=None, dtype=np.int)
    print('载入完成，耗时%f秒' % (time.time() - t))
    #数据集第一列是代表的数字，之后28x28列的数字，代表数字各个点的黑度
    x, y = data.iloc[:, 1:].values, data.iloc[:, 0].values

    # x_train, x_valid, y_train, y_valid = train_test_split(x, y, train_size=0.8, random_state=1)
    print('训练集shape: ',x.shape)
    # print(x_valid.shape)
    print('图片个数: %d, 图片像素数目: %d' % (x.shape))

    print('载入测试数据')
    t = time.time()
    data_test = pd.read_csv('./data/mnist_test.csv', header=None, dtype=np.int)
    x_test, y_test = data_test.iloc[:, 1:].values, data_test.iloc[:, 0].values
    print('训练集shape: ', x_test.shape)
    print('载入完成，耗时%f秒' % (time.time() - t))

    mpl.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(15, 9), facecolor='w')
    cmap = plt.cm.gray_r
    #从训练集中获取16张图片
    for index in range(16):
        image = x[index]
        plt.subplot(4, 8, index + 1)
        plt.imshow(image.reshape(28, 28), cmap=cmap, interpolation='nearest')
        plt.title('训练图片: %i' % y[index])
    #从测试集中获取16张图片
    for index in range(16):
        image = x_test[index]
        plt.subplot(4, 8, index + 17)
        plt.imshow(image.reshape(28, 28), cmap=cmap, interpolation='nearest')
        plt.title('测试图片: %i' % y_test[index])
    plt.tight_layout()
    plt.show()

    if classifier_type == 'SVM':
        model = svm.SVC(C=1000, kernel='rbf', gamma=1e-10)
        print('SVM开始训练...')
    else:
        model = RandomForestClassifier(100,
                                       criterion='gini',
                                       min_samples_split=2,
                                       min_impurity_split=1e-10)
        print('随机森林开始训练...')
    t = time.time()
    # model.fit(x_train, y_train)
    model.fit(x, y)
    t = time.time() - t
    print('%s 训练结束, 耗时%d分钟%.3f秒' % (classifier_type,
                                    int(t/60),
                                    t - 60 * int(t/60)))
    t = time.time()
    y_train_pred = model.predict(x)
    t = time.time() - t
    print('%s训练集准确率: %.3f%%, 耗时%d分%.3f秒' % (classifier_type,
                                            accuracy_score(y, y_train_pred) * 100,
                                            int(t/60),
                                            t - 60 * int(t/60)))

    t = time.time()
    y_test_pred = model.predict(x_test)
    t = time.time() - t
    print('%s测试集准确率: %.3f%%, 耗时%d分%.3f秒' % (classifier_type,
                                            accuracy_score(y_test, y_test_pred) * 100,
                                            int(t / 60),
                                            t - 60 * int(t / 60)))

    err = (y_test != y_test_pred)
    err_images = x_test[err]
    err_y_hat = y_test_pred[err]
    err_y = y_test[err]
    print(err_y_hat)
    print(err_y)
    plt.figure(figsize=(10, 8), facecolor='w')
    for index in range(16):
        image = err_images[index]
        plt.subplot(4, 4, index + 1)
        plt.imshow(image.reshape(28, 28), cmap=cmap, interpolation='nearest')
        plt.title('错分为: %i, 真实值: %i' % (err_y_hat[index], err_y[index]), fontsize=12)
    plt.suptitle('数字图片手写体识别: 分类器: %s' %classifier_type, fontsize=18)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.show()



if __name__ == '__main__':
    startjob()