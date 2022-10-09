

from __future__ import absolute_import, division, print_function, \
    unicode_literals
import sys
sys.path.append("..")
from nfs.modelflow.code.report import *
from nfs.modelflow.code.component import *
from nfs.modelflow.code.visualize import *
import pandas as pd
import argparse
import json
import uuid
from sklearn.neighbors import KNeighborsClassifier
import pickle
import numpy as np
import os


'''
K近邻算法进行分类
'''


def read_data(path):
    data = pd.read_csv(path)
    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    return x, y


def knn():
    utils = ComponentUtils()
    data_dir = utils.getDataDir()
    data_path_list = utils.getPaths()
    model_path = utils.getParam("model_path")
    if model_path == "None":  # 训练
        Model = os.path.join(data_dir, uuid.uuid1().__str__() + '.pickle')
        PredictFile = os.path.join(data_dir, uuid.uuid1().__str__() + '.csv')

        X, Y = read_data(data_path_list[0])

        # train
        clf = KNeighborsClassifier()
        clf.fit(X, Y)

        # val
        valX, valY = read_data(data_path_list[1])
        Predict_Y = clf.predict(valX).astype(np.str)
        #print(valY)
        #print(Predict_Y)
        # 存储验证集预测结果,每一行第一个数是label，第二个是predict
        with open(PredictFile, 'w') as f:
            for index, p in enumerate(Predict_Y):
                f.write('{},{}\n'.format(valY[index], p))

        # 存储模型
        target = [int(i) for i in valY]
        predict = [int(i) for i in Predict_Y]
        utils.saveModel(clf,ComponentUtils.modelType.sklearn)
        vis = modelFlowVisualize()
        x_data = [1,2,3,4,5,6,7,8,9]
        y_data = [9,8,7,6,5,4,3,2,1]
        acc = [0.87,0.88,0.81,0.85,0.90,0.92,0.93,0.95,0.98]
        vis.lineChart(x_data,acc,"times","acc",'title')
    
        vis.scatter(x_data,y_data,20,"scatter table")
    
        data = {"height" : [170,120,150,140],"weight" : [56,57,76,80]}
        header = ['height','weight']
        vis.table(data, header,'title')
        category = ['1','2','3']
        #target = [0,1,2,1,1,2,1,2]
        #prediction = [1,1,1,2,2,2,0,0]
        vis.confusion_matrix(category,target,predict)
        vis.draw()
        utils.saveOutput([PredictFile])

    else:  # 测试
        PredictFile = os.path.join(data_dir, uuid.uuid1().__str__() + '.csv')
        X = pd.read_csv(data_path_list[0])
        # 加载模型
        with open(model_path, 'rb') as m:
            clf = pickle.load(m)
        # 预测
        Predict_Y = clf.predict(X).astype(np.str)

        # 存储predict数据每一行第一个数是predict
        with open(PredictFile, 'w') as f:
            for index, p in enumerate(Predict_Y):
                f.write('{}\n'.format(p))
        utils.saveOutput([PredictFile])

        #result_table = [{"result_path": PredictFile, "graph_type": "0", "result_type": "predict result"}]

        #return result_table, None


if __name__ == '__main__':
    knn()

