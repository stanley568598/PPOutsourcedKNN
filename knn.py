# -*- coding: utf-8 -*-
"""
Created on Sat May  9 15:20:35 2020

@author: cychung
"""


from sklearn import datasets
from sklearn.model_selection import train_test_split
import time
import numpy as np
import os

k = 5
num_class = 10

def Euclidean_distance( data1 , data2 ):
    sum = 0
    for i in range( len(data1) ) :
        sum += ( np.round( data1[i] - data2[i] , 1 ) ) ** 2
    
    sum = sum ** 0.5
    return sum

# Relational Neighbor (RN) classifier 
# # 每個資料點 都 求出最近的label，回傳。
def rN_classifier( train_data , test_data , train_label , r ) :
    result = []
    for i in test_data:

        dataInCircle = []
        for j in range( len(train_data) ):
            distance = Euclidean_distance( i , train_data[j] )
            if( distance <= r ):
                dataInCircle.append( train_label[j] )
        
        label_zero = 0
        label_one = 0
        label_two = 0
        for k in range( len(dataInCircle) ) :

            if( dataInCircle[k] == 0 ):
                label_zero += 1

            elif( dataInCircle[k] == 1 ):
                label_one += 1

            else:
                label_two += 1


        maximum = label_zero
        m = 0

        if( label_one > maximum ) :
           maximum = label_one
           m = 1
        
        if( label_two > maximum ) :
            maximum = label_two
            m = 2
        
        result.append( m )

    return result


# K Nearest Neighbor (KNN) classifier 
# # 對每筆 test_data 找到 最近的 k個 train_data，回傳 這 k個 train_data 出現最多的 label。
def  knn_classifier( train_data , test_data , train_label , k , Num_class ) :
    
    result = []
    
    for i in test_data:

        DAL = {}                                                  # DAL = DistanceAndLabel

        for j in range( len(train_data) ) :
            distance = Euclidean_distance( i , train_data[j] )
            DAL[ distance ] = train_label[j]                      ##　重複的位置可能被其他標籤覆蓋

        SDAL = sorted( DAL.items() , key = lambda x:x[0] )        # SDAL = SortedDistanceAndLabel：將 每個 train data 離此 test data 的距離 由小到大 排序。

        predict_label = {}
        for i in range( Num_class ):
            predict_label.update( { i : 0 } )
            
        KSDAL = SDAL[ 0 : k ]                                     # 取出 SDAL 前 k 筆資料
        if( len(SDAL) < k ) :                                     # 當 SDAL 資料數少於 k，更新 Knn 的 k 值。
            k = len(SDAL)
        
        for i in range(k) :                                       # 根據 label 計算 最近的 k 筆資料中 的 label的數量。
            predict_label[ ( KSDAL[i] )[1] ] += 1

        SPL = sorted( predict_label.items() , key = lambda x:x[1] , reverse = True )

        result.append( SPL[0][0] )
        
    return result
    

if __name__ == '__main__':
    
    iris = datasets.load_digits()
    iris_data = iris.data
    iris_label = iris.target
    
    t1 = time.time()
    false = 0

    for i in range(1):
        train_data , test_data , train_label , test_label = train_test_split( iris_data , iris_label , test_size = 0.2 , random_state = 0 )

        # result = rN_classifier( train_data , test_data , train_label , 0.7 )
        result = knn_classifier( train_data , test_data , train_label , k , num_class )

        # 計算正確率
        correct = 0
        for i in range( len(result) ) :
            if( result[i] == test_label[i] ) :
                correct += 1

        print( "正確率 : " , end = ' ' )    # 97.5 %
        
        print( correct / len(test_data) * 100 , '%' )

        print( str( correct / len(result) * 100 ) + "%")

    t2 = time.time()
    
    t = t2 - t1
    print( "耗時 : " , t )      # 5 分鐘
