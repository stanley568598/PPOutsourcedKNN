# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 00:27:52 2021

@author: cychung
"""

from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split

from secret_sharing import generateShares

NUM_DELEGATE = 20  # DELEGATE = 代表號
RADIUS = 0.5       # RADIUS = 半徑
t = 2
n = 2

def PreProcessor( train_data , train_label ) :

    delegates = np.random.randint( train_data.shape[0] , size = NUM_DELEGATE )   # 20個 隨機點
    groups = []
    for i in delegates:                                                          # 對每個隨機點取出半徑0.5以內的所有data為一個group。
        delegate = train_data[i]
        group = []
        for train , label in list( zip( train_data , train_label ) ):
            
            distance = delegate - train
            
            distance = distance * distance
            distance = distance.sum() ** 0.5

            if( distance < RADIUS ):
                group.append( ( train , label ) )
            
        groups.append(group)
        
    group_average = []
    labels = []
    for group in groups:

        # print( len(group) )

        summation = 0        # 求和
        l = np.zeros(10)
        for data in group:
            summation = summation + data[0]
            l[ data[1] ] += 1
        
        label = np.argmax(l)
        labels.append(label)

        summation = np.round( summation / len(group) , 1 )
        group_average.append( (summation) )
    
    group_average = np.array(group_average)
    
    share_gas = []                  # 存放 groups_average 的 share 和 label
    for g in ( group_average ) :
        share_ga = []               # 存放 group 的 share
        for i in range( len(g) ):
            share_ga.append( generateShares( t , n , g[i] ) )
        
        share_gas.append( share_ga )
    
    share_gas = np.array( share_gas )

    return group_average , share_gas , np.array( labels )


if __name__ == '__main__':
    
    iris = datasets.load_iris()
    iris_data = iris.data
    iris_label = iris.target
    train_data , test_data ,  train_label ,  test_label = train_test_split( iris_data , iris_label , test_size = 0.2 , random_state = 0 )
    
    groups = PreProcessor( train_data , train_label )

    # print(groups[0])
    # print(groups[1])
    # print(groups[2])
