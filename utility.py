# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 00:27:52 2021

@author: cychung
"""

import numpy as np
import pandas as pd
import random

from sklearn import datasets
from sklearn.model_selection import train_test_split

from secret_sharing import generateShares
from BinaryTree import Node,BinaryTree


# """iris parameter
# NUM_DELEGATE = 16
# RADIUS = 0.5
# """

# """digits parameter
# NUM_DELEGATE = 100
# RADIUS = 20
# """

# """test
NUM_DELEGATE = 100
RADIUS = 50
# """

t = 2
n = 2

def tree( train_data , test_data , train_label ) :

    # print("\ntrain_data.shape[0] : " , train_data.shape[0] , "\nrange : ", range( train_data.shape[0] ))

    # 從 序列( 0 ~ 1437 ) 中，隨機取得 長度100 的 片段。
    delegates = random.sample( range( train_data.shape[0] ) , NUM_DELEGATE )
    # print("\ndelegates : " , delegates)
    
    groups = []
    
    for i in delegates :

        delegate = train_data[i]     # 這些 代表號 所對應的 train_data，為 隨機分群 計算長度 的 基準點。

        group = []                   # 分群

        for train , label in list( zip( train_data , train_label ) ) :

            distance = ( ( delegate - train ) ** 2 ).sum() ** 0.5
            
            if( distance < RADIUS ):
                group.append( ( train , label ) )

        groups.append(group)
    

    # 方案_1. 從 train_data 隨機選擇一個中心點
    centerIndex = np.random.choice( range( len(train_data) ) )
    center = train_data[centerIndex]

    # 方案_2. 根據 test_data 的平均值 做為 中心點
    df = pd.DataFrame(test_data)
    center = df.mean().to_numpy()
    # center = df.mean().to_numpy().squeeze()


    # 計算每群集合的平均data數值
    group_average = []
    labels = []

    for group in groups:

        summation = 0       # data 求和
        l = np.zeros(10)

        for data in group:
            summation = summation + data[0]
            l[ data[1] ] += 1

        label = np.argmax(l)
        labels.append(label)

        summation = np.round( summation / len(group) , 1 )
        group_average.append( (summation) )

    # print("group_average list_array")
    # print( group_average[0] )

    group_average = np.array( group_average )
    # print("group_average array_array")
    # print(group_average)


    # 計算 每個群 與 中心點的距離
    group_distance = []
    for i , ga in enumerate( group_average ):   # enumerate 列舉

        label = labels[i]
        distance = ( ( ga - center ) ** 2 ).sum() ** 0.5
        group_distance.append( ( i , distance ) )

    # 根據 與中心點的距離 由小到大 排序每個群
    sorted_group_index = sorted( group_distance , key = lambda tup:tup[1] )

    """
    print()
    # print( sorted_group_index )
    # sorted_group_index = [ (0,1) , (1,2) , (1,3) , (1,4) , (1,5) ]     # DEBUG 用
    
    print( groups[ sorted_group_index[0][0] ] )
    print( group_average[ sorted_group_index[0][0] ] )
    print()
    print( groups[ sorted_group_index[1][0] ] )    
    print( group_average[ sorted_group_index[1][0] ] )
    
    ss_groups = []
    for group in groups:

        ss_group = []
        for data in group:
            
            ss_data = []
            for i in range( len( data[0] ) ):
                share = generateShares( t , n , data[0][i] )
                ss_data.append( share )
            
            ss_group.append( ss_data )
        
        ss_groups.append(ss_group)
    """

    # 建樹
    nodes = []                        # nodes 用來存放接下來要處理的 node
    for group in sorted_group_index:
        index = group[0]
        value = group[1]
        node = Node( value , index )
        nodes.append(node)
    
    while( len(nodes) > 1 ):
        new_nodes = []
        for i in range( len(nodes) ) [::2] :

            if( i == len(nodes) - 1 and len(nodes) & 1 == 1 ) :
                newNode = Node( nodes[i].value )
                newNode.right = nodes[i]

            else:
                average = ( nodes[i].value + nodes[ i + 1 ].value ) / 2
                newNode = Node( average )
                newNode.left = nodes[i]
                newNode.right = nodes[ i + 1 ]
            
            new_nodes.append( newNode )
        nodes = new_nodes
    
    root = nodes[0]
    binaryTree = BinaryTree(root)

    return groups , binaryTree , center


if __name__ == '__main__' :

    iris = datasets.load_iris()
    iris_data = iris.data
    iris_label = iris.target

    train_data , test_data , train_label , test_label = train_test_split( iris_data , iris_label , test_size = 0.2 , random_state = 0 ) 
    

    groups , binaryTree , center = tree( train_data , test_data , train_label )

    print()

    groups_size = []
    for g in groups:
        g = np.array( g , dtype = object )
        groups_size.append(len(g))
    
    print('groups_size : ' , groups_size)

    print('center : ' , center)
