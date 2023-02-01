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

from sklearn import tree
from sklearn.tree import export_text
from sklearn.tree import _tree

import os
import graphviz    # 繪製圖形結構 
import pickle      # 打包模型工具

"""
dataset = 'iris'
NUM_DELEGATE = 16
RADIUS = 0.5
"""

dataset = 'digits'
NUM_DELEGATE = 1024
RADIUS = 25

t = 2
n = 2

def create_group( train_data , train_label ):

    d_indices = np.random.randint( train_data.shape[0] , size = NUM_DELEGATE )
    groups = []

    for i in d_indices:
        delegate = train_data[i]
        group = []
        for train , label in list( zip( train_data , train_label ) ):
            distance = delegate - train
            distance = distance * distance
            distance = distance.sum() ** 0.5

            if(distance < RADIUS):
                group.append( ( train , label ) )
        
        groups.append(group)
    
    return groups , d_indices

def delegate( train_data , train_label ):

    groups , d_indices = create_group( train_data , train_label )

    group_average = []
    labels = []
    for group in groups:

        summation = 0        # 求和
        l = np.zeros(10)
        for data in group:
            summation = summation + data[0]
            l [ data[1] ] += 1
        
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


def Tree( train_data , test_data , train_label ):
    
    groups , d_indices = create_group( train_data , train_label )

    # 方案_1. 從 train_data 隨機選擇一個中心點
    centerIndex = np.random.choice( range( len(train_data) ) )
    # center = train_data[centerIndex]

    # 方案_2. 根據 test_data 的平均值 做為 中心點
    df = pd.DataFrame(test_data)
    # center = df.mean().to_numpy()
    center = df.mean().to_numpy().squeeze()


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
    print( sorted_group_index )
    print()
    # print( sorted_group_index )
    # sorted_group_index = [ (0,1) , (1,2) , (1,3) , (1,4) , (1,5) ]     # DEBUG 用
        
    print( groups[ sorted_group_index[0][0] ] )
    print( group_average[ sorted_group_index[0][0] ] )
    print()
    print( groups[ sorted_group_index[1][0] ] )    
    print( group_average[ sorted_group_index[1][0] ] )
    """
    
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

    # return groups , binaryTree , center
    return groups , binaryTree , centerIndex , ss_groups


def decision_Tree( train_data , train_label ):

    groups , d_indices = create_group( train_data , train_label )
    
    X = pd.DataFrame(train_data)
    y = pd.DataFrame(train_label)
    
    X = X.iloc[ d_indices , : ]
    y = y.iloc[ d_indices , : ]

    classifier = tree.DecisionTreeClassifier()
    classifier = classifier.fit( X , y )
    
    leaf_index_dict = {}                               # 用來存放每個代表號對應的葉子ID
    
    for i in d_indices :
        data = train_data[i]
        leaf_id = classifier.apply( data.reshape( 1 , -1 ) )
        leaf_id = int(leaf_id)
        if( leaf_index_dict.get(leaf_id) is None ) :   # 檢查原本dict裡有沒有該葉節點的資訊
            leaf_index_dict[leaf_id] = []
        
        leaf_index_dict[leaf_id].append(i)
        
    """ 畫圖
    num_features = train_data.shape[1]
    num_class = len( train_label[0].value_counts() )
    
    dot_data = tree.export_graphviz( classifier, out_file = None,
                   feature_names = [ '%d' %i for i in range(num_features) ],
                   class_names = [ '%d' %i for i in range(num_class) ],
                   filled = True, rounded = True, leaves_parallel = True)
    
    graph = graphviz.Source(dot_data)

    graph.render('Tree_Graph{}'.format(NUM_DELEGATE))
    
    """

    # 儲存model
    # with open( 'clf.pickle' , 'wb' ) as f:
    #     pickle.dump( classifier , f )

    return classifier , leaf_index_dict , groups , d_indices
    
def tree_to_code( tree , feature_names ):

    tree_ = tree.tree_
    print( tree_.feature )

    feature_name = [ feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!" for i in tree_.feature ]
    print( "def tree({}):".format( "".join('testData') ) )

    def recurse( node , depth ):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            print ("{} if ss_compare( testData[{}] , {} ) : ".format( indent , name , threshold ))
            recurse( tree_.children_left[node] , depth + 1 )

            print ("{} else : # if testData[{}] > {} ".format( indent , name , threshold ))
            recurse( tree_.children_right[node] , depth + 1 )
        
        else:
            print( "{} return {} , {} ".format( indent , ( tree_.value[node] ).reshape(-1).tolist() , node ) )

    recurse ( 0 , 1 )

def findGroupIndex( d_index , d_indices ):
    for i , index in enumerate( d_indices ):
        if( index == d_index ):
            break
    return i


if __name__=='__main__':

    if( dataset == 'digits' ) :
        iris = datasets.load_digits()
    elif( dataset == 'iris' ) :
        iris = datasets.load_iris()
    
    iris_data = iris.data
    iris_label = iris.target

    train_data , test_data , train_label , test_label = train_test_split( iris_data , iris_label , test_size = 0.2 )
    
    load_dct = True
    if( load_dct ):
        with open( 'clf.pickle' , 'rb' ) as f:
            classifier = pickle.load(f)
    else:
        classifier , leaf_index_dict , groups , d_indices = decision_Tree( train_data , train_label )
    
    result = ( classifier.predict(test_data) )
    
    incorrect = 0
    for i in range( len(result) ):
        if( result[i] != test_label[i] ):
            incorrect += 1
    
    print('correct rate : ' , ( len(result) - incorrect ) / len(result) * 100 , '%' )

    tree_to_code( classifier , [ str(i) for i in range(64) ] )
