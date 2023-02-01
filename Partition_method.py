
import numpy as np
import pandas as pd
import random
import graphviz
import os
import pickle
from sklearn.utils import Bunch

from sklearn import datasets
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import _tree
from sklearn.tree import export_text

from my_datasets import loading_datasets
from secret_sharing import generateShares , reconstructSecret
from SS_multiplication_Protocol import multiply

n,t = 2,2

# Mersenne Prime 4th 5th 7th
L_Min = 2**7 - 1
L_Max = 2**13 - 1
PRIME = 2**19 - 1

class Node:
    
    def __init__(self, value, index=None):
        self.value = value
        self.index = index
        self.left = None
        self.right = None

    def set_left(self, left_node):
        self.left = left_node
    def set_right(self, right_node):
        self.right = right_node

class BinaryTree:
    
    def __init__(self, root):
        self.root = root

    def traversal(self, node):
        if(node != None):
            self.traversal(node.left)
            self.traversal(node.right)

    def findNearestGroup(self, value):
        node = self.root
        while(node.index == None):
            
            if(value > node.value):
                if(node.right != None):
                    node = node.right
                else:
                    node = node.left
            
            elif(value < node.value):
                if(node.left != None):
                    node = node.left
                else:
                    node = node.right
            
            elif(value == node.value):
                ld = abs(value - node.left.value)
                rd = abs(value - node.right.value)
                if(ld < rd):
                    node = node.left
                elif(ld > rd):
                    node = node.right
                else: 
                    node = node.right       # 預設向右走 
            
        return node.index

# ====

def initializer(dataName):      # dataName = ['iris' , 'wine' , 'breast_cancer' , 'abalone' , 'digits' , 'nursery' , 'mushroom' , 'Chess (King-Rook vs. King)']

    if(dataName == 'iris'):                             # Instances: 150 , Attributes: 4 , Class: 3
        NUM_DELEGATE = 50
        RADIUS = 0.75
    elif(dataName == 'wine'):                           # Instances: 178 , Attributes: 13 , Class: 3
        NUM_DELEGATE = 100
        RADIUS = 50
    elif(dataName == 'breast_cancer'):                  # Instances: 569 , Attributes: 30 , Class: 2
        NUM_DELEGATE = 300
        RADIUS = 50
    # elif(dataName == 'abalone'):                        # Instances: 4177 , Attributes: 8 , Class: 29
    #     NUM_DELEGATE = 275
    #     RADIUS = 0.05
    elif(dataName == 'digits'):                         # Instances: 1797 , Attributes: 64 , Class: 10
        NUM_DELEGATE = 500
        RADIUS = 35
    elif(dataName == 'mushroom'):                       # Instances: 8124 , Attributes: 22 , Class: 2
        NUM_DELEGATE = 1000
        RADIUS = 2.5
    elif(dataName == 'nursery'):                        # Instances: 12960 , Attributes: 8 , Class: 5
        NUM_DELEGATE = 1250
        RADIUS = 2
    # elif(dataName == 'Chess (King-Rook vs. King)'):     # Instances: 28056 , Attributes: 6 , Class: 18
    #     NUM_DELEGATE = 850
    #     RADIUS = 1.5
    else:
        NUM_DELEGATE = 1
        RADIUS = 100
    
    return NUM_DELEGATE, RADIUS

def create_group(train_data, train_label, dataName):
    
    NUM_DELEGATE, RADIUS = initializer(dataName)
    
    d_indices = np.random.choice(train_data.shape[0], size = NUM_DELEGATE, replace = False)     # 選取不重複的隨機點
    groups = []
    delegates_value = []
    delegates_target = []
    for i in d_indices:
        center = train_data[i]
        group = []
        delegate_value = []
        delegate_target = []
        for train, label in list(zip(train_data, train_label)):
            distance = center - train
            distance = distance * distance
            distance = distance.sum()**0.5
            if(distance < RADIUS):
                group.append((train, label))
                delegate_value.append(train)
                delegate_target.append(label)
        
        groups.append(group)
        delegates_value.append(delegate_value)
        delegates_target.append(delegate_target)

    delegates_train = []
    for d in delegates_value:
        data = pd.DataFrame(d)
        # print(data.mean())
        # print(list(data.mean()))
        delegates_train.append(list(data.mean()))
    
    delegates_label = []
    for d in delegates_target:
        data = pd.DataFrame(d)
        # print(d)
        # print(data.mode()[0][0])
        delegates_label.append(data.mode()[0][0])

    # print(delegates_train)
    # print(delegates_label)
    # print(len(delegates_train))
    # print(len(delegates_label))

    return groups, delegates_train, delegates_label        # 回傳 分群結果 , 各群代表點

def create_partition_dct(dataName, groups, delegates_train, delegates_label):
    
    X = pd.DataFrame(delegates_train)
    y = pd.DataFrame(delegates_label)

    '''
    groups, d_indices = create_group(train_data, train_label, dataName)
    
    X = pd.DataFrame(train_data)
    y = pd.DataFrame(train_label)

    # .iloc[ rows , columns ]，取得中心點的資料，作為代表點。
    X = X.iloc[ d_indices , : ]
    y = y.iloc[ d_indices , : ]
    '''

    classifier = tree.DecisionTreeClassifier()
    classifier = classifier.fit(X,y)
  
    # ====

    leaf_index_dict = {}                                    # 用來存放每個代表號對應的葉子ID。
    for i in range(len(delegates_train)):
        data = np.array(delegates_train[i])                 # data 為 list，需先轉成 array，再 reshape(1, -1) 變成 (一列, 原來行數) 的 二維資料。
        leaf_id = classifier.apply(data.reshape(1,-1))      # 返回每個代表號被預測為的葉子的索引。
        leaf_id = int(leaf_id)

        if(leaf_index_dict.get(leaf_id) is None):           # 檢查原本dict裡有沒有該葉節點的資訊
            leaf_index_dict[leaf_id] = []
            leaf_index_dict[leaf_id].append(i)              # i == 代表號 的 index == group index
        else:
            leaf_index_dict[leaf_id].append(i)              # 一個葉節點可能有多個代表號
    
    # ====

    fileName = './Partition_DCT_clf/partition_dct_clf_' + dataName + '.pickle'

    with open(fileName, 'wb') as f:
        pickle.dump( (classifier, leaf_index_dict), f )
    

    return classifier, leaf_index_dict

def load_partition_dct(dataName):
    
    fileName = './Partition_DCT_clf/partition_dct_clf_' + dataName + '.pickle'
    
    with open(fileName, 'rb') as f:
        model = pickle.load(f)
        
        classifier = model[0]
        leaf_index_dict = model[1]
  
    return classifier, leaf_index_dict

def create_dct(train_X, train_y, dataName, groups, delegates_train, delegates_label):

    classifier = tree.DecisionTreeClassifier()
    classifier = classifier.fit(train_X, train_y)
  
    # ====

    leaf_index_dict = {}                                    # 用來存放每個代表號對應的葉子ID。
    leaf_predict_dict = {}
    for i in range(len(train_X)):

        data = np.array(train_X[i])                         # data 為 list，需先轉成 array，再 reshape(1, -1) 變成 (一列, 原來行數) 的 二維資料。
        leaf_id = classifier.apply(data.reshape(1,-1))      # 返回每個代表號被預測為的葉子的索引。
        leaf_predict = classifier.predict(data.reshape(1,-1))

        leaf_id = int(leaf_id)

        if(leaf_index_dict.get(leaf_id) is None):           # 檢查原本dict裡有沒有該葉節點的資訊
            leaf_index_dict[leaf_id] = []
            leaf_predict_dict[leaf_id] = leaf_predict

    for i in range(len(delegates_train)):
        data = np.array(delegates_train[i])                 # data 為 list，需先轉成 array，再 reshape(1, -1) 變成 (一列, 原來行數) 的 二維資料。
        leaf_id = classifier.apply(data.reshape(1,-1))      # 返回每個代表號被預測為的葉子的索引。
        leaf_id = int(leaf_id)

        leaf_index_dict[leaf_id].append(i)                  # i == 代表號 的 index == group index
    
    # ====

    fileName = './DCT_clf/dct_clf_' + dataName + '.pickle'

    with open(fileName, 'wb') as f:
        pickle.dump( (classifier, leaf_index_dict, leaf_predict_dict), f )
    
    return classifier, leaf_index_dict, leaf_predict_dict

def load_dct(dataName):
    
    fileName = './DCT_clf/dct_clf_' + dataName + '.pickle'
    
    with open(fileName, 'rb') as f:
        model = pickle.load(f)
        
        classifier = model[0]
        leaf_index_dict = model[1]
        leaf_predict_dict = model[2]
  
    return classifier, leaf_index_dict, leaf_predict_dict

# ====

def SS_compare(i, k, client, value):        # x > y == (Server) value secret > (Client) query attibute secret

    x = generateShares(n, t, value)

    x1 = x[0][1]                        # [3, x1]
    y1 = client.sent_share(i, k)[1]     # [3, y1]

    L = np.random.randint(L_Min , L_Max)
    l = generateShares(n, t, L)
    l1 = l[0][1]
    l2 = l[1][1]

    z1 = x1 - y1 + l1

    x2 = x[1]                           # [5, x2]
    z2 = client.compare(i, k, l2, x2)

    R = np.random.randint(1, 50)
    _R = np.random.randint(1, 50)

    r = generateShares( n , t , R )
    _r = generateShares( n , t , _R )
    
    # point_X1 = 3 , point_X2 = 5
    point_X1 = r[0][0]
    point_X2 = r[1][0]
    
    r1 = r[0][1]
    r2 = r[1][1]
    
    _r1 = _r[0][1]
    _r2 = _r[1][1]

    z = []
    z.append([point_X1, z1])
    z.append([point_X2, z2])
    
    zr = multiply(z , r)

    lr = multiply(l , r)
    
    s1 = zr + _r1
    s2 = zr + _r2
    
    h1 = lr + _r1
    h2 = lr + _r2
    
    S = []
    S.append([point_X1, s1])
    S.append([point_X2, s2])

    H = []
    H.append([point_X1, h1])
    H.append([point_X2, h2])

    s = reconstructSecret(S)
    h = reconstructSecret(H)
    
    return s > h

def SS_traversal_DCT(test_instance_index, client, classifier):

    n_nodes = classifier.tree_.node_count                       # DCT 總節點數
    children_left = classifier.tree_.children_left              # DCT 某節點的左子節點 index
    children_right = classifier.tree_.children_right            # DCT 某節點的右子節點 index
    feature = classifier.tree_.feature                          # DCT 某節點 split 的 attibute index
    threshold = classifier.tree_.threshold                      # DCT 某節點 split 的 attibute value

    # 模擬 test 的 某筆 instance 執行預測，起始位置 為 root node id = 0。
    i = test_instance_index
    node_id = 0
    while(children_left[node_id] != children_right[node_id]):    # 此節點不是 leaf node，取得下一步的 node index；結束時，node_id = leaf node id = 預測結果。
        f = feature[node_id]
        t = threshold[node_id]
        if( SS_compare(i, f, client, t) ):      # t > query[i][f]
            node_id = children_right[node_id]
        else:
            node_id = children_left[node_id]

    # print("預測結果 leaf node = node_id: ", node_id)

    return node_id

# ====

''''''
def delegate(train_data,train_label,dataName):
    groups,d_indices=create_group(train_data,train_label,dataName)

    group_average=[]
    labels=[]
    for group in groups:
        summation=0
        
        l=np.zeros(10)
        for data in group:
            summation=summation+data[0]
            l[data[1]]+=1
        label=np.argmax(l)
        labels.append(label)
        summation=np.round(summation/len(group),1)
        group_average.append((summation))
    group_average=np.array(group_average)
    
    share_gas=[]   #存放groups_average的share和label
    for g in (group_average):
        share_ga=[]    #存放group的share
        for i in range(len(g)):
            share_ga.append(generateShares(t,n,g[i]))
        share_gas.append(share_ga)
    share_gas=np.array(share_gas)

    return group_average,share_gas, np.array(labels)

def findGroupIndex(d_index,d_indices):

    for i,index in enumerate(d_indices):
        if(index==d_index):
            break

    return i

def Tree(train_data,test_data,train_label,dataName):
    groups,d_indices=create_group(train_data,train_label,dataName)
  
    ##隨機選擇一個中心點
    df=pd.DataFrame(test_data)
    centerIndex=np.random.choice(range(len(train_data)))

    center=df.mean().to_numpy().squeeze()

    #計算每群的平均值
    group_average=[]
    labels=[]
    for group in groups:
        summation=0
    
        l=np.zeros(10)
        for data in group:

            summation=summation+data[0]
            l[data[1]]+=1

        label=np.argmax(l)

        labels.append(label)
        summation=np.round(summation/len(group),1)

        group_average.append((summation))
    group_average=np.array(group_average)
  
    #排序每個群
    group_distance=[]
    for i,ga in enumerate(group_average):
        label=labels[i]
        distance=((ga-center)**2).sum()**0.5

        group_distance.append((i,distance))

    sorted_group_index=sorted(group_distance,key=lambda tup:tup[1])

    """
    print(sorted_group_index)
    print()
    #print(sorted_group_index)
    #sorted_group_index=[(0,1),(1,2),(1,3),(1,4),(1,5)]     #DEBUG用
    
    print(groups[sorted_group_index[0][0]])
    print(group_average[sorted_group_index[0][0]])
    print()
    print(groups[sorted_group_index[1][0]])    
    print(group_average[sorted_group_index[1][0]])
    """

    ss_groups=[]
    for group in groups:
        ss_group=[]
        for data in group:
            ss_data=[]
            for i in range(len(data[0])):
                share=generateShares(t,n,data[0][i])
                ss_data.append(share)
        ss_group.append(ss_data)
    ss_groups.append(ss_group)

    #建樹
    nodes=[]                    #nodes用來存放接下來要處理的node
    for group in sorted_group_index:
        index=group[0]
        value=group[1]
        node=Node(value,index)
        nodes.append(node)
  
    while(len(nodes)>1):
        new_nodes=[]
        for i in range(len(nodes))[::2]:
            if(i==len(nodes)-1 and len(nodes)&1==1):
                newNode=Node(nodes[i].value)
                newNode.right=nodes[i]

            else:
                average=(nodes[i].value+nodes[i+1].value)/2
                newNode=Node(average)
                newNode.left=nodes[i]
                newNode.right=nodes[i+1]
            new_nodes.append(newNode)
        nodes=new_nodes
  
    root=nodes[0]
    binaryTree=BinaryTree(root)

    return groups,binaryTree,centerIndex,ss_groups

# ====

def ss_compare(ss_testData,value):
    x=ss_testData[0][1]
    value_share=generateShares(t,n,value)
    y=value_share[0][1]
    l1=np.random.randint(max(x,y),max(x,y)+100)

    z1=x-y+l1

    x2=ss_testData[1][1]
    y2=value_share[1][1]
    l2=np.random.randint(max(x2,y2),max(x2,y2)+100)

    z2=x2-y2+l2

    r=np.random.randint(1,50)
    _r=np.random.randint(1,50)

    r_share=generateShares(t,n,r)
    _r_share=generateShares(t,n,_r)

    r1=r_share[0][1]
    r2=r_share[1][1]

    _r1=_r_share[0][1]
    _r2=_r_share[1][1]


    zr=multiply([[3,z1],[5,z2]],[[3,r1],[5,r2]])
    lr=multiply([[3,l1],[5,l2]],[[3,r1],[5,r2]])

    s1=zr+_r1
    s2=zr+_r2

    h1=lr+_r1
    h2=lr+_r2

    s=reconstructSecret([[3,s1],[5,s2]])
    h=reconstructSecret([[3,h1],[5,h2]])

    return s<h

def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!" for i in tree_.feature ]
    print ("def tree({}):".format("".join('ss_testData')))

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print ("{}if ss_compare(ss_testData[{}],{}):".format(indent, name, threshold))
            recurse(tree_.children_left[node], depth + 1)
            print ("{}else:  # if ss_testData[{}] > {}".format(indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1)
        else:
            print ("{}return {},{}".format(indent, (tree_.value[node]).reshape(-1).tolist(),node))

    recurse(0, 1)

def traversal_DCT_iris(ss_testData):
    if ss_compare(ss_testData[3],1.899999976158142):
        if ss_compare(ss_testData[3],0.7000000029802322):
            return [2.0, 0.0, 0.0],2
        else:  # if ss_testData[3] > 0.7000000029802322
            if ss_compare(ss_testData[2],5.049999952316284):
                return [0.0, 9.0, 0.0],4
            else:  # if ss_testData[2] > 5.049999952316284
                return [0.0, 0.0, 1.0],5
    else:  # if ss_testData[3] > 1.899999976158142
        return [0.0, 0.0, 4.0],6

"""
def traversal_DCT_wine(ss_testData):
    if ss_compare(ss_testData[6],1.175000011920929):
        if ss_compare(ss_testData[3],14.550000190734863):
            return [0.0, 1.0, 0.0],2
        else:  # if ss_testData[3] > 14.550000190734863
            return [0.0, 0.0, 18.0],3
    else:  # if ss_testData[6] > 1.175000011920929
        if ss_compare(ss_testData[12],755.0):
            return [0.0, 12.0, 0.0],5
        else:  # if ss_testData[12] > 755.0
            if ss_compare(ss_testData[9],3.4550000429153442):
                return [0.0, 1.0, 0.0],7
            else:  # if ss_testData[9] > 3.4550000429153442
                return [18.0, 0.0, 0.0],8
    
def traversal_DCT_breastCancer(ss_testData):
    if ss_compare(ss_testData[7],0.05039000138640404):
        if ss_compare(ss_testData[20],17.589999198913574):
            return [0.0, 80.0],2
        else:  # if ss_testData[20] > 17.589999198913574
            return [2.0, 0.0],3
    else:  # if ss_testData[7] > 0.05039000138640404
        if ss_compare(ss_testData[22],103.10000228881836):
            if ss_compare(ss_testData[25],0.333249993622303):
                return [0.0, 3.0],6
            else:  # if ss_testData[25] > 0.333249993622303
                return [1.0, 0.0],7
        else:  # if ss_testData[22] > 103.10000228881836
            return [42.0, 0.0],8

def traversal_DCT_digits(ss_testData):
    if ss_compare(ss_testData[36],0.5):
        if ss_compare(ss_testData[28],2.5):
            if ss_compare(ss_testData[21],1.0):
                if ss_compare(ss_testData[20],0.5):
                    return [0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0],4
                else:  # if ss_testData[20] > 0.5
                    if ss_compare(ss_testData[49],12.0):
                        if ss_compare(ss_testData[54],6.0):
                            return [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],7
                        else:  # if ss_testData[54] > 6.0
                            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],8
                    else:  # if ss_testData[49] > 12.0
                        return [0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0],9
            else:  # if ss_testData[21] > 1.0
                return [105.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],10
        else:  # if ss_testData[28] > 2.5
            if ss_compare(ss_testData[34],2.5):
                if ss_compare(ss_testData[21],1.5):
                    return [0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0],13
                else:  # if ss_testData[21] > 1.5
                    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 25.0],14
            else:  # if ss_testData[34] > 2.5
                if ss_compare(ss_testData[20],11.0):
                    return [0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0],16
                else:  # if ss_testData[20] > 11.0
                    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],17
    else:  # if ss_testData[36] > 0.5
        if ss_compare(ss_testData[60],5.5):
            if ss_compare(ss_testData[21],0.5):
                if ss_compare(ss_testData[38],4.0):
                    return [0.0, 0.0, 0.0, 0.0, 0.0, 25.0, 0.0, 0.0, 0.0, 0.0],21
                else:  # if ss_testData[38] > 4.0
                    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],22
            else:  # if ss_testData[21] > 0.5
                if ss_compare(ss_testData[26],12.5):
                    if ss_compare(ss_testData[53],3.5):
                        if ss_compare(ss_testData[37],0.5):
                            if ss_compare(ss_testData[28],13.5):
                                return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0],27
                            else:  # if ss_testData[28] > 13.5
                                if ss_compare(ss_testData[51],15.0):
                                    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0],29
                                else:  # if ss_testData[51] > 15.0
                                    if ss_compare(ss_testData[6],0.5):
                                        if ss_compare(ss_testData[2],8.0):
                                            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],32
                                        else:  # if ss_testData[2] > 8.0
                                            return [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],33
                                    else:  # if ss_testData[6] > 0.5
                                        return [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],34
                        else:  # if ss_testData[37] > 0.5
                            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 86.0, 0.0, 0.0],35
                    else:  # if ss_testData[53] > 3.5
                        if ss_compare(ss_testData[3],15.5):
                            if ss_compare(ss_testData[53],7.0):
                                return [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],38
                            else:  # if ss_testData[53] > 7.0
                                if ss_compare(ss_testData[43],3.0):
                                    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],40
                                else:  # if ss_testData[43] > 3.0
                                    if ss_compare(ss_testData[43],8.0):
                                        return [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],42
                                    else:  # if ss_testData[43] > 8.0
                                        if ss_compare(ss_testData[60],2.5):
                                            return [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],44
                                        else:  # if ss_testData[60] > 2.5
                                            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],45
                        else:  # if ss_testData[3] > 15.5
                            return [0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],46
                else:  # if ss_testData[26] > 12.5
                    if ss_compare(ss_testData[14],14.5):
                        if ss_compare(ss_testData[4],9.5):
                            if ss_compare(ss_testData[43],7.5):
                                if ss_compare(ss_testData[34],12.5):
                                    return [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],51
                                else:  # if ss_testData[34] > 12.5
                                    return [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],52
                            else:  # if ss_testData[43] > 7.5
                                return [0.0, 0.0, 0.0, 0.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0],53
                        else:  # if ss_testData[4] > 9.5
                            if ss_compare(ss_testData[33],1.0):
                                return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.0],55
                            else:  # if ss_testData[33] > 1.0
                                if ss_compare(ss_testData[5],10.5):
                                    return [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],57
                                else:  # if ss_testData[5] > 10.5
                                    if ss_compare(ss_testData[44],2.5):
                                        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],59
                                    else:  # if ss_testData[44] > 2.5
                                        return [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],60
                    else:  # if ss_testData[14] > 14.5
                        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0],61
        else:  # if ss_testData[60] > 5.5
            if ss_compare(ss_testData[26],6.5):
                if ss_compare(ss_testData[43],1.5):
                    if ss_compare(ss_testData[30],2.0):
                        if ss_compare(ss_testData[42],10.0):
                            if ss_compare(ss_testData[62],13.0):
                                if ss_compare(ss_testData[59],5.0):
                                    return [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],68
                                else:  # if ss_testData[59] > 5.0
                                    return [0.0, 0.0, 0.0, 97.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],69
                            else:  # if ss_testData[62] > 13.0
                                if ss_compare(ss_testData[2],3.5):
                                    return [0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],71
                                else:  # if ss_testData[2] > 3.5
                                    return [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],72
                        else:  # if ss_testData[42] > 10.0
                            if ss_compare(ss_testData[44],4.5):
                                return [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],74
                            else:  # if ss_testData[44] > 4.5
                                return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0],75
                    else:  # if ss_testData[30] > 2.0
                        if ss_compare(ss_testData[60],14.5):
                            if ss_compare(ss_testData[51],5.5):
                                if ss_compare(ss_testData[54],1.5):
                                    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],79
                                else:  # if ss_testData[54] > 1.5
                                    return [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],80
                            else:  # if ss_testData[51] > 5.5
                                return [0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],81
                        else:  # if ss_testData[60] > 14.5
                            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.0],82
                else:  # if ss_testData[43] > 1.5
                    if ss_compare(ss_testData[54],0.5):
                        if ss_compare(ss_testData[51],12.0):
                            if ss_compare(ss_testData[50],8.5):
                                if ss_compare(ss_testData[29],8.5):
                                    if ss_compare(ss_testData[19],12.0):
                                        return [0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],88
                                    else:  # if ss_testData[19] > 12.0
                                        return [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],89
                                else:  # if ss_testData[29] > 8.5
                                    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0],90
                            else:  # if ss_testData[50] > 8.5
                                if ss_compare(ss_testData[5],11.5):
                                    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 14.0, 0.0],92
                                else:  # if ss_testData[5] > 11.5
                                    return [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],93
                        else:  # if ss_testData[51] > 12.0
                            if ss_compare(ss_testData[27],6.5):
                                return [0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],95
                            else:  # if ss_testData[27] > 6.5
                                return [0.0, 29.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],96
                    else:  # if ss_testData[54] > 0.5
                        if ss_compare(ss_testData[45],12.5):
                            if ss_compare(ss_testData[37],14.0):
                                if ss_compare(ss_testData[52],2.0):
                                    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],100
                                else:  # if ss_testData[52] > 2.0
                                    if ss_compare(ss_testData[10],9.5):
                                        if ss_compare(ss_testData[52],10.0):
                                            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],103
                                        else:  # if ss_testData[52] > 10.0
                                            return [0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],104
                                    else:  # if ss_testData[10] > 9.5
                                        return [0.0, 0.0, 90.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],105
                            else:  # if ss_testData[37] > 14.0
                                return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0],106
                        else:  # if ss_testData[45] > 12.5
                            if ss_compare(ss_testData[34],3.0):
                                if ss_compare(ss_testData[11],9.0):
                                    return [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],109
                                else:  # if ss_testData[11] > 9.0
                                    return [0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],110
                            else:  # if ss_testData[34] > 3.0
                                return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0],111
            else:  # if ss_testData[26] > 6.5
                if ss_compare(ss_testData[33],8.5):
                    if ss_compare(ss_testData[21],1.5):
                        if ss_compare(ss_testData[42],9.0):
                            if ss_compare(ss_testData[5],0.5):
                                if ss_compare(ss_testData[19],14.5):
                                    if ss_compare(ss_testData[45],4.0):
                                        return [0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],118
                                    else:  # if ss_testData[45] > 4.0
                                        if ss_compare(ss_testData[42],2.0):
                                            return [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],120
                                        else:  # if ss_testData[42] > 2.0
                                            if ss_compare(ss_testData[38],1.5):
                                                return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],122
                                            else:  # if ss_testData[38] > 1.5
                                                if ss_compare(ss_testData[61],11.5):
                                                    return [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],124
                                                else:  # if ss_testData[61] > 11.5
                                                    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],125
                                else:  # if ss_testData[19] > 14.5
                                    return [0.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],126
                            else:  # if ss_testData[5] > 0.5
                                if ss_compare(ss_testData[2],0.5):
                                    return [0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0],128
                                else:  # if ss_testData[2] > 0.5
                                    if ss_compare(ss_testData[18],10.0):
                                        return [0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],130
                                    else:  # if ss_testData[18] > 10.0
                                        return [0.0, 0.0, 0.0, 0.0, 0.0, 55.0, 0.0, 0.0, 0.0, 0.0],131
                        else:  # if ss_testData[42] > 9.0
                            if ss_compare(ss_testData[61],1.5):
                                if ss_compare(ss_testData[58],2.0):
                                    return [0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0],134
                                else:  # if ss_testData[58] > 2.0
                                    return [0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],135
                            else:  # if ss_testData[61] > 1.5
                                if ss_compare(ss_testData[20],12.0):
                                    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 78.0, 0.0, 0.0, 0.0],137
                                else:  # if ss_testData[20] > 12.0
                                    return [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],138
                    else:  # if ss_testData[21] > 1.5
                        if ss_compare(ss_testData[20],15.5):
                            if ss_compare(ss_testData[43],3.5):
                                if ss_compare(ss_testData[42],7.0):
                                    if ss_compare(ss_testData[35],0.5):
                                        if ss_compare(ss_testData[53],12.5):
                                            return [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],144
                                        else:  # if ss_testData[53] > 12.5
                                            return [0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],145
                                    else:  # if ss_testData[35] > 0.5
                                        if ss_compare(ss_testData[33],4.5):
                                            if ss_compare(ss_testData[29],7.5):
                                                if ss_compare(ss_testData[25],2.0):
                                                    return [0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0],149
                                                else:  # if ss_testData[25] > 2.0
                                                    if ss_compare(ss_testData[21],9.5):
                                                        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],151
                                                    else:  # if ss_testData[21] > 9.5
                                                        return [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],152
                                            else:  # if ss_testData[29] > 7.5
                                                return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 46.0],153
                                        else:  # if ss_testData[33] > 4.5
                                            if ss_compare(ss_testData[23],1.5):
                                                return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],155
                                            else:  # if ss_testData[23] > 1.5
                                                return [0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0],156
                                else:  # if ss_testData[42] > 7.0
                                    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 12.0, 0.0],157
                            else:  # if ss_testData[43] > 3.5
                                if ss_compare(ss_testData[33],1.5):
                                    if ss_compare(ss_testData[19],15.5):
                                        if ss_compare(ss_testData[28],3.0):
                                            return [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],161
                                        else:  # if ss_testData[28] > 3.0
                                            if ss_compare(ss_testData[35],2.5):
                                                return [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],163
                                            else:  # if ss_testData[35] > 2.5
                                                if ss_compare(ss_testData[30],6.5):
                                                    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 54.0, 0.0],165
                                                else:  # if ss_testData[30] > 6.5
                                                    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],166
                                    else:  # if ss_testData[19] > 15.5
                                        return [0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],167
                                else:  # if ss_testData[33] > 1.5
                                    if ss_compare(ss_testData[19],4.5):
                                        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0],169
                                    else:  # if ss_testData[19] > 4.5
                                        return [0.0, 0.0, 0.0, 0.0, 13.0, 0.0, 0.0, 0.0, 0.0, 0.0],170
                        else:  # if ss_testData[20] > 15.5
                            if ss_compare(ss_testData[12],11.0):
                                if ss_compare(ss_testData[10],15.5):
                                    return [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],173
                                else:  # if ss_testData[10] > 15.5
                                    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0],174
                            else:  # if ss_testData[12] > 11.0
                                if ss_compare(ss_testData[28],7.0):
                                    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0],176
                                else:  # if ss_testData[28] > 7.0
                                    if ss_compare(ss_testData[27],5.5):
                                        return [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],178
                                    else:  # if ss_testData[27] > 5.5
                                        if ss_compare(ss_testData[13],2.5):
                                            return [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],180
                                        else:  # if ss_testData[13] > 2.5
                                            if ss_compare(ss_testData[38],7.0):
                                                return [0.0, 49.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],182
                                            else:  # if ss_testData[38] > 7.0
                                                return [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],183
                else:  # if ss_testData[33] > 8.5
                    if ss_compare(ss_testData[9],0.5):
                        return [0.0, 0.0, 0.0, 0.0, 56.0, 0.0, 0.0, 0.0, 0.0, 0.0],185
                    else:  # if ss_testData[9] > 0.5
                        if ss_compare(ss_testData[19],4.5):
                            return [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],187
                        else:  # if ss_testData[19] > 4.5
                            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0],188
"""
''''''

# ====

def run_code(dataset):

    data = dataset.data
    label = dataset.target
    NUM_CLASS = dataset.NUM_CLASS
    dataName = dataset.dataName

    print('\n====\n', file=open('Partition_method__log.txt', 'a+'))
    print('資料集:{}'.format(dataName), file=open('Partition_method__log.txt', 'a+'))
    print('Instances: {} , Attributes: {} , Class: {}' .format( len(label) , len(data[0]) , NUM_CLASS ) , file=open('Partition_method__log.txt', 'a+'))

    NUM_DELEGATE, RADIUS = initializer(dataName)
    print('NUM_DELEGATE:' , NUM_DELEGATE , ' RADIUS ' , RADIUS)
    print('NUM_DELEGATE:' , NUM_DELEGATE , ' RADIUS ' , RADIUS , file=open('Partition_method__log.txt', 'a+'))
    
    # epoch=2
    # epoch=5
    epoch=10

    average_record = []

    print('\nepoch e: ', end = '')
    
    for e in range(epoch):

        print(e , end = ' ')
        
        # 切分訓練與測試資料
        train_X, test_X, train_y, test_y = train_test_split(data, label, test_size = 0.2)

        # 共用 partition
        groups, delegates_train, delegates_label = create_group(train_X, train_y, dataName)

        # ===========

        classifier, leaf_index_dict = create_partition_dct(dataName, groups, delegates_train, delegates_label)

        delegate_count = []
        for l in leaf_index_dict:
            # print(l)
            delegate_count.append(len(leaf_index_dict.get(l)))
        total_delegate_in_leafs = sum(delegate_count)
        mean_delegate = round(sum(delegate_count) / len(delegate_count), 1)
        # print("Leaf node number:", len(leaf_index_dict), ", Total Delegates in Leaf nodes:", total_delegate_in_leafs, ", Mean Delegates in a Leaf node:", mean_delegate)
        
        group_data_count = []
        for g in range(len(groups)):
            # print(len(groups[g]))
            group_data_count.append(len(groups[g]))
        total_data_in_group = sum(group_data_count)
        mean_data_in_leaf = round(total_data_in_group / len(leaf_index_dict), 1)
        mean_data_in_delegate = round(total_data_in_group / total_delegate_in_leafs, 1)
        # print("Total Data in Groups:", total_data_in_group, ", Mean Data in Leaf:", mean_data_in_leaf, ", Mean Data in Delegate:", mean_data_in_delegate)

        average_record.append([len(leaf_index_dict), total_delegate_in_leafs, mean_delegate, total_data_in_group, mean_data_in_leaf, mean_data_in_delegate])

    record = pd.DataFrame(average_record)
    # print(record)
    
    record_mean = record.mean()
    # print(record_mean[0])
    
    print()
    print('\nEpoch: ', epoch)

    print("Leaf node number:", record_mean[0], ", Total Delegates in Leaf nodes:", record_mean[1], ", Mean Delegates in a Leaf node:", record_mean[2])
    print("Total Data in Groups:", record_mean[3], ", Mean Data in Leaf:", record_mean[4], ", Mean Data in Delegate:", record_mean[5])

    print('\nEpoch: ', epoch , file=open('Partition_method__log.txt', 'a+'))

    print("Leaf node number:", record_mean[0], ", Total Delegates in Leaf nodes:", record_mean[1], ", Mean Delegates in a Leaf node:", record_mean[2] , file=open('Partition_method__log.txt', 'a+'))
    print("Total Data in Groups:", record_mean[3], ", Mean Data in Leaf:", record_mean[4], ", Mean Data in Delegate:", record_mean[5] , file=open('Partition_method__log.txt', 'a+'))


    return 0

if __name__ == '__main__':
    
    print('', file=open('Partition_method__log.txt', 'w'))

    # dataName = ['iris' , 'wine' , 'breast_cancer' , 'abalone' , 'digits' , 'nursery' , 'mushroom' , 'Chess (King-Rook vs. King)']
    dataName = ['iris' , 'wine' , 'breast_cancer' , 'digits' , 'mushroom' , 'nursery']
    # dataName = 'iris'

    if(isinstance(dataName, list)):
        while(len(dataName) > 0):
            dataName_i = dataName.pop(0)
            dataset = loading_datasets(dataName_i)
            run_code(dataset)
    else:
        dataset = loading_datasets(dataName)
        run_code(dataset)

    print('\n====\n')
    print('\n====\n', file=open('Partition_method__log.txt', 'a+'))