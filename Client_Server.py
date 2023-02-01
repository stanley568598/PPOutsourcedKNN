
import numpy as np
import pandas as pd
import time
from pkg_resources import evaluate_marker
from sklearn.metrics import accuracy_score
from sklearn.utils import Bunch

from sklearn import datasets
from sklearn import tree
from sklearn import neighbors
from sklearn.model_selection import train_test_split

from my_datasets import loading_datasets
from secret_sharing import generateShares , reconstructSecret
from SS_multiplication_Protocol import multiply
from Partition_method import Tree, create_group, create_partition_dct , load_partition_dct , SS_traversal_DCT, create_dct, load_dct

# ====

k = 5
n,t = 2,2

# Mersenne Prime 4th 5th 7th
L_Min = 2**7 - 1
L_Max = 2**13 - 1
PRIME = 2**19 - 1

MINIMUM = 9999999

# ====

class Server:

    def __init__(self, train_X, train_y, k, dataName , NUM_CLASS):
        self.data = train_X
        self.train_label = train_y
        self.k = k
        self.share = self.transToShare(self.data)
        self.dataName = dataName
        self.NUM_CLASS = NUM_CLASS

    def transToShare(self, data):
        share = []
        for i in range(len(data)):
            temp = []
            instance = data[i]
            for j in range(len(instance)):
                temp.append( generateShares(n , t , instance[j]) )    
            share.append(temp)   
        return np.array(share)
    
    def sent_share(self, i, j):
        if( len(self.share) == 0 ):
            return
        else:
            # print('server share:', self.share[i][j])
            return self.share[i][j][1]                      # [5, x2]
    
    def get_share(self, i, j):
        if( len(self.share) == 0 ):
            return
        else:
            return self.share[i][j][0]                      # [3, x1]
    
    def compare(self, i, j, k, client):   # x > y == (Server) dataset.value > (Client) query.value

        x1 = self.get_share(j, k)[1]          # [3, x1]
        y1 = client.sent_share(i, k)[1]       # [3, y1]

        L = np.random.randint(L_Min , L_Max)
        l = generateShares(n, t, L)
        l1 = l[0][1]
        l2 = l[1][1]
        
        z1 = x1 - y1 + l1

        server_share = self.sent_share(j, k)        # [5, x2]
        z2 = client.compare(i, k, l2, server_share)
        
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

    def get_difference(self, i, j, client_share, comp):
        if(comp):
            y = self.get_share(i, j)[1] - client_share[1]
        else:
            y = client_share[1] - self.get_share(i, j)[1]

        y = y % PRIME
        return y

    def knn_classifier(self, distance):         # distance = 每筆 train instance 跟 某個 query instance 的距離

        result = []

        DAL = []                                # DAL = DistanceAndLabel
        for i in range(len(distance)):
            d = distance[i]
            l = self.train_label[i]
            DAL.append([ d , l ])
        
        SDAL = sorted( DAL , key=(lambda x : x[0]) )    # SDAL = Sorted_DAL  (按距離由小到大排序)
        
        predict_label = {}
        neighbor_distance = SDAL[0][0]
        neighbor_count = 0
        for i in range(len(SDAL)):
            d = SDAL[i][0]
            l = SDAL[i][1]

            if d > neighbor_distance:
                if neighbor_count > self.k:
                    break
                else:
                    neighbor_distance = d

            if d <= neighbor_distance:
                predict_label[l] = predict_label.get(l, 0) + 1
                neighbor_count += 1

        SPL = sorted( predict_label.items() , key=(lambda x:x[1]), reverse=True )

        result.append(SPL[0][0])
        
        return result
    
    def ss_instance_knn(self, server, client, i):     # 對某個 query instance，Server 從 全部資料 取出一個小的 dataset 做 knn。 
        result = []

        distance = []

        for j in range(len(server.share)):
            deltaSum = 0

            for k in range(server.share.shape[1]):        # shape 取得 list 的 (row, column)

                # STEP1 每個 attribute 比較大小
                comp = server.compare(i, j, k, client)    # x > y == (Server) dataset.value > (Client) query.value

                # STEP2 計算 difference
                client_share = client.sent_share(i, k)
                difference1 = server.get_difference(j, k, client_share, comp)
                
                server_share = server.sent_share(j, k)
                difference2 = client.sent_difference(i, k, server_share, comp)

                # STEP3 計算距離
                point_X1 = client_share[0]  # client sent to server，point_X1 = 3
                point_X2 = server_share[0]  # server sent to client，point_X2 = 5
                difference = []
                difference.append([point_X1, difference1])
                difference.append([point_X2, difference2])
                delta = reconstructSecret(difference)

                deltaSum += delta**2
            
            deltaSum = deltaSum**0.5

            distance.append(deltaSum)

        result += server.knn_classifier(distance)

        return(result)

    # ====

    def SS_knn(self, client):
        result = []

        for i in range(client.query_num):
            distance = []

            for j in range(len(self.share)):
                deltaSum = 0

                for k in range(self.share.shape[1]):        # shape 取得 list 的 (row, column)

                    # STEP1 每個 attribute 比較大小
                    comp = self.compare(i, j, k, client)    # x > y == (Server) dataset.value > (Client) query.value

                    # STEP2 計算 difference
                    client_share = client.sent_share(i, k)
                    difference1 = self.get_difference(j, k, client_share, comp)
                    
                    server_share = self.sent_share(j, k)
                    difference2 = client.sent_difference(i, k, server_share, comp)

                    # STEP3 計算距離
                    point_X1 = client_share[0]  # client sent to server，point_X1 = 3
                    point_X2 = server_share[0]  # server sent to client，point_X2 = 5
                    difference = []
                    difference.append([point_X1, difference1])
                    difference.append([point_X2, difference2])
                    delta = reconstructSecret(difference)

                    deltaSum += delta**2
                
                deltaSum = deltaSum**0.5

                distance.append(deltaSum)

            result += self.knn_classifier(distance)
            
        return(result)

    def dct_knn(self, client, classifier, leaf_index_dict, leaf_predict_dict, groups, delegates_train, delegates_label):
        result = []
        
        ## 難點：Server 不可取得 client.data。
        for test in client.data:
            _2dtest = test.reshape(1, -1)           # apply函數 需要輸入 2d array
            leaf_id = classifier.apply(_2dtest)
            leaf_id = int(leaf_id)                  # 轉回int
            
            d_list = leaf_index_dict[leaf_id]       # d_list = 葉節點內全部代表號的 index
            leaf_predict = leaf_predict_dict[leaf_id]

            if len(d_list) == 0 :
                result.append( leaf_predict )
            else :
                # 找出與葉節點內所有代表號中，最近的代表號
                minimum = MINIMUM
                index = 0
                for i in d_list:
                    delegate = delegates_train[i]

                    distance = 0
                    for a in range(len(delegate)):
                        distance += ( (test[a] - delegate[a])**2 ).sum()**0.5
                    
                    if(distance < minimum):
                        minimum = distance
                        index = i
                
                group = groups[index]
                data = []
                label = []
                for d in group:
                    data.append(d[0])
                    label.append(d[1])
                

                """
                # 拿 葉節點內所有資料(包含重複資料，代表增加重複資料的權重，跟 leaf node 相關度更高) 做 ss knn (k=5 or k=3)。
                predict_leaf = []
                for index in range(len(d_list)):
                    group = groups[index]
                    predict_leaf += group
                
                data = []
                label = []
                for d in predict_leaf:
                    data.append(d[0])
                    label.append(d[1])
                """

                server = Server(data , label , 5 , self.dataName , self.NUM_CLASS)
                client = Client(_2dtest)
                result.append( server.SS_knn(client) )
            
        result = np.array(result).reshape(-1)

        return result

    def dct_SS_knn(self, client, classifier, leaf_index_dict, leaf_predict_dict, groups, delegates_train, delegates_label):
        result = []

        for test_instance_index in range(client.query_num):
    
            predict_leaf_id = SS_traversal_DCT(test_instance_index, client, classifier)
            predict_leaf_id = int(predict_leaf_id)          # 轉回int

            d_list = leaf_index_dict[predict_leaf_id]       # d_list = 葉節點內全部代表號的 index
            leaf_predict = leaf_predict_dict[predict_leaf_id]
            
            if len(d_list) == 0 :
                result.append( leaf_predict )
            else :
                predict_leaf_delegates_data = []
                predict_leaf_delegates_label = []
                for index in range(len(d_list)):
                    predict_leaf_delegates_data.append(delegates_train[index])
                    predict_leaf_delegates_label.append(index)
                
                predict_leaf_delegates_server = Server(predict_leaf_delegates_data , predict_leaf_delegates_label , 1 , "predict_leaf_delegates" , len(d_list))
                
                # 找出與葉節點內所有代表號中，最近的代表號：拿 代表號群 做 ss knn，K = 1。
                nearest_Delegate_result = self.ss_instance_knn(predict_leaf_delegates_server , client , test_instance_index)
                nearest_Delegate_index = nearest_Delegate_result[0]

                # 拿 最近的代表號之群 做 ss knn (k=5 or k=3)。
                group = groups[nearest_Delegate_index]
                data = []
                label = []
                for d in group:
                    data.append(d[0])
                    label.append(d[1])
                
            
                """
                # 拿 葉節點內所有資料(包含重複資料，代表增加重複資料的權重，跟 leaf node 相關度更高) 做 ss knn (k=5 or k=3)。
                predict_leaf = []
                for index in range(len(d_list)):
                    group = groups[index]
                    predict_leaf += group
                
                data = []
                label = []
                for d in predict_leaf:
                    data.append(d[0])
                    label.append(d[1])
                """

                group_server = Server(data , label , 5 , "predict_leaf" , self.NUM_CLASS)  # k 考慮取小於 5 (k=3)。

                # 找出代表號 group 與 這個 instance 的 knn。
                group_result = self.ss_instance_knn(group_server , client , test_instance_index)
                result.append( group_result[0] )
            
        result = np.array(result).reshape(-1)
        
        return result

    def partition_dct_knn(self, client, classifier, leaf_index_dict, groups, delegates_train, delegates_label):
        result = []
        
        ## 難點：Server 不可取得 client.data。
        for test in client.data:
            _2dtest = test.reshape(1, -1)           # apply函數 需要輸入 2d array
            leaf_id = classifier.apply(_2dtest)
            leaf_id = int(leaf_id)                  # 轉回int
            
            d_list = leaf_index_dict[leaf_id]       # d_list = 葉節點內全部代表號的 index

            
            # 找出與葉節點內所有代表號中，最近的代表號
            minimum = MINIMUM
            index = 0
            for i in d_list:
                delegate = delegates_train[i]

                distance = 0
                for a in range(len(delegate)):
                    distance += ( (test[a] - delegate[a])**2 ).sum()**0.5
                
                if(distance < minimum):
                    minimum = distance
                    index = i
            
            group = groups[index]
            data = []
            label = []
            for d in group:
                data.append(d[0])
                label.append(d[1])
            

            """
            # 拿 葉節點內所有資料(包含重複資料，代表增加重複資料的權重，跟 leaf node 相關度更高) 做 ss knn (k=5 or k=3)。
            predict_leaf = []
            for index in range(len(d_list)):
                group = groups[index]
                predict_leaf += group
            
            data = []
            label = []
            for d in predict_leaf:
                data.append(d[0])
                label.append(d[1])
            """

            server = Server(data , label , 5 , self.dataName , self.NUM_CLASS)
            client = Client(_2dtest)
            result.append( server.SS_knn(client) )
            
        result = np.array(result).reshape(-1)

        return result

    def partition_dct_SS_knn(self, client, classifier, leaf_index_dict, groups, delegates_train, delegates_label):
        result = []
    
        for test_instance_index in range(client.query_num):

            predict_leaf_id = SS_traversal_DCT(test_instance_index, client, classifier)
            predict_leaf_id = int(predict_leaf_id)          # 轉回int

            d_list = leaf_index_dict[predict_leaf_id]       # d_list = 葉節點內全部代表號的 index
            
            
            predict_leaf_delegates_data = []
            predict_leaf_delegates_label = []
            for index in range(len(d_list)):
                predict_leaf_delegates_data.append(delegates_train[index])
                predict_leaf_delegates_label.append(index)
            
            predict_leaf_delegates_server = Server(predict_leaf_delegates_data , predict_leaf_delegates_label , 1 , "predict_leaf_delegates" , len(d_list))
            
            # 找出與葉節點內所有代表號中，最近的代表號：拿 代表號群 做 ss knn，K = 1。
            nearest_Delegate_result = self.ss_instance_knn(predict_leaf_delegates_server , client , test_instance_index)
            nearest_Delegate_index = nearest_Delegate_result[0]

            # 拿 最近的代表號之群 做 ss knn (k=5 or k=3)。
            group = groups[nearest_Delegate_index]
            data = []
            label = []
            for d in group:
                data.append(d[0])
                label.append(d[1])
            
            
            """
            # 拿 葉節點內所有資料(包含重複資料，代表增加重複資料的權重，跟 leaf node 相關度更高) 做 ss knn (k=5 or k=3)。
            predict_leaf = []
            for index in range(len(d_list)):
                group = groups[index]
                predict_leaf += group
            
            data = []
            label = []
            for d in predict_leaf:
                data.append(d[0])
                label.append(d[1])
            """

            group_server = Server(data , label , 5 , "predict_leaf" , self.NUM_CLASS)  # k 考慮取小於 5 (k=3)。

            # 找出代表號 group 與 這個 instance 的 knn。
            group_result = self.ss_instance_knn(group_server , client , test_instance_index)
            result.append( group_result[0] )
            
        result = np.array(result).reshape(-1)

        return result

    '''
    def Delegate_Method(self,client):
        groups_average,share_gas,labels=delegate(self.data,self.train_label)
        """
        result=[]
        for test in client.data:
            d=[]
            for g in groups_average:
                distance=0
                distance=test-g
                distance*=distance
                distance=distance.sum()
                distance=distance**0.5
                d.append(distance)
            result.append(labels[np.argmin(d)])
            
        return result
        """
        self.train_label=labels
        self.data=groups_average
        self.share=share_gas
        self.k=1
        result=self.knn(client)
        print(result)
        return result   

    def computeDelta(self, difference):
        x1, y1, x2, y2 = difference[0][0], difference[0][1], difference[1][0], difference[1][1]
                
        r=(y2-y1)/(x2-x1)
        
        delta=y1-x1*r
        delta=np.round(delta,1)
        delta = delta % PRIME
        
        if(delta>100):
            r=(y2-y1)*499 % PRIME
        
            delta=y1-x1*r
    
            delta=np.round(delta,1)
            delta = delta % PRIME
            
        return np.round(delta,1)
    
    def tree_Method(self, client, groups, binaryTree, centerIndex, ss_groups):
        
        # 計算 distance
        center = pd.DataFrame(client.data).mean().to_numpy().squeeze()

        result = []
        for i in range(len(client.data)):
            distance = ((center-client.data[i])**2).sum()**0.5
            
            index = binaryTree.findNearestGroup(distance)
            
            group = groups[index]
            group = np.array(group, dtype=object)
            
            l = np.zeros(100, dtype = int)
            for data in group:
                l[ data[1] ] += 1
            # print(l)

            result.append(np.argmax(l))
        
        return result
    
    def tree_Method_SS(self,client,groups,binaryTree,centerIndex,ss_groups):
        #計算distance
        distance=0
        center_ss=self.share[centerIndex]

        result=[]
        origin_data=self.data
        origin_label=self.train_label
        origin_share=self.share
        for i in range(len(client.data)):
            
            self.data=origin_data
            self.label=origin_label
            self.share=origin_share
            distance=0
            deltaSum=0

            for k in range(self.data.shape[1]):
                #STEP1  交換share
                client_share=client.sent_share(i,k)
                server_share=self.sent_share(centerIndex,k)
                
                #STEP2 比較大小
                comp=(self.compare(i,centerIndex,k,client))
                #STEP2 client傳回結果
                client_formula=client.sent_Formula(i,k,server_share,comp)
                

                #STEP3 server計算距離
                delta=self.computeDelta(centerIndex,k,client_share,client_formula,comp)

                
                deltaSum+=delta**2
            
            deltaSum=deltaSum**0.5

            distance=(deltaSum)

            index=binaryTree.findNearestGroup(distance)
            group=groups[index]
            group=np.array(group)

            data=[]
            labels=[]
            for d in group:
                data.append(d[0])
                labels.append(d[1])
            


            data=np.array(data)


            self.train_label=np.array(labels)
            self.data=data
            self.shares=ss_groups[index]


            self.k=5

            
            distance=[]

            for j in range(len(self.data)):
                deltaSum=0

                for k in range(self.data.shape[1]):
                    #STEP1  交換share
                    client_share=client.sent_share(i,k)

                    server_share=self.sent_share(j,k)

                    #STEP2 比較大小
                    comp=(self.compare(i,j,k,client))
                    #STEP2 client傳回結果
                    client_formula=client.sent_Formula(i,k,server_share,comp)
                    

                    #STEP3 server計算距離
                    delta=self.computeDelta(j,k,client_share,client_formula,comp)

                    
                    deltaSum+=delta**2
                
                deltaSum=deltaSum**0.5

                distance.append(deltaSum)

            result+=self.knn_classifier(distance)
        return result
    
    def decisionTree_method(self,client,classifier,leaf_index_dict,groups,d_indices):
        result=[]
        for test in client.data:
            _2dtest=test.reshape(1,-1)          #apply函數需要輸入2d array
            leaf_id=classifier.apply(_2dtest)
            leaf_id=int(leaf_id)                #轉回int
            d_list=leaf_index_dict[leaf_id]

            
            #找出與葉節點內所有代表號中，最近的代表號
            minimum=MINIMUM
            d_index=0
            for i in d_list:

                delegate=self.data[i]
                distance=((test-delegate)**2).sum()**0.5
                if(distance<minimum):
                    minimum=distance
                    d_index=i

            
            index=findGroupIndex(d_index,d_indices)
            
            group=groups[index]
            data=[]
            label=[]
            for d in group:
                data.append(d[0])
                label.append(d[1])
            result.append(knn_classifier(data,_2dtest,label,3,10))
            
        result=np.array(result).reshape(-1)
        return result
    
    def DCT_SS(self,client,classifier,leaf_index_dict,groups,d_indices):
        result=[]
        for i,test in enumerate(client.share):

            if(dataName=='iris'):
                leaf_id=traversal_DCT_iris(test)[1]
            elif(dataName=='wine'):
                leaf_id=traversal_DCT_wine(test)[1]
            elif(dataName=='breast_cancer'):
                leaf_id=traversal_DCT_breastCancer(test)[1]
            elif(dataName=='digits'):
                leaf_id=traversal_DCT_digits(test)[1]
                
            leaf_id=int(leaf_id)                #轉回int
            d_list=leaf_index_dict[leaf_id]
            
            
            #找出與葉節點內所有代表號中，最近的代表號
            minimum=MINIMUM
            d_index=0
            for j in d_list:

                delegate=self.data[j]
                distance=((client.data[i]-delegate)**2).sum()**0.5
                if(distance<minimum):
                    minimum=distance
                    d_index=j

            
            index=findGroupIndex(d_index,d_indices)
            
            group=groups[index]
            data=[]
            label=[]
            for d in group:
                data.append(d[0])
                label.append(d[1])
            result.append(knn_classifier(data,[client.data[i]],label,3,10))
            
        result=np.array(result).reshape(-1)
        return result
    '''

class Client:

    def __init__(self, test_X):
        self.data = test_X
        self.share = []
        self.query_num = len(self.data)
        
        for i in range(len(self.data)):
            temp = []
            instance = self.data[i]
            for j in range(len(instance)):
                temp.append( generateShares(n , t , instance[j]) )
            
            self.share.append(temp)
        
        self.share=np.array(self.share)

    # ==

    def sent_share(self, i, j):
        if( len(self.share) == 0 ):
            return
        else:
            # print('clinet share:', self.share[i][j])
            return self.share[i][j][0]                      # [3, y1]
    
    def get_share(self, i, j):
        if( len(self.share) == 0 ):
            return
        else:
            return self.share[i][j][1]                      # [5, y2]
    
    def compare(self, i, j, l2, server_share):          # server_share : [5, x2]

        x2 = server_share[1]
        y2 = self.get_share(i, j)[1]
        
        z2 = x2 - y2 + l2
        
        return z2

    def sent_difference(self, i, j, server_share, comp):
        if(comp):
            y = server_share[1] - self.get_share(i, j)[1]
        else:
            y = self.get_share(i, j)[1] - server_share[1]

        y = y % PRIME
        return y
    
    def sent_Formula(self, i, j, server_share, comp):       # y = xr + delta，用來計算差值。
        if(comp):
            y=self.share[i][j][0][1] - server_share[1]
        else:
            y=server_share[1] - self.share[i][j][0][1]
        
        y = y % PRIME
        x = self.share[i][j][0][0]
        return y, x

# ====

def acc_evaluate(result, test_y):
    
    correct_rate = 0

    if(len(result) != 0):
        incorrect = 0
        for j in range(len(result)):
            if( result[j] != test_y[j] ):
                incorrect = incorrect + 1
        
        correct_rate = ( len(result) - incorrect ) / len(result) * 100
    
    return correct_rate

def run_code(dataset):

    # mode = ['SS_knn' , 'knn' , 'tree' , 'dct' , 'dct_knn' , 'dct_SS_knn' , 'partition_dct_knn' , 'partition_dct_SS_knn']
    mode = ['SS_knn' , 'knn' , 'dct' , 'dct_knn' , 'dct_SS_knn' , 'partition_dct_knn' , 'partition_dct_SS_knn']
    mode = ['knn' , 'dct' , 'dct_knn' , 'dct_SS_knn' , 'partition_dct_knn' , 'partition_dct_SS_knn']
    # mode = 'SS_knn'

    data = dataset.data
    label = dataset.target
    NUM_CLASS = dataset.NUM_CLASS
    dataName = dataset.dataName

    # print('\n====\n')
    # print('資料集:{}'.format(dataName))
    # print('Instances: {} , Attributes: {} , Class: {}' .format( len(label) , len(data[0]) , NUM_CLASS ) )

    print('\n====\n', file=open('Client_Server__log.txt', 'a+'))
    print('資料集:{}'.format(dataName), file=open('Client_Server__log.txt', 'a+'))
    print('Instances: {} , Attributes: {} , Class: {}' .format( len(label) , len(data[0]) , NUM_CLASS ) , file=open('Client_Server__log.txt', 'a+'))


    epoch=2
    # epoch=5
    # epoch=10

    print('\nepoch e: ', end = '')

    # 每一種 mode 存放一組資料 [ mode_name, total_accuracy = 0 , total_time_cost = 0 ]
    mode_record = []

    if(isinstance(mode, list)):
        for i in range(len(mode)):
            mode_i = mode[i]
            mode_record.append([ mode_i , 0 , 0 ])

    else:        
        mode_record = [ mode , 0 , 0 ]
    
    for e in range(epoch):
        
        print(e , end = ' ')
        
        # 切分訓練與測試資料
        train_X, test_X, train_y, test_y = train_test_split(data, label, test_size = 0.2)

        server = Server(train_X , train_y , 5 , dataName , NUM_CLASS)
        client = Client(test_X)

        # 共用 partition
        groups, delegates_train, delegates_label = create_group(train_X, train_y, dataName)

        # ===========

        if(isinstance(mode, list)):
            for i in range(len(mode)):
                mode_i = mode[i]
                result , time_cost = run_epoch(server, client, mode_i, groups, delegates_train, delegates_label)
                accuracy = acc_evaluate(result, test_y)
                
                mode_record[i][1] += accuracy
                mode_record[i][2] += time_cost
        else:
            result , time_cost = run_epoch(server, client, mode, groups, delegates_train, delegates_label)
            accuracy = acc_evaluate(result, test_y)
            
            mode_record[1] += accuracy
            mode_record[2] += time_cost
    
    print()

    print('\nEpoch: ', epoch)
    print('\nEpoch: ', epoch , file=open('Client_Server__log.txt', 'a+'))

    if(isinstance(mode, list)):
        for i in range(len(mode)):
            print('Mode:' , mode_record[i][0] , ' 正確率 ' , mode_record[i][1] / epoch , '%' , ' 耗時 ' , mode_record[i][2] / epoch)
            print('Mode:' , mode_record[i][0] , ' 正確率 ' , mode_record[i][1] / epoch , '%' , ' 耗時 ' , mode_record[i][2] / epoch , file=open('Client_Server__log.txt', 'a+'))
    else:
        print('Mode:' , mode_record[0] , ' 正確率 ' , mode_record[1] / epoch , '%' , ' 耗時 ' , mode_record[2] / epoch)
        print('Mode:' , mode_record[0] , ' 正確率 ' , mode_record[1] / epoch , '%' , ' 耗時 ' , mode_record[2] / epoch , file=open('Client_Server__log.txt', 'a+'))

    return 0

def run_epoch(server, client, mode, groups, delegates_train, delegates_label):

    train_X = server.data
    train_y = server.train_label
    dataName = server.dataName
    NUM_CLASS = server.NUM_CLASS

    test_X = client.data

    time_cost = 0

    # ====

    time1=time.time()

    # server.method(client)：C sends [Q]_1 to S。( server algorithm 可用 client 進行溝通 )
    result = []

    if(mode == 'SS_knn'):                               # naive construction：正確性、耗時
        result = server.SS_knn(client)
    # elif(mode == 'tree'):                               # tree based secret shairng knn
    #     groups , binaryTree , centerIndex , ss_groups = Tree( train_X , test_X , train_y )
    #     result = server.tree_Method( client , groups , binaryTree , centerIndex , ss_groups )
    elif(mode == 'knn'):                                    # knn：正確性
        classifier = neighbors.KNeighborsClassifier(n_neighbors = 5, algorithm = 'brute')
        classifier = classifier.fit(train_X, train_y)
        result = classifier.predict(test_X)
    elif(mode == 'dct'):                                    # dct：正確性
        classifier = tree.DecisionTreeClassifier()
        classifier = classifier.fit(train_X, train_y)
        result = classifier.predict(test_X)
    elif(mode == 'dct_knn'):                                # knn based on decision tree：tree genertation time
        classifier, leaf_index_dict, leaf_predict_dict = create_dct(train_X, train_y, dataName, groups, delegates_train, delegates_label)
        result = server.dct_knn(client, classifier, leaf_index_dict, leaf_predict_dict, groups, delegates_train, delegates_label)
    elif(mode == 'dct_SS_knn'):                             # knn based on secure search decision tree：正確性、耗時
        classifier, leaf_index_dict, leaf_predict_dict = load_dct(dataName)
        result = server.dct_SS_knn(client, classifier, leaf_index_dict, leaf_predict_dict, groups, delegates_train, delegates_label)
    elif(mode == 'partition_dct_knn'):                      # decision tree with partition：tree genertation time
        classifier, leaf_index_dict = create_partition_dct(dataName, groups, delegates_train, delegates_label)
        result = server.partition_dct_knn(client, classifier, leaf_index_dict, groups, delegates_train, delegates_label)
    elif(mode == 'partition_dct_SS_knn'):                   # knn based on secure search partition decision tree：正確性、耗時
        classifier, leaf_index_dict = load_partition_dct(dataName)
        result = server.partition_dct_SS_knn(client, classifier, leaf_index_dict, groups, delegates_train, delegates_label)
    
    time2=time.time()

    time_cost = time2 - time1

    return result, time_cost

if __name__ == '__main__':

    print('', file=open('Client_Server__log.txt', 'w'))

    # dataName = ['iris' , 'wine' , 'breast_cancer' , 'abalone' , 'digits' , 'nursery' , 'mushroom' , 'Chess (King-Rook vs. King)']
    dataName = ['iris' , 'wine' , 'breast_cancer' , 'digits' , 'mushroom' , 'nursery']
    # dataName = ['iris' , 'wine' , 'breast_cancer']
    # dataName = ['digits' , 'mushroom' , 'nursery']
    dataName = 'digits'

    if(isinstance(dataName, list)):
        while(len(dataName) > 0):
            dataName_i = dataName.pop(0)
            dataset = loading_datasets(dataName_i)
            run_code(dataset)
    else:
        dataset = loading_datasets(dataName)
        run_code(dataset)

    print('\n====\n')
    print('\n====\n', file=open('Client_Server__log.txt', 'a+'))