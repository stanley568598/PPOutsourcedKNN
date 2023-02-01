# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 17:15:30 2021

@author: 123
"""
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import os
import time


from multiplyProtocol import multiply
from knn import knn_classifier
from secret_sharing import generateShares,reconstructSecret

from utility import tree
import pandas as pd


q = 997
ratio = 1/5
NUM_CLASS = 10

class Client:

    def __init__( self , data ):
        self.data = data
        self.share = []
        
        for i in range( len ( self.data ) ):
            temp = []
            for j in range( len ( self.data[i] ) ):
                temp.append( generateShares( t , n , self.data[i][j] ) )
            
            self.share.append( temp )

        self.share = np.array( self.share )
    
    def sent_share( self , i , j ):

        if( len ( self.share ) == 0 ):
            return

        else:
            return self.share[i][j][1]
        
    def sent_Formula( self , i , j , server_share , comp ):     # y = xr + delta

        if( comp ):
            y = self.share[i][j][0][1] - server_share[1]
        else:
            y = server_share[1] - self.share[i][j][0][1]
        
        y = y % q
        x = self.share[i][j][0][0]

        return y , x
    
    def compare( self , i , j , server_share ):

        x = self.share[i][j][0][1]
        y = server_share[1]
        
        l = np.random.randint( max( x , y ) , max( x , y ) + 100 )
        
        z = x - y + l

        return z , l


class Server:

    def __init__( self , data , train_label , k ):
        self.data = data
        self.share = self.transToShare( self.data )
        self.train_label = train_label
        self.k = k

    def transToShare( self , data ):
        
        share = []

        for i in range( len( self.data ) ):
            
            temp = []
            for j in range( len( self.data[i] ) ):
                temp.append( generateShares( t , n , self.data[i][j] ) )

            share.append( temp )
            
        return np.array(share)
    
    def sent_share( self , i , j ):

        if( len( self.share ) == 0 ):
            return
        else:
            return self.share[i][j][0]

    def computeDelta( self , i , j , client_share , client_formula , comp ):
        
        if( comp ):
            y1 = client_share[1] - self.share[i][j][1][1]
        else:
            y1 = self.share[i][j][1][1] - client_share[1]
        
        y1 = y1 % q
        x1 = self.share[i][j][1][0]

        y2 , x2 = client_formula
        
        r = ( y2 - y1 ) / ( x2 - x1 )
        
        delta = y1 - x1 * r

        delta = np.round( delta , 1 )

        delta = delta % q
        
        if( delta > 100 ):
            
            r = ( y2 - y1 ) * 499 % q
        
            delta = y1 - x1 * r
    
            delta = np.round( delta , 1 )

            delta = delta % q
            
        return np.round( delta , 1 )
    
    def knn( self , client ):

        result = []
        for i in range( len( client.data ) ):

            distance = []

            for j in range( len ( self.data ) ):
                
                deltaSum = 0

                for k in range( self.data.shape[1] ):

                    # STEP1  交換share
                    client_share = client.sent_share( i , k )
                    server_share = self.sent_share( j , k )
                    
                    # STEP2 比較大小
                    comp = ( self.compare( i , j , k , client ) )

                    # STEP2 client傳回結果
                    client_formula = client.sent_Formula( i , k , server_share , comp )

                    # STEP3 server計算距離
                    delta = self.computeDelta( j , k , client_share , client_formula , comp )

                    deltaSum += delta ** 2   # ** = 指數運算
                
                deltaSum = deltaSum ** 0.5   # ** 0.5 = 開根號

                distance.append( deltaSum )

            result += self.knn_classifier( distance )
            # print( result )

        return(result)
        
    def  knn_classifier( self , distance ):

        result = []
        DAL = {}                                                  # DAL = DistanceAndLabel

        for j in range( len( distance ) ):
            
            d = distance[j]
            DAL[d] = self.train_label[j]

        SDAL = sorted( DAL.items() , key = lambda x:x[0] )        # SDAL = SortedDistanceAndLabel : ( 將 每個train data 離 test data的距離 由小到大 排序 )    
        
        predict_label = {}
        for i in range( NUM_CLASS ):
            predict_label.update( { i : 0 } )
            
        KSDAL = SDAL[ 0 : self.k ]
        
        if( len( SDAL ) < self.k ):
            self.k = len(SDAL)

        for i in range( self.k ):
            predict_label[ ( KSDAL[i] ) [1] ] += 1

        SPL = sorted( predict_label.items() , key = lambda x:x[1] , reverse = True )

        result.append( SPL[0][0] )
        
        return result
            
    def compare( self , i , j , k , client ):

        z1 , l1 = client.compare( i , k , self.sent_share( j , k ) )
        
        x2 = client.sent_share( i , k )[1]
        y2 = self.share[j][k][1][1]

        l2 = np.random.randint( max(x2 , y2) , max(x2 , y2) + 100)
        
        z2 = x2 - y2 + l2
        
        r = np.random.randint( 1 , 50 )
        _r = np.random.randint( 1 , 50 )
        
        r_share = generateShares( t , n , r )
        _r_share = generateShares( t , n , _r )
        
        r1 = r_share[0][1]
        r2 = r_share[1][1]
        
        _r1 = _r_share[0][1]
        _r2 = _r_share[1][1]
        
        zr = multiply ( [ [ 3 , z1 ] , [ 5 , z2 ] ] , [ [ 3 , r1 ] , [ 5 , r2 ] ] )
        lr = multiply ( [ [ 3 , l1 ] , [ 5 , l2 ] ] , [ [ 3 , r1 ] , [ 5 , r2 ] ] )
        
        s1 = zr + _r1
        s2 = zr + _r2
        
        h1 = lr + _r1
        h2 = lr + _r2
        
        s = reconstructSecret( [ [ 3 , s1 ] , [ 5 , s2 ] ] )
        h = reconstructSecret( [ [ 3 , h1 ] , [ 5 , h2 ] ] )
        
        return s > h
        
    def Delegate_Method( self , client ):

        groups_average , share_gas , labels = delegate( self.data , self.train_label )

        """
        result = []
        for test in client.data:
            d = []
            for g in groups_average:
                distance = 0
                distance = test - g
                distance *= distance
                distance = distance.sum()
                distance = distance ** 0.5
                d.append( distance )
            result.append( labels[ np.argmin(d) ] )
            
        return result
        """

        self.train_label = labels
        self.data = groups_average
        self.share = share_gas

        self.k = 1
        result = self.knn( client )

        print(result)
        return result
                   
    """
    def tree_Method( self , client , groups , binaryTree , centerIndex , ss_groups ):
        
        # 計算distance
        distance = 0
        center_ss = self.share[ centerIndex ]

        result = []
        origin_data = self.data
        origin_label = self.train_label
        origin_share = self.share

        for i in range( len( client.data ) ):
            
            self.data = origin_data
            self.label = origin_label
            self.share = origin_share

            distance = 0
            deltaSum = 0

            for k in range( self.data.shape[1] ):
                
                # STEP1  交換share
                client_share = client.sent_share( i , k )
                server_share = self.sent_share( centerIndex , k )
                
                # STEP2 比較大小
                comp = ( self.compare ( i , centerIndex , k , client ) )
                # STEP2 client傳回結果
                client_formula = client.sent_Formula( i , k , server_share , comp )
                
                # STEP3 server計算距離
                delta = self.computeDelta( centerIndex , k , client_share , client_formula , comp )

                deltaSum += delta ** 2
            
            deltaSum = deltaSum ** 0.5

            distance = ( deltaSum ) 

            index = binaryTree.findNearestGroup( distance )
            
            group = groups[ index ]
            group = np.array( group )

            data = []
            labels = []
            for d in group:
                data.append( d[0] )
                labels.append( d[1] )
            
            data = np.array(data)

            self.train_label = np.array(labels)
            self.data = data
            self.shares = ss_groups[index]

            self.k = 5

            distance = []

            for j in range( len(self.data) ):
                
                deltaSum = 0

                for k in range( self.data.shape[1] ):

                    # STEP1  交換share
                    client_share = client.sent_share( i , k )
                    server_share = self.sent_share( j , k )

                    # STEP2 比較大小
                    comp = ( self.compare( i , j , k , client ) )
                    # STEP2 client傳回結果
                    client_formula = client.sent_Formula( i , k , server_share , comp )
                    
                    # STEP3 server計算距離
                    delta = self.computeDelta( j , k , client_share , client_formula , comp )

                    deltaSum += delta ** 2
                
                deltaSum = deltaSum ** 0.5

                distance.append( deltaSum )

            result += self.knn_classifier( distance )

        return result
    """

    def tree_Method( self , client , groups , binaryTree ):

        center = pd.DataFrame(client.data).mean().to_numpy().squeeze()

        result = []

        groups_size = []
        for g in groups:
            g = np.array( g , dtype = object )
            groups_size.append(len(g))
        print(groups_size)

        for i in range( len(client.data) ):
            
            distance = ( ( center - client.data[i] ) ** 2 ).sum() ** 0.5 
            
            index = binaryTree.findNearestGroup( distance )
            group = groups[index]
            group = np.array( group , dtype = object )
            
            ## add by stanley
            # if i == 0:
            #     print(group)
            #     print(len(group))
            
            new_train_data = []
            new_test_data = []
            new_train_label = []
            new_test_data.append(client.data[i])
            for data in group:
                new_train_data.append(data[0])
                new_train_label.append(data[1])

            new_result = knn_classifier( new_train_data , new_test_data , new_train_label , 5 , NUM_CLASS )
            result.append( new_result[0] )
            ##

            ## l = np.zeros( 10 , dtype = int )

            ## for data in group:
            ##     l[ data[1] ] += 1

            ## result.append( np.argmax(l) )    # l [] 的 最大值 的 index
        
        return result


# 使用 Tree，跑 digits 資料

if __name__ == '__main__':

    iris = datasets.load_digits()
    iris_data = iris.data
    iris_label = iris.target
    
    n , t = 2 , 2

    for times in range(5):

        train_data , test_data ,  train_label ,  test_label = train_test_split( iris_data , iris_label , test_size = 0.2 )

        groups , binaryTree , center = tree( train_data , test_data , train_label )

        origin_knn = test_label


        """
        # 原始knn
        time1 = time.time()

        result = knn_classifier( train_data , test_data , train_label , 5 , NUM_CLASS )

        incorrect = 0
        for i in range( len(result) ):
            if( origin_knn[i] != test_label[i] ):
                incorrect += 1

        print('correct rate : ' , ( len(result) - incorrect ) / len(result) * 100 , '%')

        time2 = time.time()

        t1 = time2 - time1

        print( '原始 knn，耗時:' , t1 , '秒' )

        print()
        """

        # Tree
        time1 = time.time()

        client = Client( test_data )
        server = Server( train_data , train_label , 5 )
        
        # result = server.Delegate_Method( client )
        result = server.tree_Method( client , groups , binaryTree )

        # print( origin_knn.tolist() )
        # print()
        # print( result )

        incorrect = 0
        for i in range( len(result) ):
            if( result[i] != test_label[i] ):
                incorrect += 1
        
        time2 = time.time()
        
        t1 = time2 - time1

        print()

        print('correct rate : ' , ( len(result) - incorrect ) / len(result) * 100 , '%')

        print('使用 Tree，耗時 : ' , t1 , '秒')

        print()
        
        """
        train_data,train_label = PreProcessor( train_data , train_label )
        
        time1 = time.time()
        client = Client( test_data )
        server = Server( train_data , train_label , 5 , 3 )
        result = server.knn( client )
        result1 = knn_classifier( train_data , test_data , train_label , 5 , 3 )
        
        incorrect = 0
        for i in range( len(result) ):
            if( result[i] != result1[i] ):
                incorrect += 1
        
        time2 = time.time()
        
        t2 = time2 - time1
        print('使用代表號，耗時:' , t2 , '秒')
        print('節省了:' , ( 1 - t2 / t1 ) * 100 , '%的時間')
        
        print('correct rate:' , ( len(result) - incorrect ) / len(result) * 100 , '%')
        """
