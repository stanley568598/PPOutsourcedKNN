import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import time
from math import *

from knn import knn_classifier
from secret_sharing import generateShares


TRAIN_LENGTH = 120
TEST_LENGTH = 30

# def client( s1 , s2 ):
    
def compare( trainSet , testSet ):

    # data1: train 120
    # data2: test 30

    comp = []

    for test in testSet:
        for train in trainSet:
            distance = []
            for i in range( len(train) ):
                difference = 0
                for j in range( len( train[i] ) ):
                    delta = test[i][j][1] - train[i][j][1]
                    difference += abs(delta)
                distance.append(difference)

            total_distance = 0
            for d in distance:
                total_distance += d

            comp.append( total_distance )

    comp = np.array( comp )
    return comp


def knn( comp , train_label , k ):

    result = []
    for i in range( TEST_LENGTH ):
        distance_dict = {}
        for j in range( TRAIN_LENGTH ):
            distance_dict[ comp[ i * 120 + j ] ] = train_label[ j ]
        
        distance_dict = sorted( distance_dict.items() , key = lambda x:x[0] )
        
        k_label = list(distance_dict)[ 0 : k ]
        # print( k_label )        
        
        label_dict = { 0 : 0 , 1 : 0 , 2 : 0 }
        for j in range(k):
            label_dict[ k_label [j][1] ] += 1
        # print(label_dict)
        
        label = sorted( label_dict.items() , key = lambda x:x[1] , reverse = True )
        
        result.append( label[0][0] )
    
    return result

    #  for i in range( TEST_LENGTH ):    
        # distance = comp[ i * 120 : i * 120 + 120 ]


if __name__ == '__main__':

    iris = datasets.load_iris()
    iris_data = iris.data
    iris_label = iris.target

    n , t = 2 , 2

    t1 = time.time()
    false = 0
    for times in range(1):
        train_data , test_data , train_label , test_label = train_test_split( iris_data , iris_label , test_size = 0.2 )
        
        SS_trainData = []
        for i in range( len(train_data) ):
            temp = []
            for j in range( len(train_data[i]) ):
                temp.append( generateShares( t , n , train_data[i][j] ) )
            SS_trainData.append( temp )
            
        SS_trainData = np.array(SS_trainData)

        SS_testData = []
        for i in range( len(test_data) ):
            temp = []
            for j in range( len(test_data[i]) ):
                temp.append( generateShares( t , n , test_data[i][j] ) )
            SS_testData.append( temp )
        
        SS_testData = np.array(SS_testData)
        print( SS_trainData.tolist() )
        print(test_label.tolist())

        # ans = compare( SS_trainData , SS_testData )
        # result = knn( ans , train_label , 4 )
        # print( result )
    
        '''
        result = knn_classifier( train_data , test_data , train_label , 5)
        print( result )
        print( test_label.tolist() )

        # 計算正確率
        for i in range( len(result) ):
            if( result[i] != test_label.tolist()[i] ):
                false += 1
        
        print( false )
        print( 'Correct Rate : ' + str( ( len(result) - false ) / len(result) * 100 ) + "%" )
        
        t2 = time.time() 
        print( 'time elapsed : ' + str( t2 - t1 ) + ' seconds' ) 
        '''