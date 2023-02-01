# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 18:23:04 2021

@author: 123
"""
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

index = 35

if __name__=='__main__':
    iris = datasets.load_digits()
    iris_data = iris.data
    iris_label = iris.target
    train_data , test_data , train_label , test_label = train_test_split( iris_data , iris_label , test_size = 0.2 , random_state = 0 )
    
    df = pd.DataFrame(train_data)
    # print( df.loc[ : , [index] ].to_numpy() )
    print( df.mode().to_numpy().tolist() )
    
    plt.hist( df.loc[ : , [index] ].to_numpy() , density = False, cumulative = False, label = "INDEX0" )
    plt.legend()
    plt.xlabel('Value')
    plt.show()
