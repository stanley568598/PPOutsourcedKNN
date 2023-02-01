
from sklearn import datasets
from sklearn.model_selection import train_test_split

import time
import numpy as np
import pandas as pd
from sklearn.utils import Bunch

from sklearn import tree
from sklearn import neighbors

from sklearn import metrics

def loading_datasets(dataName):

    print('\n====\n')
    print('資料集:{}'.format(dataName))

    if(dataName=='iris'):
        
        load_dataset=datasets.load_iris()
        
        '''
        f = open('./datasets/iris.names', 'r')
        description = f.read()
        # print(type(description))
        f.close

        data_pd = pd.read_csv('./datasets/iris.data', header=None)
        # print(data_pd.head())
        # print(type(data_pd.columns))

        data = data_pd.drop(data_pd.columns[-1], axis=1)
        target = data_pd[data_pd.columns[-1]]
        '''

        dataset = Bunch(
            DESCR = load_dataset.DESCR,
            data = load_dataset.data,
            target = load_dataset.target,
            NUM_CLASS = 3,
            dataName = dataName
        )

        # print(dataset.keys())
        # print(dataset.DESCR)

    elif(dataName=='wine'):

        load_dataset=datasets.load_wine()
        
        '''
        f = open('./datasets/wine.names', 'r')
        description = f.read()
        # print(type(description))
        f.close

        data_pd = pd.read_csv('./datasets/wine.data', header=None)
        # print(data_pd.head())
        # print(type(data_pd.columns))

        data = data_pd.drop(data_pd.columns[0], axis=1)
        target = data_pd[data_pd.columns[0]]
        '''

        dataset = Bunch(
            DESCR = load_dataset.DESCR,
            data = load_dataset.data,
            target = load_dataset.target,
            NUM_CLASS = 3,
            dataName = dataName
        )

        # print(dataset.keys())
        # print(dataset.DESCR)

    elif(dataName=='breast_cancer'):

        load_dataset=datasets.load_breast_cancer()
        
        '''
        f = open('./datasets/wdbc.names', 'r')
        description = f.read()
        # print(type(description))
        f.close

        data_pd = pd.read_csv('./datasets/wdbc.data', header=None)
        # print(data_pd.head())
        # print(type(data_pd.columns))

        # data_pd.columns = ["ID number","Diagnosis",...]
        diagnosis_mapping = { "M": 0 , "B": 1 }
        data_pd = data_pd.drop(data_pd.columns[0], axis=1)
        data_pd[data_pd.columns[0]] = data_pd[data_pd.columns[0]].map(diagnosis_mapping)

        # print(data_pd.head())

        data = data_pd.drop(data_pd.columns[0], axis=1)
        target = data_pd[data_pd.columns[0]]
        '''

        dataset = Bunch(
            DESCR = load_dataset.DESCR,
            data = load_dataset.data,
            target = load_dataset.target,
            NUM_CLASS = 2,
            dataName = dataName
        )

        # print(dataset.keys())
        # print(dataset.DESCR)

    elif(dataName=='digits'):
        
        load_dataset=datasets.load_digits()
        
        '''
        f = open('./datasets/optdigits.names', 'r')
        description = f.read()
        # print(type(description))
        f.close

        data_pd = pd.read_csv('./datasets/optdigits.tes', header=None)
        # print(data_pd.head())
        # print(type(data_pd.columns))
        
        data = data_pd.drop(data_pd.columns[-1], axis=1)
        target = data_pd[data_pd.columns[-1]]
        '''

        dataset = Bunch(
            DESCR = load_dataset.DESCR,
            data = load_dataset.data,
            target = load_dataset.target,
            NUM_CLASS = 10,
            dataName = dataName
        )
        
        # print(dataset.keys())
        # print(dataset.DESCR)

    elif(dataName=='mushroom'):

        f = open('./datasets/agaricus-lepiota.names', 'r')
        description = f.read()
        # print(type(description))
        f.close

        data_pd = pd.read_csv('./datasets/agaricus-lepiota.data', header=None)
        # print(data_pd.head())
        # print(type(data_pd.columns))

        org_columns = data_pd.columns
        data_pd.columns = ["poisonous","cap_shape","cap_surface","cap_color","bruises","odor","gill_attachment","gill_spacing","gill_size","gill_color","stalk_shape","stalk_root","stalk_surface_above_ring","stalk_surface_below_ring","stalk_color_above_ring","stalk_color_below_ring","veil_type","veil_color","ring_number","ring_type","spore_print_color","population","habitat"]
        poisonous_mapping = { "e": 1 , "p": 2 }
        cap_shape_mapping = { "b": 1 , "c": 2 , "x": 3 , "f": 4 , "k": 5 , "s": 6 }
        cap_surface_mapping = { "f": 1 , "g": 2 , "y": 3 , "s": 4 }
        cap_color_mapping = { "n": 1 , "b": 2 , "c": 3 , "g": 4 , "r": 5 , "p": 6 , "u": 7 , "e": 8 , "w": 9 , "y": 10 }
        bruises_mapping = { "t": 1 , "f": 2 }
        odor_mapping = { "a": 1 , "l": 2 , "c": 3 , "y": 4 , "f": 5 , "m": 6 , "n": 7 , "p": 8 , "s": 9 }
        gill_attachment_mapping = { "a": 1 , "d": 2 , "f": 3 , "n": 4 }
        gill_spacing_mapping = { "c": 1 , "w": 2 , "s": 3 }
        gill_size_mapping = { "b": 1 , "n": 2 }
        gill_color_mapping = { "k": 1 , "n": 2 , "b": 3 , "h": 4 , "g": 5 , "r": 6 , "o": 7 , "p": 8 , "u": 9 , "e": 10 , "w": 9 , "y": 10 }
        stalk_shape_mapping = { "e": 1 , "t": 2 }
        stalk_root_mapping = { "b": 1 , "c": 2 , "u": 3 , "e": 4 , "z": 5 , "r": 6 , "?": 0 }
        stalk_surface_above_ring_mapping = { "f": 1 , "y": 2 , "k": 3 , "s": 4 }
        stalk_surface_below_ring_mapping = { "f": 1 , "y": 2 , "k": 3 , "s": 4 }
        stalk_color_above_ring_mapping = { "n": 1 , "b": 2 , "c": 3 , "g": 4 , "o": 5 , "p": 6 , "e": 7 , "w": 8 , "y": 9 }
        stalk_color_below_ring_mapping = { "n": 1 , "b": 2 , "c": 3 , "g": 4 , "o": 5 , "p": 6 , "e": 7 , "w": 8 , "y": 9 }
        veil_type_mapping = { "p": 1 , "u": 2 }
        veil_color_mapping = { "n": 1 , "o": 2 , "w": 3 , "y": 4 }
        ring_number_mapping = { "n": 0 , "o": 1 , "t": 2 }
        ring_type_mapping = { "c": 1 , "e": 2 , "f": 3 , "l": 4 , "n": 0 , "p": 5 , "s": 6 , "z": 7 }
        spore_print_color_mapping = { "k": 1 , "n": 2 , "b": 3 , "h": 4 , "r": 5 , "o": 6 , "u": 7 , "w": 8 , "y": 9 }
        population_mapping = { "a": 1 , "c": 2 , "n": 3 , "s": 4 , "v": 5 , "y": 6 }
        habitat_mapping = { "g": 1 , "l": 2 , "m": 3 , "p": 4 , "u": 5 , "w": 6 , "d": 7 }
        data_pd["poisonous"] = data_pd["poisonous"].map(poisonous_mapping)
        data_pd["cap_shape"] = data_pd["cap_shape"].map(cap_shape_mapping)
        data_pd["cap_surface"] = data_pd["cap_surface"].map(cap_surface_mapping)
        data_pd["cap_color"] = data_pd["cap_color"].map(cap_color_mapping)
        data_pd["bruises"] = data_pd["bruises"].map(bruises_mapping)
        data_pd["odor"] = data_pd["odor"].map(odor_mapping)
        data_pd["gill_attachment"] = data_pd["gill_attachment"].map(gill_attachment_mapping)
        data_pd["gill_spacing"] = data_pd["gill_spacing"].map(gill_spacing_mapping)
        data_pd["gill_size"] = data_pd["gill_size"].map(gill_size_mapping)
        data_pd["gill_color"] = data_pd["gill_color"].map(gill_color_mapping)
        data_pd["stalk_shape"] = data_pd["stalk_shape"].map(stalk_shape_mapping)
        data_pd["stalk_root"] = data_pd["stalk_root"].map(stalk_root_mapping)
        data_pd["stalk_surface_above_ring"] = data_pd["stalk_surface_above_ring"].map(stalk_surface_above_ring_mapping)
        data_pd["stalk_surface_below_ring"] = data_pd["stalk_surface_below_ring"].map(stalk_surface_below_ring_mapping)
        data_pd["stalk_color_above_ring"] = data_pd["stalk_color_above_ring"].map(stalk_color_above_ring_mapping)
        data_pd["stalk_color_below_ring"] = data_pd["stalk_color_below_ring"].map(stalk_color_below_ring_mapping)
        data_pd["veil_type"] = data_pd["veil_type"].map(veil_type_mapping)
        data_pd["veil_color"] = data_pd["veil_color"].map(veil_color_mapping)
        data_pd["ring_number"] = data_pd["ring_number"].map(ring_number_mapping)
        data_pd["ring_type"] = data_pd["ring_type"].map(ring_type_mapping)
        data_pd["spore_print_color"] = data_pd["spore_print_color"].map(spore_print_color_mapping)
        data_pd["population"] = data_pd["population"].map(population_mapping)
        data_pd["habitat"] = data_pd["habitat"].map(habitat_mapping)
        data_pd.columns = org_columns
        
        # print(data_pd)
        
        data = data_pd.drop(data_pd.columns[0], axis=1)
        target = data_pd[data_pd.columns[0]]

        dataset = Bunch(
            DESCR = description,
            data = data.values,
            target = target.values,
            NUM_CLASS = 2,
            dataName = dataName
        )
        
        # print(dataset.keys())
        # print(dataset.DESCR)

    elif(dataName=='nursery'):
        
        f = open('./datasets/nursery.names', 'r')
        description = f.read()
        # print(type(description))
        f.close

        data_pd = pd.read_csv('./datasets/nursery.data', header=None)
        # print(data_pd.head())
        # print(type(data_pd.columns))

        org_columns = data_pd.columns
        data_pd.columns = ["parents","has_nurs","form","children","housing","finance","social","health","class"]
        parents_mapping = { "usual": 1 , "pretentious": 2 , "great_pret": 3 }
        has_nurs_mapping = { "proper": 1 , "less_proper": 2 , "improper": 3 , "critical": 4 , "very_crit": 5 }
        form_mapping = { "complete": 1 , "completed": 2 , "incomplete": 3 , "foster": 4 }
        children_mapping = { "1" : 1 , "2" : 2 , "3" : 3 , "more": 4 }
        housing_mapping = { "convenient": 1 , "less_conv": 2 , "critical": 3 }
        finance_mapping = { "convenient": 1 , "inconv": 2 }
        social_mapping = { "nonprob": 1 , "slightly_prob": 2 , "problematic": 3 }
        health_mapping = { "recommended": 1 , "priority": 2 , "not_recom": 3 }
        class_mapping = { "not_recom": 1 , "recommend": 2 , "very_recom": 3 , "priority": 4 , "spec_prior": 5 }
        data_pd["parents"] = data_pd["parents"].map(parents_mapping)
        data_pd["has_nurs"] = data_pd["has_nurs"].map(has_nurs_mapping)
        data_pd["form"] = data_pd["form"].map(form_mapping)
        data_pd["children"] = data_pd["children"].map(children_mapping)
        data_pd["housing"] = data_pd["housing"].map(housing_mapping)
        data_pd["finance"] = data_pd["finance"].map(finance_mapping)
        data_pd["social"] = data_pd["social"].map(social_mapping)
        data_pd["health"] = data_pd["health"].map(health_mapping)
        data_pd["class"] = data_pd["class"].map(class_mapping)
        data_pd.columns = org_columns
        
        # print(data_pd)

        data = data_pd.drop(data_pd.columns[-1], axis=1)
        target = data_pd[data_pd.columns[-1]]

        dataset = Bunch(
            DESCR = description,
            data = data.values,
            target = target.values,
            NUM_CLASS = 5,
            dataName = dataName
        )
        
        # print(dataset.keys())
        # print(dataset.DESCR)

    elif(dataName=='abalone'):
        f = open('./datasets/abalone.names', 'r')
        description = f.read()
        # print(type(description))
        f.close

        data_pd = pd.read_csv('./datasets/abalone.data', header=None)
        # print(data_pd.head())
        # print(type(data_pd.columns))
        
        # data_pd.columns = ["Sex","Length","Diameter","Height","Whole weight","Shucked weight","Viscera weight","Shell weight","Rings"]
        sex_mapping = { "M": 0 , "F": 1 , "I": 2 }
        data_pd[data_pd.columns[0]] = data_pd[data_pd.columns[0]].map(sex_mapping)

        # print(data_pd.head())

        # data_pd = data_pd.drop(data_pd.columns[0], axis=1)
        data = data_pd.drop(data_pd.columns[-1], axis=1)
        target = data_pd[data_pd.columns[-1]]

        dataset = Bunch(
            DESCR = description,
            data = data.values,
            target = target.values,
            NUM_CLASS = 29,
            dataName = dataName
        )
        
        # print(dataset.keys())
        # print(dataset.DESCR)
    
    elif(dataName=='Chess (King-Rook vs. King)'):
        f = open('./datasets/krkopt.info', 'r')
        description = f.read()
        # print(type(description))
        f.close

        data_pd = pd.read_csv('./datasets/krkopt.data', header=None)
        # print(data_pd.head())
        # print(type(data_pd.columns))
        
        org_columns = data_pd.columns
        data_pd.columns = ["White_King_col","White_King_row","White_Rook_col","White_Rook_row","Black_King_col","Black_King_row","class"]
        col_mapping = { "a": 1 , "b": 2 , "c": 3 , "d": 4 , "e": 5 , "f": 6 , "g": 7 , "h": 8 , "i": 9 , "j": 10 , "k": 11 , "l": 12 , "m": 13 , "n": 14 , "o": 15 , "p": 16 , "q": 17 , "r": 18 , "s": 19 , "t": 20 , "u": 21 , "v": 22 , "w": 23 , "x": 24 , "y": 25 , "z": 26 }
        class_mapping = { "draw": -1 , "zero": 0 , "one": 1 , "two": 2 , "three": 3 , "four": 4 , "five": 5 , "six": 6 , "seven": 7 , "eight": 8 , "nine": 9 , "ten": 10 , "eleven": 11 , "twelve": 12 , "thirteen": 13 , "fourteen": 14 , "fifteen": 15 , "sixteen": 16 }
        data_pd["White_King_col"] = data_pd["White_King_col"].map(col_mapping)
        data_pd["White_Rook_col"] = data_pd["White_Rook_col"].map(col_mapping)
        data_pd["Black_King_col"] = data_pd["Black_King_col"].map(col_mapping)
        data_pd["class"] = data_pd["class"].map(class_mapping)
        data_pd.columns = org_columns
        
        # print(data_pd)

        data = data_pd.drop(data_pd.columns[-1], axis=1)
        target = data_pd[data_pd.columns[-1]]

        dataset = Bunch(
            DESCR = description,
            data = data.values,
            target = target.values,
            NUM_CLASS = 18,
            dataName = dataName
        )
        
        # print(dataset.keys())
        # print(dataset.DESCR)

    data=dataset.data
    label=dataset.target
    # print(data)
    # print(label)
    
    NUM_CLASS=dataset.NUM_CLASS

    print( 'Instances: {} , Attributes: {} , Class: {}' .format( len(label) , len(data[0]) , NUM_CLASS ) )

    return dataset

def testing_dataset_DCT_KNN(dataset):

    # print(type(dataset))

    data=dataset.data
    label=dataset.target
    # print(data)
    # print(label)
    
    # epoch=1
    # epoch=10
    epoch=100
    
    DCT_acc=0
    DCT_time=0
    KNN_acc=0
    KNN_time=0
    
    for e in range(epoch):

        # 切分訓練與測試資料
        train_X, test_X, train_y, test_y = train_test_split(data, label, test_size = 0.2)

        # ===========

        # 建立 DCT 分類器
        DCT = tree.DecisionTreeClassifier()

        time1=time.time()
        # 訓練過程
        clf = DCT.fit(train_X, train_y)

        # 預測
        predicted = clf.predict(test_X)
        time2=time.time()

        # 績效
        accuracy = metrics.accuracy_score(test_y, predicted)

        # print('DCT: 正確率 ' , accuracy , ' 耗時 ' , time2-time1)
        DCT_acc = DCT_acc + accuracy
        DCT_time = DCT_time + (time2-time1)

        # ===========

        # 建立 KNN 分類器
        KNN = neighbors.KNeighborsClassifier(n_neighbors = 5, algorithm = 'brute')   
            # n_neighbors (default = 5)。
            # weights (default = 'uniform')：uniform weights，All points in each neighborhood are weighted equally。

        time1=time.time()
        # 訓練過程
        clf = KNN.fit(train_X, train_y)

        # 預測
        predicted = clf.predict(test_X)
        time2=time.time()

        # 績效
        accuracy = metrics.accuracy_score(test_y, predicted)

        # print('KNN: 正確率 ' , accuracy , ' 耗時 ' , time2-time1)
        KNN_acc = KNN_acc + accuracy
        KNN_time = KNN_time + (time2-time1)

    print('Average of epoch = ' , epoch)
    print('DCT: 正確率 ' , DCT_acc / epoch , ' 耗時 ' , DCT_time / epoch)
    print('KNN: 正確率 ' , KNN_acc / epoch , ' 耗時 ' , KNN_time / epoch)

    print('Average of epoch = ' , epoch , file=open('my_datasets__log.txt', 'a+'))
    print('DCT: 正確率 ' , DCT_acc / epoch , ' 耗時 ' , DCT_time / epoch , file=open('my_datasets__log.txt', 'a+'))
    print('KNN: 正確率 ' , KNN_acc / epoch , ' 耗時 ' , KNN_time / epoch , file=open('my_datasets__log.txt', 'a+'))

    # ===========

    return 0

# =========================================================================================

def save_loading_log(dataset):
    
    data = dataset.data
    label = dataset.target
    NUM_CLASS = dataset.NUM_CLASS
    dataName = dataset.dataName

    print('\n====\n', file=open('my_datasets__log.txt', 'a+'))
    print('資料集:{}'.format(dataName), file=open('my_datasets__log.txt', 'a+'))
    print( 'Instances: {} , Attributes: {} , Class: {}' .format( len(label) , len(data[0]) , NUM_CLASS ) , file=open('my_datasets__log.txt', 'a+') )
    
                   
if __name__ == '__main__':
    
    print('', file=open('my_datasets__log.txt', 'w'))

    # dataName = 按照原始knn速度的順序。
    dataName = ['iris' , 'wine' , 'breast_cancer' , 'digits' , 'abalone' , 'mushroom' , 'nursery' , 'Chess (King-Rook vs. King)']
    dataName = ['iris' , 'wine' , 'breast_cancer' , 'digits' , 'mushroom' , 'nursery']
    # dataName = 'iris'

    if(isinstance(dataName, list)):
        while(len(dataName) > 0):
            dataName_i = dataName.pop(0)
            
            dataset = loading_datasets(dataName_i)
            save_loading_log(dataset)

            testing_dataset_DCT_KNN(dataset)
    else:
        dataset = loading_datasets(dataName)
        save_loading_log(dataset)

        testing_dataset_DCT_KNN(dataset)

    print('\n====\n')
    print('\n====\n', file=open('my_datasets__log.txt', 'a+'))
