# PPOutsourcedKNN

> This is an experiment based on the research of "Efficient Privacy Preserving Nearest Neighboring Classification from Tree Structures and Secret Sharing".

## 介紹

本研究的目標是處理保有隱私 kNN 使用大型資料集產生的執行效率問題。

首先，我們以 secret sharing 技術為基底，打造輕量級的密碼工具，提供外包式 kNN 分類服務的安全性。接著，提出基於樹的加速做法，以及基於決策樹的優化方案。最終，完成「基於樹結構和秘密共享的高效保有隱私最近鄰分類」研究。

## 研究論文

[Efficient Privacy Preserving Nearest Neighboring Classification from Tree Structures and Secret Sharing](./assets/documents/Efficient%20Privacy%20Preserving%20Nearest%20Neighboring%20Classification%20from%20Tree%20Structures%20and%20Secret%20Sharing.pdf)

- 出版於 IEEE ICC 2022：https://ieeexplore.ieee.org/document/9838718

- 投影片：[IEEE ICC 2022 - PPOutsourcedKNN ( Slides )](./assets/documents/IEEE%20ICC%202022%20-%20PPOutsourcedKNN%20(%20Slides%20).pptx)

## 檔案說明

- Client_Server.py：模擬系統運作的主要程式。

    - 產生 Client_Server__log.txt 紀錄執行結果。

- my_datasets.py：載入資料集，進行資料集的資料前處理，包括 "欄位整理"、"char -> int"。

    - 執行後，可以測試"整理後的資料集"在一般 KNN、一般 DCT 的分類效果。
    
    - 產生 my_datasets__log.txt 紀錄各資料集狀況，用以評估是否適合用以進行實驗。

- Partition_method.py：用來進行我們方案的資料前處理，包括 "分群"、"建立樹狀索引結構"、"SS_traversal_DCT"。

    - 執行後，可以測試"各個資料集的分群狀況"。
    
    - 產生 Partition_method__log.txt 紀錄分群參數和資料分佈，用以調整所需的實驗參數。

<details>

<summary>更多詳細內容</summary>

### 調用函數

- generateShares()

    - 描述：secret sharing 技術，將 secret 轉換成 share。

    - 來源：secret_sharing.py
    
    - 參數：n, t, secret

        > n：參與者數量。
        > t：還原門檻值。
        > secret：需為小於 PRIME ( = 2**19 - 1 ) 的整數值。
    
    - 回傳：shares

        > shares = [ [x, f(x)], ... ]，數量為 n。

- reconstructSecret()

    - 描述：secret sharing 技術，將 share 還原成 secret。

    - 來源：secret_sharing.py
    
    - 參數：shares

        > shares = [ [x, f(x)], ... ]，數量至少滿足 t 個。

    - 回傳：secret

- multiply()

    - 描述：Secure Secret Sharing Multiplication ( SSSM )，基於 2-out-of-2 的 secret sharing 安全乘法協議。

    - 來源：SS_multiplication_Protocol.py

    - 參數：x_shares, y_shares

    - 回傳：xy_shares

- loading_datasets()
    
    - 描述：從 datasets 資料夾 讀取原始資料，並轉換成適合本專案執行的資料格式。
    
    - 來源：my_datasets.py

    - 參數：dataName
        
        > 預設選項包括：dataName = ['iris' , 'wine' , 'breast_cancer' , 'digits' , 'abalone' , 'mushroom' , 'nursery' , 'Chess (King-Rook vs. King)']
    
    - 回傳：dataset
        ~~~
        dataset = Bunch(
            DESCR = description,
            data = data.values,
            target = target.values,
            NUM_CLASS = class,
            dataName = dataName
        )
        ~~~
    
    - 備註：本實驗皆使用 UCI Repository 所提供之原始資料，以利資料集遺失 or 有版本更新時，能快速地進行轉換。
    
    - 測試：執行 my_datasets.py，執行結果將顯示於 my_datasets__log.txt，包含 "資料集的相關訊息" 以及 "原始資料的 KNN、DCT 模型試算結果"。

- create_group()

    - 描述：製作分群資料。

    - 來源：Partition_method.py

    - 參數：train_data, train_label, dataName

        > 使用 dataName 讀取配置給資料集的適當代表數量 ( NUM_DELEGATE )、半徑 ( RADIUS ) 設定。
        > 
        > 目前提供的配置對象有：dataName = ['iris' , 'wine' , 'breast_cancer' , 'digits' , 'nursery' , 'mushroom']

    - 回傳：groups, delegates_train, delegates_label

        > groups：明文的分群資料。
        > delegates_train：各群代表的資料。
        > delegates_label：各群代表的類別。

    - 備註：每當新增資料集時，也需增加適配的分群參數，以利實驗程式的運行。
    
    - 測試：執行 來源：Partition_method.py，執行結果將顯示於 來源：Partition_method__log.txt，包含 "DCT 方案中，分群參數產生的資料量估算"。

        > 每個資料集適合的分群參數，需考慮資料數量、資料的分佈遠近，都會影響分群的狀況。更進一步，影響到 DCT 方案的預處理效果。

- Tree()

    - 描述：基於樹的預處理方案。

    - 來源：Partition_method.py

    - 參數：train_data, test_data, train_label,dataName

    - 回傳：groups, binaryTree, centerIndex, ss_groups

        > groups：明文的分群資料。
        > centerIndex：隨機選擇的距離基準點。
        > binaryTree：以 距離排序 的 二元搜索樹。
        > ss_groups：轉換成密文的分群資料。

- create_dct()、load_dct()

    - 描述：原始資料集 的 DCT 預處理。

    - 來源：Partition_method.py

    - 參數：train_X, train_y, dataName, groups, delegates_train, delegates_label
        
        ~~~
        以原始資料 train_X, train_y 建構 DCT classifier。
        ~~~

    - 回傳：classifier, leaf_index_dict, leaf_predict_dict
        
        ~~~
        建立每筆資料，在 DCT classifier 的索引與預測紀錄。
        ~~~

- create_partition_dct()、load_partition_dct()
    
    - 描述：分群資料 的 DCT 預處理。

    - 來源：Partition_method.py

    - 參數：dataName, groups, delegates_train, delegates_label
        
        ~~~
        以分群的代表資料 delegates_train, delegates_label 建構 DCT classifier。
        ~~~

    - 回傳：classifier, leaf_index_dict
        
        ~~~
        建立每筆分群代表資料，在 DCT classifier 的索引與預測紀錄。
        ~~~

- SS_traversal_DCT()

    - 描述：Server 將 DCT classifier 轉換成 secret sharing 密文，與 client 執行安全比較協議，完成安全的 DCT 預測。

    - 來源：Partition_method.py

    - 參數：test_instance_index, client, classifier
        
        > client 實體：可以與 client 交互，傳遞協作計算的 share。
        > test_instance_index：查詢資料的索引值。

    - 回傳：node_id

</details>

## 注意事項

1. 本系統雖然具備 (n, t) 的 secret sharing 設定，但安全協議只支援 (2, 2) 的 兩方運算版本，因此請勿修改相關設定。

2. 在 Client_Server.py 的 kNN 相關工作，k 值為可選數值，在此實驗使用的預設值為 5，可能潛在被寫成定值的設定，若需調整 "請重複檢查所有參數是否都以變數讀取"。

## Demo

主程式為【Client_Server.py】：模擬 Server 和 Client 實體，運行 PP Outsourced KNN 的各種行為，包括 share 計算、client 和 server 的溝通操作。

- 保有隱私外包式 k 最鄰近分類 ( Privacy Preserving Outsourced K Nearest Neighbor classification，PPOutsourcedKNN )。

    - Server 持有 Dataset，提供 kNN 分類服務。
    - Client 持有 query data，想得到分類結果。
    - 雙方都想避免自己的資料外洩給其他人，期望能透過保有隱私計算，形成安全服務。

- 本專案之目的為收集相關實驗數據，Server 提供下列運行模式：SS_knn , dct_knn , dct_SS_knn , partition_dct_knn , partition_dct_SS_knn。

    - SS_knn：直接以 secret sharing 做 kNN 的方案。
    - dct_knn：以 整個資料集 做 DCT 預處理部份。
    - dct_SS_knn：使用 整個資料集 的 DCT 預處理後，紀錄 SS_knn 的 執行狀況。
    - partition_dct_knn：以 分群資料 做 DCT 預處理部份。
    - partition_dct_SS_knn：讀取 分群資料 的 DCT 預處理後，紀錄 SS_knn 的 執行狀況。

- 執行結果，將記錄到 Client_Server__log.txt。
    
    - 以 Report_and_Problem 資料夾，整合紀錄所有的實驗數據。

### 運行流程、控制參數

Client_Server.py 的運行順序：\_\_main\_\_ -> run_code() -> run_epoch()。

1. \_\_main\_\_：選擇資料集，由左至右運行。
    
    - dataName = ['iris' , 'wine' , 'breast_cancer' , 'digits' , 'nursery' , 'mushroom']

2. run_code()：設定測試的功能項目、運行次數、資料集分割比例。

    - mode = [ 'SS_knn' , 'knn' , 'dct' , 'dct_knn', 'dct_SS_knn' , 'partition_dct_knn' , 'partition_dct_SS_knn' ]
    
        - SS_knn：測量 ( 正確性、耗時 )。
        - knn：測量 ( 正確性基準 )，一般 kNN，無安全性。
        - dct：測量 ( 正確性基準 )，一般 DCT，無安全性。
        - dct_knn：測量 ( 預處理 耗時 )。
        - dct_SS_knn：測量 ( 正確性、kNN 耗時 )。
        - partition_dct_knn：測量 ( 預處理 耗時 )。
        - partition_dct_SS_knn測量 ( 正確性、kNN 耗時 )。

    - epoch = 2，設定總輪數，計算實驗結果的平均狀況。
  
    - train_X, test_X, train_y, test_y = train_test_split( data, label, test_size = 0.2 )

        - 每一輪重新劃分資料集，但同一輪中的各種 mode 都會以相同的資料分割狀況、相同的分群狀況 來執行 run_epoch()。
      
3. run_epoch()：依照上述設定，執行實驗。

### 實驗結果

~~~

====

# report：others (桌機，epoch=10)

====

資料集:iris
Instances: 150 , Attributes: 4 , Class: 3

Epoch:  10
Mode: knn  正確率  96.66666666666666 %  耗時  0.0011041402816772462
Mode: dct  正確率  94.33333333333333 %  耗時  0.0007025003433227539
Mode: dct_knn  正確率  94.0 %  耗時  0.32570714950561525
Mode: dct_SS_knn  正確率  88.66666666666667 %  耗時  0.6111212253570557
Mode: partition_dct_knn  正確率  90.66666666666667 %  耗時  0.3321072101593018
Mode: partition_dct_SS_knn  正確率  88.66666666666667 %  耗時  0.6377087593078613

====

資料集:wine
Instances: 178 , Attributes: 13 , Class: 3

Epoch:  10
Mode: knn  正確率  68.33333333333334 %  耗時  0.0013001441955566406
Mode: dct  正確率  90.83333333333333 %  耗時  0.0006011724472045898
Mode: dct_knn  正確率  79.44444444444443 %  耗時  0.8682280540466308
Mode: dct_SS_knn  正確率  52.777777777777786 %  耗時  1.2628082752227783
Mode: partition_dct_knn  正確率  70.55555555555557 %  耗時  0.9519117832183838
Mode: partition_dct_SS_knn  正確率  65.27777777777779 %  耗時  2.615841031074524

====

資料集:breast_cancer
Instances: 569 , Attributes: 30 , Class: 2

Epoch:  10
Mode: knn  正確率  91.57894736842104 %  耗時  0.004412007331848144
Mode: dct  正確率  92.19298245614034 %  耗時  0.003799915313720703
Mode: dct_knn  正確率  91.57894736842105 %  耗時  8.55389883518219
Mode: dct_SS_knn  正確率  55.6140350877193 %  耗時  12.127776861190796
Mode: partition_dct_knn  正確率  87.10526315789475 %  耗時  8.412370538711547
Mode: partition_dct_SS_knn  正確率  72.98245614035088 %  耗時  26.867888426780702

====

資料集:digits
Instances: 1797 , Attributes: 64 , Class: 10

Epoch:  10
Mode: knn  正確率  98.55555555555557 %  耗時  0.014509797096252441
Mode: dct  正確率  84.75 %  耗時  0.011299395561218261
Mode: dct_knn  正確率  85.80555555555554 %  耗時  165.02260496616364
Mode: dct_SS_knn  正確率  4.888888888888889 %  耗時  4.528235864639282
Mode: partition_dct_knn  正確率  78.5 %  耗時  222.583691239357
Mode: partition_dct_SS_knn  正確率  71.91666666666667 %  耗時  397.9019126176834

====

資料集:mushroom
Instances: 8124 , Attributes: 22 , Class: 2

Epoch:  10
Mode: knn  正確率  99.91384615384615 %  耗時  0.19534366130828856
Mode: dct  正確率  100.0 %  耗時  0.005600547790527344
Mode: dct_knn  正確率  100.0 %  耗時  206.04541919231414
Mode: dct_SS_knn  正確率  82.11076923076924 %  耗時  831.819278550148
Mode: partition_dct_knn  正確率  90.4553846153846 %  耗時  180.03477954864502
Mode: partition_dct_SS_knn  正確率  87.32923076923076 %  耗時  848.6094256877899

====

資料集:nursery
Instances: 12960 , Attributes: 8 , Class: 5

Epoch:  10
Mode: knn  正確率  96.57021604938271 %  耗時  0.46597139835357665
Mode: dct  正確率  99.69135802469135 %  耗時  0.007006931304931641
Mode: dct_knn  正確率  98.97762345679013 %  耗時  395.59430117607116
Mode: dct_SS_knn  正確率  71.81712962962965 %  耗時  773.5896620035171
Mode: partition_dct_knn  正確率  95.22762345679013 %  耗時  447.44145998954775
Mode: partition_dct_SS_knn  正確率  71.88271604938271 %  耗時  810.7875809669495

====

~~~

## 開發紀錄

1. treebase version
  
    - 目標：利用 "分群"+"樹狀索引結構"，減少 kNN 計算量，達成 PPkNN 的 加速。
    - 問題：如何取得中心點，建構適合所有查詢資料的 tree？
        - 原本的 kNN，使用查詢資料的資料點，作為距離的基準，計算出各筆資料和查詢點的距離來進行周遭 k 分類。
        - 但今天要在還不確定查詢點下，得出一個距離，形成 tree，外包到 Server。
    - 結果：(隨機選擇中心點 / 以資料集的平均值為中心點) accuracy 都很低。
    
2. DCT version
  
    - 目標：利用 decision tree 的分類能力，做出適合所有查詢資料的 "樹狀索引結構"。
    - 做法：
        1. 以"全部資料"做出 decision tree 分類器。
        2. 紀錄 分群代表號 所在的 leaf。
        3. 以 decision tree 分類器，找出 查詢點 所在的 leaf。
        4. 找出與 leaf 中所有代表號，最近的代表號。
        5. 以 最近代表號之群 進行 SS_kNN。
    - 結果：
        - 成功解決：以預設中心點產生"樹狀索引"問題。
        - 衍伸問題：查詢資料跑到的 leaf，剛好不存在任何代表號，則無法進行 SS_kNN。

3. partition_dct_SS_knn version

    - 目標：直接使用 "分群資料" 做 dct 前處理，解決 leaf 可能不存在代表號的問題。
    - 做法：
        1. Partition 之後，以 群中資料的平均值 + label 的眾數 => 代表號。
        2. 以 "代表號" 做出 decision tree 分類器。(避免 leaf 不存在代表號)
        3. 將 DCT 分類器 轉換成 "樹狀索引"，讓 查詢資料 能快速的找到一些特徵上相近的代表號。(不需要讓 knn 檢查大量非相關資料)
            - 此處，DCT 索引 以 secret sharing 實作，確保 (查詢資料、decision tree 分類器) 的安全性。
        4. 用 SS_knn (換成以查詢資料為基準) 找出與葉節點內所有代表號中，最近的代表號。( 對 leaf 的代表號們，執行一次少量資料的 SS_knn )
        5. 以 最近代表號之群 進行 SS_kNN。( 對此最近的分群資料，執行一次少量資料的 SS_knn )
    - 結果：
        - 實驗數據儲存於【3. Report_( partition_dct_SS_knn ) - Our scheme 實驗結果)】
        - 比較：SS_knn 執行時間、正確性。
    - 未來討論：直接在代表號上做更多層分群，能收斂到像樹狀結構一樣。
        - 速度上可能更慢。(索引過程：雖然資料量相對更少，但每層都要用 SS_knn 找到最近的資料)
        - 相較而言，DCT 索引 更快速，而且能繼承 DCT 的分類效果。 
