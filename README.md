# PPKNN

透過 vscode 開啟資料夾。

執行 Client_Server.py：模擬 PPKNN 運行過程，包括 share 計算、client 和 server 的溝通操作。

實驗結果，將會記錄到 Client_Server__log.txt。

## 實驗參數 設定與選擇

- my_datasets.py：用來載入原始資料集，整理成系統所能正常運行的 dataset。(資料夾 datasets 皆為 UCI Repository 所提供之原始資料，以利當本專案所提供之資料集遺失 or 有更新版本的資料集出現時，能快速地透過本專案進行實驗)

- Partition_method.py：設定各資料集的分群參數(代表號數量、半徑)。[請注意，每當新增資料集時，也需增加適配的分群參數，以利其他程式運行]

- Client_Server.py：kNN 的 k值 理論上可調整 (但可能有地方被我寫死成定值 5，建議 double check)；secret sharing 的 n,t 只支援 2,2 (請勿任意修改)。

### 實驗結果 資料收集

Client_Server.py 運行順序：__ main __ -> run_code() -> run_epoch()。

1. __ main __：本專案提供下列資料集，將依序由左至右運行，可只保留所需的資料集。
   
   - dataName = ['iris' , 'wine' , 'breast_cancer' , 'abalone' , 'digits' , 'nursery' , 'mushroom' , 'Chess (King-Rook vs. King)']

2. run_code()：可設定測試的功能項目、運行次數、資料集分割比例。
    
    - mode = ['SS_knn' , 'knn' , 'dct' , 'dct_knn' , 'partition_dct_knn' , 'partition_dct_SS_knn']
      - SS_knn：naive construction，直接以 secret sharing 做 kNN 的方案。【 測量：正確性、耗時 】
      - knn：一般 kNN (無安全性)，用來評估 正確性。
      - dct：一般 DCT (無安全性)，用來評估 正確性。
      - dct_knn：[DCT version]，完整資料DCT，SS_knn based on DCT分類器，(無 查詢資料、DCT分類器 安全性；可能有不存在代表號的leaf -> 用原本leaf predict label 作為結果)。【 測量：建構DCT + 分群資料 SS_knn 正確率、耗時 => 建構DCT的預處理時間(通常只花費一次) 】
      - partition_dct_knn：分群資料DCT，SS_knn based on DCT分類器，(無 查詢資料、DCT分類器 安全性)。【 測量：分群資料 建構DCT + 分群資料 SS_knn 正確率、耗時 => 建構DCT的預處理時間(通常只花費一次) 】
      - partition_dct_SS_knn：[Our scheme]，分群資料DCT，SS_knn based on DCT索引 (dct secure search)。【 測量：分群資料 DCT索引 + 分群資料 SS_knn 正確率、耗時】

    - epoch=2，設定總輪數，計算實驗結果的平均狀況。
  
    - train_X, test_X, train_y, test_y = train_test_split(data, label, test_size = 0.2)
      - 每一輪重新劃分資料集，但同一輪中的各種 mode 都會以相同的資料集分割狀況、相同的分群狀況 來執行 run_epoch()。
      - 建議：以 my_datasets.py 檢查改動後分類效果是否在可接受範圍。
    
3. run_epoch()：依照上述設定，執行一輪實驗。
