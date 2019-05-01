## data.py
(data, label)の２つのカラムからなるCSV fileを受け取り以下のことを行う。

- vocab辞書を生成・保存。
- train, testそれぞれに対して、(data, label)のタプルを返すiteratorを生成・保存。


## dataloader.py
`data.py`で作成されたiteratorを受け取り、pytorchで実装されているdataloader instanceを
生成する関数が実装されている。
この中に、paddingを行う関数も入っており、dataloaderのiterateごとに実行される。

## models.py
RNN -> Linearという分類モデルが実装されている。

## train.py
dataloaderを生成し、modelを初期化し実際に学習を行うところ。
epochを回しきったら、test dataloaderを使用し評価に使用するtest dataに対しても推論を行い
結果を`pred.txt`に書き出すようになっている。
