# KSVD
python3 を使ってスパースモデリングの辞書学習アルゴリズムの１つの K-SVD を実装しました。
K-SVD のなかの追跡アルゴリズムには直交マッチング追跡を用いています。
教材には[スパースモデリング](http://www.kyoritsu-pub.co.jp/bookdetail/9784320123946)を用い、
実装においては[こちらの記事](https://qiita.com/kibo35/items/67dedba4ea464cc494b0)を大変参考にさせて頂きました。

# 環境
Python3.6
macOS High Sierra 10.13.4

# インストール
使用しているライブラリを requirements.txt に記載しているので、下記のコマンドによって依存ライブラリはインストールできます。
```zsh
pip install -r requirements.txt
```

# 実験
K-SVD の性能を測るために、合成データに対する実験と実データに対する実験を行いました。

## 合成データによる実験
実験の設定は以下になります。
- 30 x 60 のランダムな辞書を生成（要素をガウス分布から iid 抽出し、列は正規化）
- 上記の辞書から 4,000 個の信号事例を生成
  - 各事例はランダムに選択された 4 個のアトム（辞書の一つの列）の線形結合であらわされる
  - 係数はガウス分布 N(0, 1) から抽出
  - 平均が 0 で分散が 0.1 のガウスノイズを加えたものとする
- スパースベクトルの非ゼロ要素の個数を 4 に固定する
- 学習回数は 50 回

実験結果は figure/result_of_ksvd.png になります。実際に動かして確認したい方は以下のコマンドで同じ結果を確認できます。
```zsh
python ksvd.py
```

## 実データによる実験
figure/barbara.pngを使用して辞書学習を行います。この実験の設定は以下のようになります
- barbara.png から 8 x 8 画素の画像パッチを抽出。←これをスパースに表現する辞書を学習する
- これらのパッチから一様に分布するように 1 / 10 のパッチを取り出して学習に使用する
- 64 x 64 の 2 次元分離可能 DCT 辞書を用いて辞書を初期化する
  - 8 x 11 の 1 次元 DCT 行列 A1D を作成
  - k 番目のアトム (k = 1, 2, ... , 11) は a_k = cos((i-1)(k-1)pi/11)(i = 1, 2, ..., 8)
  - 最初のアトム以外は平均を引き去り、クロネッカー積を用いて 64 x 64 の辞書を作成
- スパースベクトルの非ゼロ要素の数は 4 個
- 学習回数は 50 回

実験に使用したパッチの例は figure/barbara_patches.png で、DCT 辞書の例は figure/dictionary.png で確認できます。
学習した辞書は figure/ksvd_barbara_dic.png で、平均表現誤差の結果は figure/barbara_ksvd_err.png で確認できます。

プログラムを動かす場合は以下のコマンドで動かすことができます。
```zsh
python ksvd_experiments.py
```

※ numpy を使った特異値分解がとても遅いので、マルチコア対応の numpy で学習をしないとなかなか終わりません。
　 マルチコア対応を使用しても 1 時間ほどかかりますが。numpy と scikit-learn を比較した[こちらの記事](https://soralab.space-ichikawa.com/2016/11/python-svd/)などをご参考に。
  
  
