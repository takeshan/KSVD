# -*- coding:utf-8 -*-

import numpy as np
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from tqdm import tqdm


def OMP(A, b, eps, k0):
    """
    直行マッチング追跡 (orthogonal matching pursuit: OMP) 
    
    タスク: (P0) minx ||x0|| subject to b = Ax の近似解を求める
    A: n x m 行列
    b: n 要素の観測結果
    eps: 誤差の閾値
    k0: 非ゼロ要素の個数
    """

    # 初期化
    m = A.shape[1]
    x = np.zeros(m)
    S = np.zeros(m)
    r = b.copy()

    for _ in range(k0):
        # 誤差の計算
        idx = np.where(S == 0)[0]
        err = np.dot(r, r) - np.dot(A[:, idx].T, r) ** 2

        # サポートの更新
        S[idx[err.argmin()]] = 1
        
        # 暫定解の更新
        As = A[:, S == 1] 
        x[S == 1] = np.dot(np.linalg.pinv(As), b)

        # 残差の更新
        r = b - np.dot(A, x)
        norm_r = np.linalg.norm(r)

        if norm_r < eps:
            break

    return x, S

def evaluate(x, _x, S, _S):
    """
    OMP の成否を L2 誤差とサポートの復元度合いの２つで評価する
    
    x: 厳密解
    _x: 近似解
    S: 厳密解のサポート
    _S: 近似解のサポート
    """

    # L2 誤差を評価
    err = np.linalg.norm(x - _x) / np.linalg.norm(x) ** 2

    # サポート間の復元度合いの評価
    dist = support_distance(S, _S)

    return err, dist
    

def support_distance(S, _S):
    """
    サポート間の距離を計算する
    定義: dist(S, _S) = (max{|S|, |_S|} - |S and _S|) / max{|S|, |_S|}
    スパースモデリング 第3章 p55 より
    
    S: 厳密解のサポート
    _S: 近似解のサポート
    """

    val = max(np.sum(S), np.sum(_S))

    return float(val - np.sum(S * _S)) / val
    


def main():
    """
    OMP の性能をチェックする
    """
    # eps = 1E-4 とし, 反復回数を 1000 回として、非ゼロ要素の個数を変化させながら OMP を評価する
    num = 10
    K0 = np.array(range(1, num + 1))
    eps = 1e-4
    k = 1000

    errs = np.zeros(num + 1)
    dists = np.zeros(num + 1)
    
    for k0 in tqdm(K0):
        for _ in range(k):
            # ランダムな行列 A (30 x 50) を作成し, L2 正規化する
            A = (np.random.rand(30, 50) - 0.5) * 2
            for m in range(A.shape[1]):
                A[:, m] = A[:, m] / np.linalg.norm(A[:, m])
                
            # 非ゼロ要素の数が k0 個であるスパースなベクトル x を作成する
            x = np.zeros(A.shape[1])
            S = np.zeros(A.shape[1])
            tmp = np.random.rand(A.shape[1]) + 1     
            for i in range(k0):
                id = np.random.randint(0, A.shape[1])
                x[id] = tmp[id] if tmp[id] >= 0.5 else -1 * tmp[id]
                S[id] = 1
        
            # 上記の x を用いて, 観測ベクトル b を生成する
            b = np.dot(A, x)
            
            _x, _S = OMP(A, b, eps, k0)
            err, dist = evaluate(x, _x, S, _S)

            errs[k0] += err
            dists[k0] += dist

        errs[k0] /= k
        dists[k0] /= k
      
    # plot
    fig = plt.figure(figsize=(15, 3))
    ax_L2 = fig.add_subplot(211)
    ax_L2.set_xlabel('# of non zero components')
    ax_L2.set_ylabel('the error of average L2')
    ax_L2.plot(K0, errs[1:])

    ax_dist = fig.add_subplot(212)
    ax_dist.set_xlabel('# of non zero components')
    ax_dist.set_ylabel('distance between supports')
    ax_dist.plot(K0, dists[1:])    

    plt.show()
    

if __name__ == '__main__':
    main()
