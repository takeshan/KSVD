# -*- coding:utf-8 -*-
# 参考: https://github.com/kibo35/sparse-modeling/blob/master/ch03.ipynb

import numpy as np
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import *

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

        norm_r = np.linalg.norm(r) ** 2
        if norm_r < eps:
            break

    return x, S
    
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
            A, _ = collum_normalizarion(A) 
                
            # 非ゼロ要素の数が k0 個であるスパースなベクトル x を作成する
            x = np.zeros(A.shape[1])
            index = np.random.randint(0, A.shape[1], k0)
            for id in index:
                x[id] = np.random.rand() + 1
                if np.random.rand() < 0.5:
                    x[id] *= -1
        
            # 上記の x を用いて, 観測ベクトル b を生成する
            b = np.dot(A, x)

            # サポート S 作成
            S = np.zeros(A.shape[1])
            S[index] = 1
            
            _x, _S = OMP(A, b, eps, k0)
            errs[k0] += l2_err(x, _x)
            dists[k0] += support_distance(S, _S)


        errs[k0] /= k
        dists[k0] /= k

        print('k0: {}, err: {}, dist: {}'.format(k0, errs[k0], dists[k0]))
        
    # plot
    fig = plt.figure(figsize=(6, 8))
    ax_L2 = fig.add_subplot(211)
    ax_L2.set_xlabel('# of non zero components')
    ax_L2.set_ylabel('the error of average L2')
    ax_L2.plot(K0, errs[1:])
    ax_L2.set_xlim(1, 10)
    ax_L2.set_ylim(0, 1.0)
    ax_L2.set_yticks(np.array(range(11)) * 0.1)

    ax_dist = fig.add_subplot(212)
    ax_dist.set_xlabel('# of non zero components')
    ax_dist.set_ylabel('distance between supports')
    ax_dist.plot(K0, dists[1:])    
    ax_dist.set_xlim(1, 10, 1)
    ax_dist.set_ylim(0, 0.8, 0.1)
    
    plt.show()
    

if __name__ == '__main__':
    main()
