# -*- coding:utf-8 -*-
# 参考: https://github.com/kibo35/sparse-modeling/blob/master/ch12.ipynb

import numpy as np
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from omp import OMP

def collum_normalizarion(A):
    """
    行列 A の列に関して正規化する
    
    A: 正規化したい行列(ndarray)
    """
    tmpA = A.copy()
    for m in range(A.shape[1]):
        tmpA[:, m] /= np.linalg.norm(tmpA[:, m])

    return tmpA

def KSVD(Y, m, k0, sig, iter_num, A0=None, initial_dictonary=None):
    """
    辞書学習アルゴリズムの K-SVD アルゴリズム
    
    Y: 観測した信号事例 n x m 行列 n は信号の次元を表し, m は事例の数を表す
    m: 求める辞書の列数
    k0: スパースベクトルの非ゼロ要素の数
    sig: ノイズのパラメータ
    iter_num: 反復回数
    A0: 真の辞書
    initial_dictonary: 初期辞書
    """

    # 初期化
    if initial_dictonary is None:
        A = Y[:, :m]
        A = collum_normalizarion(A)
    else:
        A = initial_dictonary

    log = []
    for _ in tqdm(range(iter_num)):
        # スパース符号化
        X = np.zeros((m, Y.shape[1]))
        eps = A.shape[0] * (sig ** 2)

        for i in range(Y.shape[1]):
            X[:, i], _ = OMP(A, Y[:, i], eps, k0)

        # K-SVD 辞書更新
        for j in range(m):
            omega = X[j, :] != 0
            X[j, omega] = 0
            Res_err = Y[:, omega] - np.dot(A, X[:, omega])
            U, S, V = np.linalg.svd(Res_err)
            A[:, j] = U[:, 0]
            X[j, omega] = S[0] * V.T[:, 0]

        val = np.linalg.norm(Y - np.dot(A, X))
        
        if A0 is not None:
            per = percent_of_recovering_atom(A, A0)
            log.append([val.mean(), per])
            print('mean error: {}, percent: {}'.format(val.mean(), per))
        else:
            log.append(val.mean())
            print('mean error: {}'.format(val.mean()))
            
        if val ** 2 < eps:
            break
        
    return A, np.array(log)

def percent_of_recovering_atom(A, A0, threshold=0.99):
    """
    アトムの復元率を測る
    
    A: K-SVD によって得られた辞書
    A0: 信号事例を生成するのに使われた辞書
    threshold: アトムを復元できたとみなす閾値
    """

    per = 0
    for m in range(A.shape[1]):
        a = A0[:, m]
        if np.abs(np.dot(a.T, A).max() > threshold):
           per += 1 

    per = (per * 100) / A.shape[1]
    return per
    
    
def main():
    """
    K-SVD アルゴリズムの性能を評価する
    """

    # ランダムな辞書 (30 x 60) を作成
    A0 = np.random.randn(30, 60)
    A0 = collum_normalizarion(A0)

    # 上記辞書から信号事例を 4000 個作成
    Y = np.zeros((30, 4000))
    k0 = 4
    sigma = 0.1

    for i in range(4000):
        Y[:, i] = np.dot(A0[:, np.random.permutation(60)[:k0]], np.random.rand(4)) + np.random.randn(30) * sigma

    # K-SVD により辞書を学習する
    iter_num = 20
    A, log = KSVD(Y, A0.shape[1], k0, sigma, iter_num, A0)
    print('log: ', log)
    print('log.shape: ', log.shape)
    # 得られた辞書の結果をプロットする
    fig = plt.figure(figsize=(8, 6))
    ax_err = fig.add_subplot(211)
    ax_per = fig.add_subplot(212)

    ax_err.plot(range(iter_num), log[:, 0])
    ax_err.set_xlabel('# of iteration')
    ax_err.set_ylabel('mean error')

    ax_per.plot(range(iter_num), log[:, 1])
    ax_per.set_xlabel('# of iteration')
    ax_per.set_ylabel('percentage of recovering atom')

    plt.show()

if __name__ == '__main__':
    main()
