# -*- coding:utf-8 -*-
# 参考: https://github.com/kibo35/sparse-modeling/blob/master/ch12.ipynb

import numpy as np
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from omp import OMP
from utils import *

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
        A, W = collum_normalizarion(A)
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
        for j0 in range(m):
            omega = X[j0, :] != 0
            if np.sum(omega) == 0:
                continue
            X[j0, omega] = 0
            Res_err = Y - np.dot(A, X)
            Res_err = Res_err[:, omega]

            U, S, V = np.linalg.svd(Res_err)
            A[:, j0] = U[:, 0]
            X[j0, omega] = S[0] * V.T[:, 0]
            
        val = np.linalg.norm(Y - np.dot(A, X)) ** 2
        A = fix_dictionary(Y, A, X)
        
        if A0 is not None:
            per = percent_of_recovering_atom(A, A0, 0.99)
            mean_err = np.abs(Y - np.dot(A, X)).mean()
            log.append([mean_err, per])
            print('mean error: {}, percent: {}'.format(mean_err, per))
        else:
            mean_err = np.abs(Y - np.dot(A, X)).mean()
            log.append(mean_err)
            print('mean error: {}'.format(mean_err))

        if val < eps:
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
        if np.abs(np.dot(a.T, A)).max() > threshold:
           per += 1 

    return float(per) / A.shape[1] * 100

def fix_dictionary(Y, A, X):
    """
    K-SVD の精度を向上させるために更新の際に辞書を修正する
    * 修正の方針
    辞書中のアトムが他のアトムと似ている or ほとんど利用されていない なら
    最も表現誤差が大きい正規化した事例と取りかえる
    "ほとんど利用されていないアトム" は事例集合が空集合かどうかで判断する

    Y: 信号事例
    A: 更新している辞書
    X: スパースベクトル
    """
    # 正規化
    normY, _ = collum_normalizarion(Y)
    normA, _ = collum_normalizarion(A)
    
    # グラム行列を使うことでアトムの類似度をはかる
    gram = np.dot(normA.T, normA)
    gram = gram - np.diag(np.diag(gram))

    # 閾値を設定
    t1 = 4
    t2 = 0.9    
    err = sum(normY - np.dot(normA, X))
    
    for i in range(A.shape[1]):
        if max(gram[i, :] > t2) or (sum(X[i, :] != 0) <= t1):
            idx = np.argmax(err)
            err[idx] = 0
            normA[:, i] = normY[:, idx]
            gram = np.dot(normA.T, normA)
            gram = gram - np.diag(np.diag(gram))
            
    return normA

def main():
    """
    K-SVD アルゴリズムの性能を評価する
    """

    # ランダムな辞書 (30 x 60) を作成
    A0 = np.random.randn(30, 60)
    A0, _ = collum_normalizarion(A0)

    # 上記辞書から信号事例を 4000 個作成
    Y = np.zeros((30, 4000))
    k0 = 4
    sigma = 0.1

    for i in range(4000):
        Y[:, i] = np.dot(A0[:, np.random.permutation(60)[:k0]], np.random.randn(4)) + np.random.randn(30) * sigma

    # K-SVD により辞書を学習する
    iter_num = 50
    A, log = KSVD(Y, A0.shape[1], k0, sigma, iter_num, A0=A0)

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
