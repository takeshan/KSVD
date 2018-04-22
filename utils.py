# -*- coding:utf-8 -*-

import numpy as np

def collum_normalizarion(A):
    """
    与えられた行列を列で正規化する。その際の正規化に使用した対角行列も return する

    A: 列正規化したい行列
    """

    diag = np.diag(np.dot(A.T, A))
    W = np.diag(np.ones(diag.shape) / np.sqrt(diag))
    normA = np.dot(A, W)

    return normA, W

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

def l2_err(A, _A):
    """
    行列間の L2 誤差を求める
    """
    return np.dot(A - _A, A - _A) / np.dot(A, A)

def extract_patch(img, patch_size):
    """
    入力された画像から patch_size の大きさのパッチを取り出す

    img: パッチを抽出したい画像
    patch_size: 抽出するパッチのサイズ(パッチは正方形を仮定)
    """
    
    h, w = img.shape[:2]
    patches = []
    for i in range(h - patch_size + 1):
        for j in range(w - patch_size + 1):
            tmp = img[i:i+patch_size, j:j+patch_size]
            patches.append(tmp)

    return np.array(patches)

