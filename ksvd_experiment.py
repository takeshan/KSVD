# -*- coding:utf-8 -*-
# ./figure/babara.png を使用して辞書学習を評価する
import numpy as np
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from skimage import io

from utils import *
from ksvd import KSVD

def show_dictionary(A, name=None):
    """ 辞書を表示 """
    n = int(np.sqrt(A.shape[0]))
    m = int(np.sqrt(A.shape[1]))
    A_show = A.reshape((n, n, m, m))
    fig, ax = plt.subplots(m, m, figsize=(4, 4))
    for row in range(m):
        for col in range(m):
            ax[row, col].imshow(A_show[:, :, col, row], cmap='gray', interpolation='Nearest')
            ax[row, col].axis('off')
    if name is not None:
        plt.savefig(name, dpi=220)
                

                
def main():
    # 画像からパッチを抽出
    img = io.imread('./figure/barbara.png', as_grey=True)
    patch_size = 8
    patches = extract_patch(img, patch_size)

    # 2 次元分離可能 DCT 辞書を作成
    A_1D = np.zeros((8, 11))
    for k in range(11):
        for i in range(8):
            A_1D[i, k] = np.cos(i * k * np.pi / 11)

        if k != 0:
            A_1D[:, k] -= A_1D[:, k].mean()

    A_2D = np.kron(A_1D, A_1D)

    # 抽出したパッチで学習する
    idx = np.random.randint(0, patches.shape[0], int(patches.shape[0] / 10))
    Y = patches[idx]
    Y = Y.reshape(len(idx), 64).swapaxes(0, 1)

    sig = 0
    k0 = 4
    iter_num = 50
    
    babara_dic, babara_log = KSVD(Y, A_2D.shape[1], k0, sig, iter_num, initial_dictonary=A_2D.copy())
    show_dictionary(babara_dic, './figure/ksvd_babara_dic.png')

    plt.plot(babara_log, label='K-SVD')
    plt.ylabel('mean error')
    plt.xlabel('# of iteration')
    plt.legend(loc='best')
    plt.grid()
    plt.show()
    
if __name__ == '__main__':
    main()
