# -*- coding:utf-8 -*-
import numpy as np
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.io import imread

from ksvd_qiita import DctionaryLearning

im = imread('./figure/babara.png', as_grey=True).astype(np.float)

patch_size = 8
patches = []
for row in range(im.shape[0] - patch_size + 1):
    for col in range(im.shape[1] - patch_size + 1):
        patches.append(im[row:row + patch_size, col:col + patch_size])
patches = np.array(patches)

n = 8
sample = patches[np.random.permutation(patches.shape[0])[:n ** 2]]
print(sample.shape)
sample = sample.reshape((n, n, patch_size, patch_size))
patch_work = np.swapaxes(sample, 1, 2).reshape((n * patch_size, n * patch_size))

fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].imshow(im, cmap='gray', interpolation='Nearest')
ax[0].axis('off')
ax[0].set_title('Barbara')
ax[1].imshow(patch_work, cmap='gray', interpolation='Nearest')
ax[1].axis('off')
ax[1].set_title('a part of patches')
plt.tight_layout()
plt.savefig('barbara_patches.png', dpi=220)

A_1D = np.zeros((8, 11))
for k in np.arange(11):
    for i in np.arange(8):
        A_1D[i, k] = np.cos(i * k * np.pi / 11.)
    if k != 0:
        A_1D[:, k] -= A_1D[:, k].mean()
            
A_2D = np.kron(A_1D, A_1D)
            
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
        
show_dictionary(A_2D, name='dct_dictionary.png')

Y = patches[::10].reshape((-1, 64)).swapaxes(0, 1)
sig = 0
k0 = 4

dl = DctionaryLearning()
print('start to learn')
A_KSVD_barbara, log_KSVD_barbara = dl.KSVD(Y, sig, A_2D.shape[1], k0, n_iter=50, initial_dictionary=A_2D.copy())

show_dictionary(A_KSVD_barbara, name='ksvd_dictionary_barbara.png')

plt.plot(log_KSVD_barbara, label='K-SVD')
plt.ylabel('mean error')
plt.xlabel('# of iteration')
plt.legend(loc='best')
plt.grid()
plt.savefig('Barbara_K-SVD.png', dpi=220)
