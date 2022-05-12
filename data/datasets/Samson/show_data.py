import scipy.io as scio
import os
import os.path as osp
import matplotlib.pylab as plt
from PIL import Image


path_mat = './Data_Matlab/samson_1.mat'
data = scio.loadmat(path_mat)["V"]

path_lable = './GroundTruth/end3.mat'
data_label = scio.loadmat(path_lable)['A']

plt.imshow(data[0].reshape(95,95))
plt.show()
plt.imshow(data_label[2].reshape(95,95))
plt.show()
pass