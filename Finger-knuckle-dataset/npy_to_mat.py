import os

import numpy as np
import scipy.io

path = "/media/zhenyuzhou/Data/Project/Finger-Knuckle-Contactless/deep-learning/codekevin/fknet/test/assistant-fk/performance-compare/right-index/"

file = os.listdir(path)

for i in file:
    if i.split('.')[-1] == 'npy':
        npy_file = os.path.join(path, i)
        data = np.load(npy_file, allow_pickle=True)[()]
        scipy.io.savemat(npy_file[:npy_file.find('.npy')] + ".mat", {'D_genuine':data['g_scores'], 'D_imposter':data['i_scores']})
