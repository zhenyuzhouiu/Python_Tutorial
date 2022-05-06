import numpy as np
import scipy.io

path = ["/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/codekevin/fknet/test/RFN-WRSprotocol3.npy",
        "/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/codekevin/fknet/test/RFN-WSprotocol3.npy"]


for i in range(2):
    data = np.load(path[i], allow_pickle=True)[()]
    scipy.io.savemat(path[i][:path[i].find('.npy')] + "_cmc.mat", {'D_genuine':data['g_scores'], 'D_imposter':data['i_scores']})
