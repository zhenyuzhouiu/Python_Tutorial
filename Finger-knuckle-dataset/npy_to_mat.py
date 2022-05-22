import numpy as np
import scipy.io

path = ["/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/codekevin/fknet/test/two-fkv3/fknet.npy",
        "/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/codekevin/fknet/test/two-fkv3/WRS-0.001-protocol3.npy",
        "/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/codekevin/fknet/test/two-fkv3/WS-0.01-protocol3.npy",
        "/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/codekevin/fknet/test/two-fkv3/WS-0.001-protocol3.npy",
        "/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/codekevin/fknet/test/fkv3/fkv3-rfn-ws-protocol.npy"]


for i in range(1):
    data = np.load(path[i], allow_pickle=True)[()]
    scipy.io.savemat(path[i][:path[i].find('.npy')] + ".mat", {'D_genuine':data['g_scores'], 'D_imposter':data['i_scores']})
