import numpy as np
import scipy.io

path = ["/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/test/gui-cross-hd/rfn-wrs-crosshd-protocol.npy",
        "/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/test/gui-cross-hd/rfn-ws-crosshd-protocol.npy",
        "/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/test/fkv3/efficientnet-wrs-protocol.npy",
        "/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/test/fkv3/efficientnet-ws-protocol.npy",
        "/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/test/fkv3/rfn-oldwrs2newwrs-protocol.npy",
        "/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/test/fkv3/rfn-oldwrs-protocol.npy",
        "/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/test/fkv3/rfn-ws-protocol.npy"]


for i in range(2):
    data = np.load(path[i], allow_pickle=True)[()]
    scipy.io.savemat(path[i][:path[i].find('.npy')] + ".mat", {'D_genuine':data['g_scores'], 'D_imposter':data['i_scores']})
