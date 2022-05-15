import numpy as np
import scipy.io

path = ["/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/codekevin/fknet/test/3d1s3d/3d1s3d_RFN-128-shiftedloss-lr0.001-protocol.npy",
        "/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/codekevin/fknet/test/3d1s3d/3d1s3d_RFN-128-wholeimagerotationandtranslation-lr0.001-protocol.npy",
        "/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/codekevin/fknet/test/3d1s-2d/3d1s-rfn-ws-protocol.npy",
        "/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/codekevin/fknet/test/3d1s-2d/3d1s-rfn-wrs-protocol.npy",
        "/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/codekevin/fknet/test/fkv3/fkv3-rfn-ws-protocol.npy"]


for i in range(2):
    data = np.load(path[i], allow_pickle=True)[()]
    scipy.io.savemat(path[i][:path[i].find('.npy')] + ".mat", {'D_genuine':data['g_scores'], 'D_imposter':data['i_scores']})
