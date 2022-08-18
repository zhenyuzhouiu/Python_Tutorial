import numpy as np
import open3d.cuda.pybind.visualization
from open3d import *

def main():
    cloud = open3d.io.read_point_cloud('./Scaniverse 2022-08-18 130212.ply')
    open3d.visualization.draw_geometries([cloud])

if __name__ == '__main__':
    main()