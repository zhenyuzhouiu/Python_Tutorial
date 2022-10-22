import numpy as np
from open3d import *
import os

def main(path):
    cloud = open3d.io.read_point_cloud(path)
    open3d.visualization.draw_geometries([cloud])

if __name__ == '__main__':
    source_path = "./ply/"
    files = os.listdir(source_path)
    for f in files:
        path = os.path.join(source_path, f)
        main(path)