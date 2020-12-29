'''
2020-04-18
孙正茂
实现PCA分析和法向量计算，并加载数据集中的文件进行验证
'''

import open3d as o3d 
import os
import numpy as np
from pyntcloud import PyntCloud
from tqdm import tqdm
import time

def PCA(data, correlation=False, sort=True):
    '''
    功能：计算PCA的函数
        这里只是让计算一个整体的主方向,不是每个点的主方向,所以直接套路就好
    输入：
        data：点云，NX3的矩阵
        correlation：区分np的cov和corrcoef，不输入时默认为False
        sort: 特征值排序，排序是为了其他功能方便使用，不输入时默认为True
    输出：
        eigenvalues：特征值
        eigenvectors：特征向量
    '''
    N, C = data.shape

    # [1] Normalized by the center
    xyz_center = np.mean(data, 0)
    xyz = data - xyz_center
    xyz_np = np.array(xyz)
    
    # [2] compute SVD
    # if correlation:
    # H = np.cov(data.transpose())
    # H = np.corrcoef(data.transpose())
    # else:
    #     H = np.corrcoef(xyz_np)
    H = np.matmul(xyz_np.transpose(), xyz_np)
    eigenvalues, eigenvectors = np.linalg.eig(H)

    if sort:
        sort = eigenvalues.argsort()[::-1] # 默认是从小到大排序,-1变成从大到小
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:, sort]

    return eigenvalues, eigenvectors


def main(path):
    # 指定点云路径
    cat_index = 0 # 物体编号，范围是0-39，即对应数据集中40个物体
    root_dir = '/media/sunzhengmao/SZM/PointCloud/ModelNet40/ModelNet40_ply' # 数据集路径
    cat = os.listdir(root_dir)
    filename = os.path.join(root_dir, cat[cat_index],'train', cat[cat_index]+'_0001.ply') # 默认使用第一个点云

    # 加载原始点云
    point_cloud_pynt = PyntCloud.from_file(filename)
    point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
    # o3d.visualization.draw_geometries([point_cloud_o3d]) # 显示原始点云

    # 从点云中获取点，只对点进行处理
    points = point_cloud_pynt.points
    N, C = points.shape
    print('total points number is:', points.shape[0])

    # 用PCA分析点云主方向
    w, v = PCA(points)
    point_cloud_vector = v[:, 0].reshape(1,3)*1000#点云主方向对应的向量
    print('the main orientation of this pointcloud is: ', point_cloud_vector)
    # 此处只显示了点云，还没有显示PCA
    # 无法对单独一个向量进行可视化,要不转成点,要不转成某一个点上的法向量
    # normals = np.array(point_cloud_vector, dtype=np.float64)
    # point_cloud_o3d.points = o3d.utility.Vector3dVector(normals)
    # o3d.visualization.draw_geometries([point_cloud_o3d],'open3d')
    
    # 循环计算每个点的法向量
    pcd_tree = o3d.geometry.KDTreeFlann(point_cloud_o3d)
    normals = []
    knn = 10
    for i in tqdm(range(N)):
        xyz = np.array(points)
        neibor = pcd_tree.search_knn_vector_3d(xyz[i,:], knn)
        w, v = PCA(xyz[neibor[1],:])
        normals.append(v[:,2])

    normals = np.array(normals, dtype=np.float64)
    # 此处把法向量存放在了normals中
    point_cloud_o3d.normals = o3d.utility.Vector3dVector(normals)
    o3d.visualization.draw_geometries([point_cloud_o3d])


if __name__ == '__main__':
    path = "/media/sunzhengmao/SZM/PointCloud/ModelNet40/ModelNet40_ply/airplane/train/airplane_0001.ply"
    main(path)