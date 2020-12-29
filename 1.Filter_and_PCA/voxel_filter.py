'''
2020-04-16
孙正茂
实现voxel滤波，并加载数据集中的文件进行验证
'''

import open3d as o3d 
import os
import numpy as np
from pyntcloud import PyntCloud
from tqdm import tqdm
import time

def voxel_filter_approximate(point_cloud, leaf_size, r=10):
    '''
    功能：对点云进行voxel滤波
    输入：
        point_cloud：输入点云
        leaf_size: voxel尺寸
    '''
    points = np.array(point_cloud)
    filtered_points = []

    # 1.Compute min and max
    xyz_min = np.min(point_cloud)
    xyz_max = np.max(point_cloud)
    Dimension = np.ceil((xyz_max - xyz_min) / r)
    hashTable = [[] for i in range(int(leaf_size))]
    
    for index, point in tqdm(enumerate(points)):
        h_xyz = np.floor((point - xyz_min)/r)
        h = h_xyz[0] + h_xyz[1]*Dimension[0] + h_xyz[2]*Dimension[0]*Dimension[1]

        # 2. 计算hash_table
        hashValue = h % leaf_size
        i_container = hashTable[int(hashValue)]
        if len(i_container) == 0:
            i_container.append(index)
        else:
            the_last_index = i_container[-1]
            the_last_point = points[the_last_index]
            h_xyz_last_one = np.floor((the_last_point - xyz_min) / r)
            h_last_one = h_xyz_last_one[0]+h_xyz_last_one[1]*Dimension[0]+h_xyz_last_one[2]*Dimension[0]*Dimension[1]
            if np.sum(h_xyz==h_xyz_last_one)==3:
                i_container.append(index)
            else:
                # 从里面随便输出一个,然后把该容器清空,来放我的新的点
                filtered_points.append(the_last_point)
                hashTable[int(hashValue)] = [index]
    
    # 3.平均采样
    select_points = [points[hashTable[line][-1]] for line in range(int(leaf_size)) if len(hashTable[line])>0]
    filtered_points.extend(select_points)        
    print("近似滤波后会剩下{}个点".format(len(filtered_points)))

    filtered_points = np.array(filtered_points, dtype=np.float32)
    return filtered_points

def voxel_filter_new(data, leaf_size=0.02, method="mean", least_points_threshold=1):
    '''
    '''
    data = np.array(data)
    [N,d] = data.shape
    # record = np.arange(N)
    result = np.array([])

    # 1. compute the max,min,dimension,size
    xyz_min = np.min(data, axis=0)
    xyz_max = np.max(data, axis=0)
    Dimension = np.ceil((xyz_max - xyz_min) / leaf_size)
    container_size = Dimension[0]*Dimension[1]*Dimension[2]

    # 2. compute the hash_index
    h_xyz = np.floor((data - xyz_min.reshape(1,-1)) / leaf_size)
    h_index = (h_xyz[:,0] + h_xyz[:,1]*Dimension[0] + h_xyz[:,2]*Dimension[0]*Dimension[1]).astype("int")

    # 3. compute each voxel point
    for i in range(int(np.min(h_index)),int(np.max(h_index))):
        data_i = data[h_index==i]
        if data_i.shape[0] < least_points_threshold:
            continue
        # voxel_point = np.mean(data_i, axis=0)
        # if method == "random":
        #     voxel_point = data_i[0]
        # result.append(voxel_point)
        # record[h_index==i] = 1
        result = np.r_[result, data_i] if result.size else data_i
    
    return np.array(result)


def voxel_filter(point_cloud, leaf_size=10, selected_average=True):
    points = np.array(point_cloud)
    filtered_points = []

    # 1.Compute min and max
    xyz_min = np.min(point_cloud)
    xyz_max = np.max(point_cloud)
    Dimension = np.ceil((xyz_max - xyz_min) / leaf_size)
    container_size = Dimension[0]*Dimension[1]*Dimension[2]

    h_index = {}
    for i, point in tqdm(enumerate(points)):
        h_xyz = np.floor((point-xyz_min)/leaf_size)
        h = h_xyz[0] + h_xyz[1]*Dimension[0] + h_xyz[2]*Dimension[0]*Dimension[1]
        h_index[i] = int(h)
    
    # h_xyz = np.floor((points - xyz_min) / leaf_size)
    # h_index = int(h_xyz[:,0] + h_xyz[:,1]*Dimension[0] + h_xyz[:,2]*Dimension[0]*Dimension[1])
    # hash_max = np.max(h_index)
    
    h_index = sorted(h_index.items(), key = lambda i : i[1])

    filterPoint = [points[h_index[0][0]]]
    for it, index_hValue in enumerate(h_index):
        if it == 0:
            continue
        else:
            if index_hValue[1] == h_index[it-1][1]: #如果等于上一个,那就加起来吧
                filterPoint.append(points[index_hValue[0]])
            else:
                if selected_average:
                    selected_one = np.mean(filterPoint) # 如果不等于上一个了,那就先把之前累加的挑出来一个点进行输出
                else:
                    selected_one = filterPoint[0]
                filtered_points.append(selected_one)
                filterPoint = [points[index_hValue[0]]]    

    filtered_points = np.array(filtered_points, dtype=np.float64)
    return filtered_points


def main():
    # 从ModelNet数据集文件夹中自动索引路径，加载点云
    cat_index = 0 # 物体编号，范围是0-39，即对应数据集中40个物体
    root_dir = '/media/sunzhengmao/SZM/PointCloud/ModelNet40/ModelNet40_ply' # 数据集路径
    cat = os.listdir(root_dir)
    filename = os.path.join(root_dir, cat[cat_index],'train', cat[cat_index]+'_0001.ply') # 默认使用第一个点云
    point_cloud_pynt = PyntCloud.from_file(filename)

    # 加载自己的点云文件
    # file_name = "/Users/renqian/Downloads/program/cloud_data/11.ply"
    # point_cloud_pynt = PyntCloud.from_file(filename)

    # 转成open3d能识别的格式
    point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
    o3d.visualization.draw_geometries([point_cloud_o3d]) # 显示原始点云

    # 调用voxel滤波函数，实现滤波
    start = time.time()
    filtered_cloud = voxel_filter(point_cloud_pynt.points, 10, False)
    end = time.time()
    print("random voxel random cost: {}s, the number of reserved points is {}.".format(end-start, len(filtered_cloud)))
    
    start = time.time()
    filtered_cloud = voxel_filter_new(point_cloud_pynt.points, 10, True)
    end = time.time()
    print("average voxel random cost: {}s, the number of reserved points is {}.".format(end-start, len(filtered_cloud)))
    
    start = time.time()
    filtered_cloud = voxel_filter_approximate(point_cloud_pynt.points, 100, 50)
    end = time.time()
    print("approximate voxel random cost: {}s, the number of reserved points is {}.".format(end-start, len(filtered_cloud)))
    
    
    point_cloud_o3d.points = o3d.utility.Vector3dVector(filtered_cloud)
    o3d.visualization.draw_geometries([point_cloud_o3d])

if __name__ == '__main__':
    main()