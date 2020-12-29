import numpy as np 
import open3d as o3d 
import os
import sys
import utils.read_txt

import_path = "/home/sunzhengmao/PointNet/深蓝三维点云处理/Utils"
sys.path.append(import_path)



def detect_Harris3D_I_feature(data:np.ndarray):
    pass

def detect_Harris6D_feature(data:np.ndarray, parameter_list):
    pass


def detect_Harris3D_n_feature(data:np.ndarray, parameter_list, threshold):
    '''
    功能：提取Harris 3D角点，通过normal来代替Intensity
    '''
    [N,d] = data.shape
    kdtree = o3d.geometry.KDTreeFlann(data.T)

    output_index=[]
    for i in range(N):
        query = data[i,:]
        neibor = kdtree.search_radius_vector_3d(query, radius=0.2)
        points_i = data[neibor[1], :]

        # get local surface normal
        points_i_norm_nx3 = (points_i-query.reshape(1,3))/neibor[0]
        A=np.linalg.multi_dot([points_i_norm_nx3.T, points_i_norm_nx3])
        eigenValue, eigenVector = np.linalg.eig(A)
        small_index = np.argmin(eigenValue)
        normal = eigenVector[:,small_index]

        # compute response value R
        M = np.linalg.multi_dot([normal.reshape(3,1), normal.reshape(1,3)])
        eigenValue_M,_ = np.linalg.eig(M)
        R = np.min(eigenValue_M)
    
        if R > threshold:
            output_index.append(i)
            
    return output_index



def main():
    root_path = "/media/sunzhengmao/SZM/PointCloud/ModelNet40/modelnet40_normal_resampled/"
    root_dir = os.listdir(root_path)
    for index, name_i in enumerate(root_dir[1:]):
        print("======[  第{}个文件  ]======".format(index))
        data_path = root_path + name_i + '/' + name_i + '_0001.txt'
        data_points = read_txt(data_path)   

        # random select a point, centify
        query = data_points[0,:].reshape(1,3)
        vector = data_points-query
        distance = np.sqrt(np.sum(vector * vector, axis=1))
        distance.sort()
        radius = distance[30]

        [N,d] = data_points.shape
        iss_feature_index = detect_Harris3D_n_feature(data_points,radius=radius,radius_NMS=0.5)
        if len(iss_feature_index)==0:
            print("strong constrain")
            continue

if __name__ == "__main__":
    main()