import numpy as np 
import open3d as o3d 
import os
import sys
import math

import_path = "/home/sunzhengmao/PointNet/深蓝三维点云处理/Utils"
sys.path.append(import_path)
from utils import show_point_color

def read_txt(data_path):
    result = []
    with open(data_path, "r") as fread:
        point = fread.readline().split(',')
        while len(point)==6:
            result.append([point[0], point[1], point[2]])
            point = fread.readline().split(',')
    return np.array(result).astype('float32')


'''
功能：通过旋转向量计算旋转矩阵
'''
def compute_R(rotate_vector):
    rotate_vector = np.array(rotate_vector)
    thita = np.linalg.norm(rotate_vector)
    rotate_vector = (rotate_vector/thita).reshape(3,1)
    rx = rotate_vector[0]
    ry = rotate_vector[1]
    rz = rotate_vector[2]

    R_part1 = math.cos(thita) * np.eye(3)
    R_part2 = (1-math.cos(thita)) * np.linalg.multi_dot([rotate_vector, rotate_vector.T])
    R_part3 = math.sin(thita) * (np.array([0,-rz,ry,rz,0,-rx,-ry,rx,0]).reshape(3,3))
    R = R_part1 + R_part2 + R_part3
    return np.array(R).astype('float32')


def detect_Harris3D_I_feature(data:np.ndarray):
    pass

def detect_Harris6D_feature(data:np.ndarray, parameter_list):
    pass

'''
功能：提取Harris 3D角点，通过normal来代替Intensity
'''
def detect_Harris3D_n_feature(data:np.ndarray, parameter_list, threshold):
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


'''
功能：提取ISS特征点
radius：        对每个点进行半径搜索时的半径
radius_NMS：    对选出来的候选特征点进行NMS时，用了半径搜索的办法，该值即为所需的半径
gamma1 ：       第一个特征值与第二个特征值比值的阈值
gamma2：        第二个特征值与第三个特征值比值的阈值
lambda_th：     对第三个特征值判断时的阈值

'''
def detect_ISS_feature(data:np.ndarray, radius=0.2, radius_NMS=0.5, gamma1=1.5, gamma2=1.5, rate_feature_uper=0.1, rate_feature_lower=0.05):
    [N,d] = data.shape
    kdtree = o3d.geometry.KDTreeFlann(data.T)

    lambda3 = []
    eigen2_3 = []
    eigen1_2 = []
    for index in range(N):
        query = data[index,:]
        neibor = kdtree.search_radius_vector_3d(query, radius=radius)

        # 0. compute weight
        distance = np.array(neibor[2])
        distance[distance==0] = 1.0
        weight_nx1 = 1.0/distance
        weight_nx3 = np.tile(weight_nx1.reshape(-1,1), (1,3))
        weight_sum = np.sum(weight_nx1)

        neibor_points = data[neibor[1],:]
        neibor_points_nx3 = neibor_points - query.reshape(1,3)
        neibor_points_weight_nx3 = weight_nx3 * neibor_points_nx3
        # neibor_points_nx3 /= np.max(neibor_points_nx3)

        # 1. compute weighted conv matrix and eigenvalue decomposition
        A = np.linalg.multi_dot([neibor_points_nx3.T, neibor_points_weight_nx3])
        A = A / weight_sum
        eigenValue, eigenVector = np.linalg.eig(A)
        big_small_sort = np.argsort(-eigenValue)
        eigenValue = eigenValue[big_small_sort]

        # 2. start discard some points
        # 2-1 λ1与λ2相差较大，λ2与λ3相差较大
        eigen2_3.append(eigenValue[1]/eigenValue[2])
        eigen1_2.append(eigenValue[0]/eigenValue[1])
        # if eigenValue[0]/eigenValue[1] < gamma1 or eigenValue[1]/eigenValue[2] < gamma2:
        #     lambda3.append(0.0)
        #     continue
        # 2-2 feature should be the point whose lambda3 is small 
        lambda3.append(eigenValue[2])
    
    # 3-1. dynamic process eigen 比值
    lambda3 = np.array(lambda3)
    eigen2_3 = np.array(eigen2_3)
    eigen1_2 = np.array(eigen1_2)

    sort_23 = np.argsort(-eigen2_3)
    sort_eigen23 = eigen2_3[sort_23] # 从大到小将eigen值进行排序
    sort_12 = np.argsort(-eigen1_2)
    sort_eigen12 = eigen1_2[sort_12]

    # 动态调整lambda值，直到合适的数量在lower和uper之间
    eigen_riato = 0.5
    for i in range(10):
        lambda3_tmp = lambda3.copy()
        threshold23 = sort_eigen23[int(eigen_riato*N)]
        threshold12 = sort_eigen12[int(eigen_riato*N)]
        lambda3_tmp[eigen2_3 < threshold23] = 0.0
        lambda3_tmp[eigen1_2 < threshold12] = 0.0

        num_lambda3 = np.sum(lambda3_tmp > 0)
        if num_lambda3 > N*rate_feature_uper:
            eigen_riato -= 0.1
        elif num_lambda3 < N*rate_feature_lower:
            eigen_riato += 0.02
        else:
            break

    # 3-2. dynamic split the points whose lambda3 is bigger
    lambda3 = lambda3_tmp
    big_small_sort = np.argsort(-lambda3)
    print("there are {} points satisfied".format(num_lambda3))
    lambda3 = lambda3[big_small_sort[:num_lambda3]]
    candidate_index = np.arange(N)[big_small_sort[:num_lambda3]]


    # 4. process NMS
    candidate = data[candidate_index,:]
    N_index = list(candidate_index)
    kdtree_can = o3d.geometry.KDTreeFlann(candidate.T)

    output_index = []
    for index_can in N_index:
        query = data[index_can,:]
        neibor = kdtree_can.search_radius_vector_3d(query, radius=radius_NMS)

        tmp = [x for x in neibor[1] if candidate_index[x] in N_index]#从neibor[1]中排除N_index已经没了的点
        if len(tmp) == 0:
            continue
        # lambda3_i = lambda3[tmp]
        # small_index = np.argmax(lambda3_i)

        neibor_index = candidate_index[neibor[1]]
        N_index = [x for x in N_index if x not in neibor_index] # 防止下次再遍历到这个邻域中的点
        output_index.append(candidate_index[tmp[0]])

    return np.array(output_index)     


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
        iss_feature_index = detect_ISS_feature(data_points,radius=radius,radius_NMS=0.1)

        # after rotation, detect the repeatively
        rotate_vector = np.array([1,2,3])
        R = compute_R(rotate_vector)
        data_rotate = np.linalg.multi_dot([data_points, R])
        iss_feature_index_rotate = detect_ISS_feature(data_rotate, radius=radius, radius_NMS=0.1)

        if np.all(iss_feature_index==iss_feature_index_rotate):
            print("they have invariently")

        if len(iss_feature_index)==0:
            print("strong constrain")
            continue

        cluster_index = np.zeros(N)
        cluster_index[iss_feature_index] = 1
        show_point_color(data_points, cluster_index)  

    i=1

if __name__ == "__main__":
    main()
    a=np.array([1,2,3,4,5])
    b = np.argsort(a)

    i=1