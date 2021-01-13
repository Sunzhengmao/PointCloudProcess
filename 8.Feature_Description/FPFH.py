import numpy as np 
import open3d as o3d 
import math
import os
import sys

import_path = "/home/sunzhengmao/PointNet/深蓝三维点云处理/Utils/"
sys.path.append(import_path)
from utils import show_point_color,ShowPoint


'''
功能：   将特征点读入，并保存为float格式
        如果用double的话，可能由于尾数太多造成负数，因此引出复数
        可以在最后的时候用astype统一转换格式，不用每次读进来就开始转换
'''
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

'''
功能：计算data中每个点到query的距离
'''
def compute_distance(data, query):
    data = np.array(data)
    [N,d] = data.shape
    vector_query_data = data-query.reshape(1,3)
    distance = np.sqrt(np.sum(vector_query_data * vector_query_data, axis=1))
    return distance

'''
功能：计算query点处的法向量，利用其邻域点进行计算，
     同时保证法向量的指向为点所在较多的方向
'''
def compute_normal(query, neibor_data):
    neibor_data = np.array(neibor_data)
    [N,d] = neibor_data.shape

    # get normal by eigenDecompotion
    vector_query_data = (neibor_data - query.reshape(1,3))
    A = np.linalg.multi_dot([vector_query_data.T, vector_query_data])
    eigenValue, eigenVector = np.linalg.eig(A)
    smallest_index = np.argmin(eigenValue) 
    surface_normal = eigenVector[:,smallest_index]

    # vertify if we need to add minus
    # 应该是正的比较多才对
    signal = np.linalg.multi_dot([vector_query_data, surface_normal.reshape(3,1)])
    positive = np.sum(signal>0)
    negtive = np.sum(signal<0)
    if negtive > positive:
        surface_normal *= -1

    return surface_normal

'''
功能：计算PFH特征
'''
def compute_alpha_phi_thita(query1, query2, n1, n2):
    p2_p1 = query2 - query1
    u = n1
    v = np.cross(p2_p1, u)
    w = np.cross(u, v)

    # normalize
    u /= np.linalg.norm(u)
    v /= np.linalg.norm(v)
    w /= np.linalg.norm(w)

    # get result
    alpha = np.dot(v, n2)
    phi = np.dot(u, p2_p1)/np.linalg.norm(p2_p1)
    thita = math.atan2(np.dot(w,n2), np.dot(u,n2))
    while thita < -1.57:
        thita += 3.14
    while  thita > 1.57:
        thita -= 3.14

    return alpha, phi, thita


'''
功能：计算SPFH特征点，即计算中心点到邻域点的
'''
def compute_SPFH(query, data, radius, kdtree, dict_point_normal:dict, B=5):
    data = np.array(data)
    [N, d] = data.shape
    description = np.zeros(3*B)

    # 1. get query's normal
    query_neibor = kdtree.search_radius_vector_3d(query, radius)
    if tuple(query) not in dict_point_normal.keys():
        query_normal = compute_normal(query=query, neibor_data=data[query_neibor[1]])
        dict_point_normal[tuple(query)] = tuple(query_normal)
    else:
        query_normal = np.array(dict_point_normal[tuple(query)])

    # 2. get neibors' normal
    for i in query_neibor[1][1:]:
        neibor_i = data[i]
        # 2-1 if we had not computed the normal before
        if tuple(neibor_i) not in dict_point_normal.keys():
            neibor_i_neibor = kdtree.search_radius_vector_3d(neibor_i, radius)
            neibor_i_normal = compute_normal(query=neibor_i, neibor_data=data[neibor_i_neibor[1]])
            dict_point_normal[tuple(neibor_i)] = tuple(neibor_i_normal)
        else:
            neibor_i_normal = np.array(dict_point_normal[tuple(neibor_i)])

        # 3. conmputer the alpha(-1,1), phi(-1,1) and thita(-pi/2, pi/2)
        alpha, phi, thita = compute_alpha_phi_thita(query, neibor_i, query_normal, neibor_i_normal)

        description[ 0 + int(np.floor((alpha+1)/(2/B)))]           += 1
        description[ 5 + int(np.floor((phi+1)/(2/B)))]             += 1
        description[10 + int(np.floor((thita+1.5708)/(3.14/B)))]   += 1
    
    return description, query_neibor


'''
功能：对特征点query求取描述子
'''
def compute_FPFH_description(data, query, B=5):
    data = np.array(data)
    [N,d] = data.shape
    kdtree = o3d.geometry.KDTreeFlann(data.T)
    description = np.zeros(B*3)
    dict_point_normal = {} # record the normal

    # 0. dynamic get the appropriate radius
    distance_sort = np.sort(compute_distance(data,query))
    radius = distance_sort[30]

    # 1. compute SPFH of query point
    description_query, neibor_query = compute_SPFH(query, data, radius, kdtree, dict_point_normal, B)
    description += description_query

    # 2. compute SPFH of neighbor points
    for i in neibor_query[1][1:]:
        neibor_i = data[i]
        description_neibor_i, _ = compute_SPFH(neibor_i, data, radius, kdtree, dict_point_normal, B)

        pq_pk = query - neibor_i
        distance = np.sqrt(np.sum(pq_pk*pq_pk))
        description += description_neibor_i / (neibor_query[0] * distance)
    
    return description



def main():
    data_path = "/media/sunzhengmao/SZM/PointCloud/ModelNet40/modelnet40_normal_resampled/"
    list_dir = os.listdir(data_path)
    for model in list_dir[0:]:     

        if "chair" in model:
            model_path = data_path+model+'/'+model+'_0001.txt'
            model_data = read_txt(model_path)
            [N,d] = model_data.shape
            cluster_index = np.zeros(N)
            cluster_index[0] = 1
            cluster_index[100] = 2
            show_point_color(model_data, cluster_index)

            # get random rotation
            rotate_vector = np.array([1,2,3])
            R = compute_R(rotate_vector)

            model_data_rotate = np.linalg.multi_dot([model_data, R])
            show_point_color(model_data_rotate, cluster_index)

            point1 = model_data[0]
            description1 = compute_FPFH_description(model_data, point1)
            print(description1)
            
            point2 = model_data_rotate[0]
            description2 = compute_FPFH_description(model_data_rotate, point2)
            print(description2)

            point3 = model_data[100]
            description3 = compute_FPFH_description(model_data, point3)
            print(description3)

            delta_12 = description1 - description2
            delta_13 = description1 - description3
            delta_23 = description2 - description3
            similar_dist12 = np.sqrt(np.sum(delta_12*delta_12))
            different_dist13 = np.sqrt(np.sum(delta_13*delta_13))
            different_dist23 = np.sqrt(np.sum(delta_23*delta_23))
            print("similar_dist:",similar_dist12)
            print("different_dist:",different_dist13)
            print("different_dist:",different_dist23)

            # z_sort = np.sort(model_data[:,1])
            # z_value = z_sort[2]
            # cluster_index = np.zeros(N)
            # index = np.arange(N)[model_data[:,1]<z_value]
            # cluster_index[501] = 1
            # cluster_index[index] = 1
            # # show_point_color(model_data, cluster_index)

            i=1



if __name__ == "__main__":
    main()