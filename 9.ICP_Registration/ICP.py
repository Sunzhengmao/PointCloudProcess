import os
import math
import sys
path_import = os.path.dirname(__file__)
sys.path.append(path_import)

from FPFH import compute_FPFH_normal_description
from evaluate_rt import read_oxfeord_bin,read_reg_results,reg_result_row_to_array,visualize_pc_pair
import numpy as np
import time
import open3d as o3d
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import struct
import copy


from evaluate_rt import read_oxfeord_bin,read_reg_results,reg_result_row_to_array,visualize_pc_pair
from ISS import detect_ISS_feature
# from FPFH import compute_FPFH_description, compute_R, compute_FPFH_normal_description
from utils import ShowPoint, show_point_color


def get_quaternion_from_R(R):
    '''
    功能：将旋转矩阵转为四元数
    '''
    qw = np.sqrt(np.trace(R)+1)/2
    qx = (R[1,2] - R[2,1])/(4*qw)
    qy = (R[2,0] - R[0,2])/(4*qw)
    qz = (R[0,1] - R[1,0])/(4*qw)
    return np.array([qw, qx, qy, qz])

def get_T_from_Rt(R, t):
    '''
    功能：将R和t变成变换矩阵
    '''
    T = np.eye(4)
    T[:3, :3] = R
    T[:3,3] = t.reshape(3,)
    return T

def get_matched(feat_index1, descript1, feat_index2, descript2, methods='union', threshold=1.1):
    '''
    功能：将特征点的描述子descript1和descript2进行匹配，将匹配上的点的索引以tuple形式进行保存
        即(index1,index2)，得到对应点在总data中的index

    feat_index1:    特征点1在所有点中的index
    descript1:      1中这些特征点的描述子
    feat_index2:    特征点2在所有点中的index
    descript2:      2中这些特征点的描述子
    method：        选并集union还是选交集intersection
    threshold:      
    '''
    [N1, d1] = descript1.shape
    [N2, d2] = descript2.shape

    match_12 = []
    match_21 = []

    for idx1 in range(N1): # idx1是feat_index1和descript1的索引，是0到50的一部分，用的时候是feat_index1[idx1]
        vector = descript2 - descript1[idx1].reshape(1,-1)
        distance = np.sqrt(np.sum(vector * vector, axis=1))
        distance_sort = np.sort(distance)
        ratio_2_1 = distance_sort[1]/distance_sort[0]
        if ratio_2_1 < threshold: # 如果第一小比第二小小5倍，就认为可靠
            continue
        idx2 = np.argmin(distance)
        match_12.append((feat_index1[idx1], feat_index2[idx2]))
    print("mated_12 has {} pairs".format(len(match_12)))
    
    for idx2 in range(N2):
        vector = descript1 - descript2[idx2].reshape(1,-1)
        distance = np.sqrt(np.sum(vector * vector, axis=1))
        distance_sort = np.sort(distance)
        ratio_2_1 = distance_sort[1]/distance_sort[0]
        if ratio_2_1 < threshold: # 如果第一小比第二小小5倍，就认为可靠
            continue
        idx1 = np.argmin(distance)
        match_21.append((feat_index1[idx1], feat_index2[idx2]))
    print("mated_21 has {} pairs.".format(len(match_21)))

    if methods=='union':# 取并集
        result = list(set(match_21).union(set(match_12)))
    if methods=='intersection': # 取交集
        result = list(set(match_12).intersection(set(match_21)))
    print("method is", methods)
    print("the last result has {} pairs.".format(len(result)))

    return np.array(result)


def data_association(source, target, data_s, data_t, kd, eps, thre, type='fpfh',method='union'):
    s = np.zeros([len(source), 33])
    t = np.zeros([len(target), 33])
    for i in tqdm(range(len(target))):
        t[i] = compute_FPFH_normal_description(target[i], eps=eps, data=data_t, kd=kd[1])
    for i in tqdm(range(len(source))):
        s[i] = compute_FPFH_normal_description(source[i], eps=eps, data=data_s, kd=kd[0])

    index_filtered_s = np.argwhere(s[:, 0] != -1).flatten()
    index_filtered_t = np.argwhere(t[:, 0] != -1).flatten()
    s = s[index_filtered_s]
    t = t[index_filtered_t]
    source = np.array(source)
    target = np.array(target)
    source = source[index_filtered_s]
    target = target[index_filtered_t]

    # 从source中找target的对应点
    s = s.reshape([-1, 1, s.shape[-1]])
    t = t.reshape([1, -1, s.shape[-1]])
    dis_1 = s - t
    dis_1 = np.sqrt(np.sum(dis_1 * dis_1, axis=-1))
    index_1 = np.argmin(dis_1, axis=1)  # min index in each row
    min_dis_1 = np.min(dis_1, axis=1)
    index_dis_1 = np.argwhere(min_dis_1 < thre).flatten()
    index_inline = index_1[index_dis_1]
    source_1 = source[index_dis_1]
    index_inline_1 = np.reshape(index_inline, [-1])


    # 从target中找source的对应点
    s = s.reshape([1, -1, s.shape[-1]])
    t = t.reshape([-1, 1, s.shape[-1]])
    dis_2 = s - t
    dis_2 = np.sqrt(np.sum(dis_2 * dis_2, axis=-1))
    index_2 = np.argmin(dis_2, axis=1)
    min_dis_2 = np.min(dis_2, axis=1)
    index_dis_2 = np.argwhere(min_dis_2 < thre).flatten()
    index_inline = index_2[index_dis_2]
    target_2 = target[index_dis_2]
    index_inline_2 = np.reshape(index_inline, [-1])

    # 直接将对应点连起来
    source = np.concatenate([source_1, source[index_inline_2]])
    index_inline = np.concatenate([index_inline_1, index_dis_2])
    # # 取交集
    # matched_st = dict(zip(source_1, index_inline_1))
    # matched_ts = dict(zip(target_2, index_inline_2))    
    # if methods=='union':# 取并集
    #     result = list(set(matched_st).union(set(matched_ts)))
    # if methods=='intersection': # 取交集
    #     result = list(set(matched_st).intersection(set(matched_ts)))
    # result = np.array(result)
    # source = result[:,0]
    # index_inline result[:,1]

    data1_mx6_matched = data_s[source]
    data2_nx6_matched = data_t[target[index_inline]]
    return data1_mx6_matched[:,:3], data2_nx6_matched[:,:3]


def compute_transform_PQ(data1, data2):
    '''
    功能：根据ICP算法，计算转换之间的R和t
    '''
    A_mxN = np.array(data1).T
    B_mxN = np.array(data2).T
    [N, d1] = data1.shape
    [N1, d2] = data2.shape
    assert N1 == N
    
    L = np.eye(N) - np.ones((N,N))/N
    A_prime = np.linalg.multi_dot([A_mxN, L])
    B_prime = np.linalg.multi_dot([B_mxN, L])

    BAT = np.linalg.multi_dot([B_prime, A_prime.T])
    U,sigma,VT = np.linalg.svd(BAT)

    R_21 = np.linalg.multi_dot([U,VT])
    # special reflection case
    # if np.linalg.det(R_21) < 0:
    #     VT[2,:] *= -1
    #     R_21 = np.dot(U, VT)

    B_RA = B_mxN - np.linalg.multi_dot([R_21, A_mxN])
    t_21 = np.linalg.multi_dot([B_RA, np.ones(N).reshape(N,1)])/N

    return R_21, t_21


def compute_Rt_ransac(data1_nx3, data2_nx3, iter=1000000, thre=4):
    '''
    功能：  通过match和两对点，用ransac方法，计算出最好的R和t
            在Data1和Data2已经是匹配好的点，所以在计算error时不用再找最近点了，直接对应行相减即可
    data1：第一个点云中的匹配点
    data2：第二个点云中的匹配点
    iter： ransac要计算的迭代次数
    thre： 满足两点距离小于thre时，被认为是一个inlier点
    '''    
    [N, d] = data1_nx3.shape
    R_set = []
    t_set = []
    error_set = np.zeros(iter)
    # error_set = []

    for i in tqdm(range(iter)):
        # 1. select 3 pair points randomly
        index = np.random.choice(N, 6, replace=False)
        R21_i, t21_i = compute_transform_PQ(data1_nx3[index], data2_nx3[index])
        if np.linalg.det(R21_i) < 0:
            R_set.append(-1)
            t_set.append(-1)
            continue
        
        # 2. compute inlier/outlier
        ERROR = data2_nx3.T - np.linalg.multi_dot([R21_i, data1_nx3.T]) - t21_i.reshape(3,1)
        ERROR = ERROR.T
        error = np.sum(ERROR**2, axis=1)
        num = np.sum(error < thre)
        error_set[i]=num 
        # error_set.append(num)
        R_set.append(R21_i)
        t_set.append(t21_i)        
        # inlier_ratio, distance = compute_inlier_ratio_error(data1_nx3, data2_nx3, R21_i, t21_i)
        # if distance < smallest_distance:
        #     best_inlier_ratio = inlier_ratio
        #     smallest_distance = distance
        #     best_R = R21_i
        #     best_t = t21_i

    error_set = np.array(error_set)
    best_index = np.argmax(error_set)
    best_R = R_set[best_index]
    best_t = t_set[best_index]
    
    return best_R, best_t

    
def compute_ICP_process(data_source, data_target):

    ####  [ 1. Initialization ]  ####
    # 1. read source data
    print("1. detect ISS feature...")
    kp_index_source = detect_ISS_feature(data_source[:,0:3], radius=1.2, radius_NMS=1.5, rate_feature_lower=0.01, rate_feature_uper=0.015)
    kp_index_target = detect_ISS_feature(data_target[:,0:3], radius=1.2, radius_NMS=1.5, rate_feature_lower=0.01, rate_feature_uper=0.015)

    # 2. 利用描述子将对应点匹配起来
    print("2. make data association...")
    kd_source = o3d.geometry.KDTreeFlann(data_source[:,:3].T)
    kd_target = o3d.geometry.KDTreeFlann(data_target[:,:3].T)
    data1_nx3_matched, data2_nx3_matched = data_association(kp_index_source, kp_index_target, data_source, data_target, kd=(kd_source, kd_target), thre=6, eps=1.5)

    # 3. 计算R和t的初始解
    print("3. compute initialized R and t...")
    # R, t = compute_Rt_ransac(data1_nx3_matched, data2_nx3_matched, iter=100000, thre=4)
    # T_end = get_T_from_Rt(R, t)

    R21, t21 = compute_Rt_ransac(data1_nx3_matched, data2_nx3_matched, iter=100000, thre=4)
    T21 = get_T_from_Rt(R21, t21)


    data1_mx3 = data_source[:,:3]
    data2_nx3 = data_target[:,:3]
    [M, d] = data1_mx3.shape
    [N, d] = data2_nx3.shape

    best_inlier = 0
    smallest_error = 1e6
    iter_num = 30
    ####  [ 2. ICP ]  ####
    print("4. start ICP...")
    for iter_i in tqdm(range(iter_num)):        

        # 2-0 first show the points
        temp2_3xm = np.linalg.multi_dot([R21, data1_mx3.T]) + t21.reshape(3,1)
        temp2_mx3 = temp2_3xm.T
        # visualize_pc_pair(data2_nx3.T, temp2_3xm)

        # 2-1 kdtree find the nearest point in data2
        index1 = []
        index2 = []
        for idx in range(M):
            query = temp2_mx3[idx]
            neighbor = kd_target.search_knn_vector_3d(query, knn=1)
            if np.sqrt(neighbor[2][0]) > 1:
                continue
            index1.append(idx)
            index2.append(neighbor[1][0])
        index1 = np.array(index1)
        index2 = np.array(index2)
        data1_mx3_matched = temp2_mx3[index1]
        data2_nx3_matched = data2_nx3[index2]

        # 2-2 一股脑的全扔进来算R和t，很容易陷入某个局部最优解
        R21_new, t21_new = compute_transform_PQ(data1_mx3_matched, data2_nx3_matched) # 计算新的R和t
        # --- 直接更新R21和t21，没有步长这一说，此时的inlier-ratio就是新的R21和t21得到的
        T21_new = get_T_from_Rt(R21_new, t21_new)
        T21 = np.linalg.multi_dot([T21_new, T21])

        # 2-3 detect if delta R t is small enough
        delta_R = np.sqrt(np.sum((R21 - T21[:3,:3])**2))
        delta_t = np.sqrt(np.sum((t21_new)**2))
        if delta_R < 1e-2 and delta_t < 1e-2:
            break

        R21 = T21[:3,:3]
        t21 = T21[:3, 3]
        T21_inv = np.linalg.inv(T21)
        # print("t21_inv:",T21_inv[:3,3])
        # print("q:      ",get_quaternion_from_R(T21[:3,:3]))
        # print(" ")


    # SHOW THE RESULT
    temp2_3xm = np.linalg.multi_dot([T21[:3,:3], data1_mx3.T]) + T21[:3, 3].reshape(3,1)
    temp2_mx3 = temp2_3xm.T
    # visualize_pc_pair(data2_nx3.T, temp2_3xm)
    T21_inv = np.linalg.inv(T21)
    t21_inv = T21_inv[:3,3]
    q21 = get_quaternion_from_R(T21[:3,:3])

    print("THE BEST t21_inv:",T21_inv[:3,3])
    print("THE BEST q:      ",get_quaternion_from_R(T21[:3,:3]))

    return t21_inv, q21



def main():
    dir = '/home/sunzhengmao/PointNet/深蓝三维点云处理/9.Registration/homework_code/registration_dataset/point_clouds/'
    root_path = "/home/sunzhengmao/PointNet/深蓝三维点云处理/9.Registration/homework_code/registration_dataset/"
    reg_result_path = root_path + 'reg_result.txt'
    point_line = read_reg_results(reg_result_path)

    output = "reg_result2.txt"
    with open(output, "w+") as fwrite:

        for i in range(295, len(point_line)):
            print("=====    [   现在我们开始处理第{}个数据   ]   =====".format(i))
            gt_row = point_line[i]
            gt_idx1, gt_idx2, gt_t, gt_rot = reg_result_row_to_array(gt_row)

            data1_mx6_dir = dir + str(gt_idx1) + '.bin'
            data1_mx6 = read_oxfeord_bin(data1_mx6_dir).T

            data2_nx6_dir = dir + str(gt_idx2) + '.bin'
            data2_nx6 = read_oxfeord_bin(data2_nx6_dir).T
            t21, q21 = compute_ICP_process(data1_mx6, data2_nx6)

            # write the result in txt
            t21 = np.array(t21).reshape(-1)
            q21 = np.array(q21).reshape(-1)
            data = np.r_[t21, q21].astype(str)
            line = str(gt_idx1) + "," + str(gt_idx2)
            for i in range(7):
                line += "," + data[i]
            fwrite.write(line+"\n")
    
    print("YES, WE ARE DONE!")



if __name__ == "__main__":
    main()

    output = "test.txt"
    t21 = np.array([1,2,3]).reshape(-1).astype(float)
    q21 = np.array([0,0,0,0]).reshape(-1).astype(float)
    idx1 = "1"
    idx2 = "2"
    with open(output, "w+") as fwrite:
        data = np.r_[t21, q21].astype(str)
        line = idx1+","+idx2
        for i in range(7):
            line += "," + data[i]
        fwrite.write(line+"\n")
        fwrite.write(line)
    
    
    a = np.array([1,2,3,4,5,6,7,8,9]).reshape(3,3)
    # np.savetxt("test.txt", a, delimiter=',')
    i = 1 

