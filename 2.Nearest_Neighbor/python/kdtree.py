# kdtree的具体实现，包括构建和查找

import random
import math
import numpy as np
import copy

from result_set import KNNResultSet, RadiusNNResultSet

# Node类，Node是tree的基本组成元素
class Node:
    def __init__(self, axis, value, left, right, point_indices):
        self.axis = axis # 是朝哪个方向进行拆分的
        self.value = value # 这个节点的值
        self.left = left
        self.right = right
        self.point_indices = point_indices # 是指这个node左右所有的点吧

    def is_leaf(self):
        if self.value is None:
            return True
        else:
            return False

    def __str__(self):
        output = ''
        output += 'axis %d, ' % self.axis
        if self.value is None:
            output += 'split value: leaf, '
        else:
            output += 'split value: %.2f, ' % self.value
        output += 'point_indices: '
        output += str(self.point_indices.tolist())
        return output

# 功能：构建树之前需要对value进行排序，同时对一个的key的顺序也要跟着改变
#      为了得到中序值吧可能
# 输入：
#     key：键
#     value:值
# 输出：
#     key_sorted：排序后的键
#     value_sorted：排序后的值
def sort_key_by_vale(key, value):
    assert key.shape == value.shape
    assert len(key.shape) == 1
    sorted_idx = np.argsort(value) # index
    key_sorted = key[sorted_idx]
    value_sorted = value[sorted_idx]
    return key_sorted, value_sorted

# 功能: 得到第k个最小值
# 输入: 
#       number_list     待排序的数组 
#       left            最左边值的索引
#       right           最右边值的索引
#       k               想找的第k个最小值
# 输出:
#       第k个最小值
def get_k_min(number_list, left, right, k):
    if left == right:
        return number_list[left]
    if left > right:
        print("left > right, wrong...")
        return -1
    
    cut = Partition(number_list, left, right)
    cut_index = cut - left + 1# 减去起始点影响
    if cut_index == k:
        return number_list[cut]
    if cut_index > k:
        return get_k_min(number_list, left, cut-1, k)
    if cut_index < k:
        return get_k_min(number_list, cut+1, right, k-cut_index)
    
    return 0    

# 功能:  得到随便一个分割的index
# 输入:  
#       number_list     待排序的索引
#       left            最左边的值的索引
#       right           最右边值的索引 
# 输出:
#       index           随机一个值(倒数第二个)在排序好之后应在的位置索引
def Partition(number_list, left, right):
    if left == right:
        return left
    if left > right:
        return -1
    
    # middle = (right - left) >> 1 + left
    temp_value = number_list[right-1]
    number_list[right-1] = number_list[right]

    while(left != right):
        while(number_list[left] <= temp_value and left < right):
            left += 1
        if left < right:
            number_list[right] = number_list[left]
            right -= 1
        while(number_list[right] > temp_value and left < right):
            right -= 1
        if left < right:
            number_list[left] = number_list[right]
            left += 1
        
    number_list[left] = temp_value
    return left

# 功能: 记录当前轴的方向,然后变到下一个轴上
def axis_round_robin(axis, dim):
    if axis == dim-1:
        return 0
    else:
        return axis + 1

# 功能：通过递归的方式构建树
# 输入：
#     root: 树的根节点
#     db: 点云数据
#     point_indices：排序后的键
#     axis: scalar
#     leaf_size: scalar
# 输出：
#     root: 即构建完成的树
def kdtree_recursive_build(root, db, point_indices, axis, leaf_size):
    if root is None:
        root = Node(axis, None, None, None, point_indices)

    # determine whether to split into left and right
    if len(point_indices) > leaf_size:
        # --- get the split position ---
        # 作业1
        # 屏蔽开始

        # ----------- | 老师给的代码 | ------------
        # N = len(point_indices)
        # if N > 200:
        #     index_temp = np.random.choice(point_indices, 200)
        # else:
        #     index_temp = point_indices
        # point_indices_sorted, _ = sort_key_by_vale(index_temp, db[index_temp, axis])  # M
        # # point_indices_sorted, _ = sort_key_by_vale(point_indices, db[point_indices, axis])  # M

        # middle_left_idx = math.ceil(point_indices_sorted.shape[0]>>1)-1
        # middle_left_point_index = point_indices_sorted[middle_left_idx]
        # middle_left_point_value = db[middle_left_point_index, axis]

        # middle_right_idx = middle_left_idx + 1
        # middle_right_point_index = point_indices_sorted[middle_right_idx]
        # middle_right_point_value = db[middle_right_point_index, axis]

        # root.value = (middle_left_point_value + middle_right_point_value) * 0.5

        # axis = axis_round_robin(axis,dim=db.shape[1])

        # point_indices_left = point_indices[db[point_indices,axis] < root.value]
        # point_indices_right= point_indices[db[point_indices,axis] >= root.value]

        # root.left = kdtree_recursive_build(root.left, 
        #                                    db,
        #                                    point_indices[db[point_indices,axis] < root.value], 
        #                                    axis, 
        #                                    leaf_size)
        # root.right= kdtree_recursive_build(root.right, 
        #                                    db, 
        #                                    point_indices[db[point_indices,axis] >= root.value],
        #                                    axis, 
        #                                    leaf_size)
        # N = len(point_indices)
        # if N > 200:
        #     index_temp = np.random.choice(point_indices, 200)
        # else:
        #     index_temp = point_indices
        # point_indices_sorted, _ = sort_key_by_vale(index_temp, db[index_temp, axis])  # M
        # # point_indices_sorted, _ = sort_key_by_vale(point_indices, db[point_indices, axis])  # M

        # middle_left_idx = math.ceil(point_indices_sorted.shape[0]>>1)-1
        # middle_left_point_index = point_indices_sorted[middle_left_idx]
        # middle_left_point_value = db[middle_left_point_index, axis]

        # middle_right_idx = middle_left_idx + 1
        # middle_right_point_index = point_indices_sorted[middle_right_idx]
        # middle_right_point_value = db[middle_right_point_index, axis]

        # root.value = (middle_left_point_value + middle_right_point_value) * 0.5

        # axis = axis_round_robin(axis,dim=db.shape[1])

        # root.left = kdtree_recursive_build(root.left, 
        #                                    db,
        #                                    point_indices_sorted[:middle_right_idx], 
        #                                    axis, 
        #                                    leaf_size)
        # root.right= kdtree_recursive_build(root.right, 
        #                                    db, 
        #                                    point_indices_sorted[middle_right_idx:],
        #                                    axis, 
        #                                    leaf_size)


        # -------------- | 自己尝试:加了找中值的代码 | ------------
        # N = len(point_indices)
        # k = N >> 1 # 中位数
        # temp_db = copy.deepcopy(db[point_indices, axis]) # 用自己写的找最小值,因为有while循环,所以不是很快
        # middle_value = get_k_min(temp_db, 0, N-1, k)

        # # 用mask来回避for循环--------------
        # middle_left_point_mask = db[point_indices, axis] <= middle_value
        # middle_left_point_index = point_indices[middle_left_point_mask]

        # middle_right_point_mask = db[point_indices, axis] > middle_value
        # middle_right_point_index = point_indices[middle_right_point_mask]

        # # 这样确实挺慢的
        # # middle_left_point_index = [i for i in point_indices if db[i, axis]<=middle_value]
        # # middle_right_point_index = [i for i in point_indices if db[i, axis]>middle_value]

        # root.value = (temp_db[k] + temp_db[k+1]) * 0.5

        # axis = axis_round_robin(axis,dim=db.shape[1])

        # root.left = kdtree_recursive_build(root.left, 
        #                                    db,
        #                                    middle_left_point_index, 
        #                                    axis, 
        #                                    leaf_size)
        # root.right= kdtree_recursive_build(root.right, 
        #                                    db, 
        #                                    middle_right_point_index,
        #                                    axis, 
        #                                    leaf_size)

        # -------------- | 自己尝试:直接用均值来代替中值,以加快速度 | ------------
        N = len(point_indices)
        k = N >> 1 # 中位数
        if N > 150:
            index_temp = np.random.choice(point_indices, 20)
        else:         
            index_temp = point_indices
        middle_value = np.mean(db[index_temp, axis])

        # 用mask来回避for循环,取出左右两边的点的index --------------
        middle_left_point_mask = db[point_indices, axis] <= middle_value
        middle_left_point_index = point_indices[middle_left_point_mask]
        middle_right_point_mask = db[point_indices, axis] > middle_value
        middle_right_point_index = point_indices[middle_right_point_mask]

        root.value = middle_value

        root.left = kdtree_recursive_build(root.left, 
                                           db,
                                           middle_left_point_index, 
                                           axis_round_robin(axis,dim=db.shape[1]), 
                                           leaf_size)
        root.right= kdtree_recursive_build(root.right, 
                                           db, 
                                           middle_right_point_index,
                                           axis_round_robin(axis,dim=db.shape[1]), 
                                           leaf_size)

        # 屏蔽结束
    return root


# 功能：翻转一个kd树
# 输入：
#     root：kd树
#     depth: 当前深度
#     max_depth：最大深度
def traverse_kdtree(root: Node, depth, max_depth):
    depth[0] += 1
    if max_depth[0] < depth[0]:
        max_depth[0] = depth[0]

    if root.is_leaf():
        print(root)
    else:
        traverse_kdtree(root.left, depth, max_depth)
        traverse_kdtree(root.right, depth, max_depth)

    depth[0] -= 1

# 功能：构建kd树（利用kdtree_recursive_build功能函数实现的对外接口）
# 输入：
#     db_np：原始数据
#     leaf_size：scale
# 输出：
#     root：构建完成的kd树
def kdtree_construction(db_np, leaf_size):
    N, dim = db_np.shape[0], db_np.shape[1]

    # build kd_tree recursively
    root = None
    root = kdtree_recursive_build(root,
                                  db_np,
                                  np.arange(N),
                                  axis=0,
                                  leaf_size=leaf_size)
    return root


# 功能：通过kd树实现knn搜索，即找出最近的k个近邻
# 输入：
#     root: kd树
#     db: 原始数据
#     result_set：搜索结果
#     query：索引信息
# 输出：
#     搜索失败则返回False
def kdtree_knn_search(root: Node, db: np.ndarray, result_set: KNNResultSet, query: np.ndarray):
    if root is None:
        return False

    if root.is_leaf():
        # compare the contents of a leaf
        leaf_points = db[root.point_indices, :]
        diff = np.linalg.norm(np.expand_dims(query, 0) - leaf_points, axis=1)
        for i in range(diff.shape[0]):
            result_set.add_point(diff[i], root.point_indices[i])
        return False

    # 作业2
    # 提示：仍通过递归的方式实现搜索
    # 屏蔽开始
    # 1.先找子节点
    if query[root.axis] <= root.value:
        kdtree_knn_search(root.left, db, result_set, query)
        if math.fabs(query[root.axis] - root.value) < result_set.worstDist():#如果这个圆伸到了另一半区域里
            kdtree_knn_search(root.right, db, result_set, query) # 就搜完左边搜右边,搜完上边搜下边
        
    else:
        kdtree_knn_search(root.right, db, result_set, query)
        if math.fabs(query[root.axis] - root.value) < result_set.worstDist():
            kdtree_knn_search(root.left, db, result_set, query)

    # 屏蔽结束

    return False

# 功能：通过kd树实现radius搜索，即找出距离radius以内的近邻
# 输入：
#     root: kd树
#     db: 原始数据
#     result_set:搜索结果
#     query：索引信息
# 输出：
#     搜索失败则返回False
def kdtree_radius_search(root: Node, db: np.ndarray, result_set: RadiusNNResultSet, query: np.ndarray):
    if root is None:
        return False

    if root.is_leaf():
        # compare the contents of a leaf
        leaf_points = db[root.point_indices, :]
        diff = np.linalg.norm(np.expand_dims(query, 0) - leaf_points, axis=1)
        for i in range(diff.shape[0]):
            result_set.add_point(diff[i], root.point_indices[i])
        return False
    
    # 作业3
    # 提示：通过递归的方式实现搜索
    # 屏蔽开始
    # 1. 老规矩,先找子节点
    if query[root.axis] <= root.value:
        kdtree_radius_search(root.left, db, result_set, query)#只要找到了,就得一层一层向上去翻,没有提前终止条件
        if math.fabs(query[root.axis] - root.value) < result_set.worstDist(): #如果找完之后发现这个圆与另一半有相交
            kdtree_radius_search(root.right, db, result_set, query)
    
    else:
        kdtree_radius_search(root.right, db, result_set, query)
        if math.fabs(query[root.axis] - root.value) < result_set.worstDist(): #最短距离都伸到了另一半里了
            kdtree_radius_search(root.left, db, result_set, query)

    # 屏蔽结束

    return False



def main():
    # configuration
    db_size = 6400
    dim = 3
    leaf_size = 4
    k = 5

    db_np = np.random.rand(db_size, dim)

    root = kdtree_construction(db_np, leaf_size=leaf_size)

    depth = [0]
    max_depth = [0]
    traverse_kdtree(root, depth, max_depth)
    print("tree max depth: %d" % max_depth[0])

    query = np.asarray([0, 0, 0])
    result_set = KNNResultSet(capacity=k)
    kdtree_knn_search(root, db_np, result_set, query)
    
    print(result_set)
    
    diff = np.linalg.norm(np.expand_dims(query, 0) - db_np, axis=1)
    nn_idx = np.argsort(diff)
    nn_dist = diff[nn_idx]
    print(nn_idx[0:k])
    print(nn_dist[0:k])
    
    
    print("Radius search:")
    query = np.asarray([0, 0, 0])
    result_set = RadiusNNResultSet(radius = 0.5)
    kdtree_radius_search(root, db_np, result_set, query)
    print(result_set)


if __name__ == '__main__':
    main()
    number = [9,10,4,5,2,6,0,7,1]
    value = get_k_min(number, 0, len(number)-1, 2)


    i = 1