'''
2020-06-20 
孙正茂
'''
import open3d as o3d 
import numpy as np 


def read_txt(data_path):
    result = []
    with open(data_path, "r") as fread:
        point = fread.readline().split(',')
        while len(point)==6:
            result.append([point[0], point[1], point[2]])
            point = fread.readline().split(',')
    return np.array(result).astype('float32')


# 点云的显示,利用open3d
def ShowPoint(data:np.ndarray):
    point_cloud = o3d.PointCloud()
    point_cloud.points = o3d.Vector3dVector(data[:,:3])
    o3d.visualization.draw_geometries([point_cloud],'open3d')

def ShowPointNormal(data, normal):
    point_cloud = o3d.PointCloud()
    point_cloud.points = o3d.Vector3dVector(data[:,:3])
    point_cloud.normals = o3d.Vector3dVector(normal[:,:3])
    o3d.visualization.draw_geometries([point_cloud], 'point_normal')


def ShowPointColor(data:np.ndarray, color:np.array, window_name='point_color'):
    point_cloud = o3d.PointCloud()
    point_cloud.points = o3d.Vector3dVector(data[:,:3])
    point_cloud.colors = o3d.Vector3dVector(color)
    o3d.visualization.draw_geometries([point_cloud], window_name)
    
def ShowPointCallback(data):
    point_cloud = o3d.PointCloud()
    point_cloud.points = o3d.Vector3dVector(data[:,:3])
    o3d.visualization.draw_geometries([point_cloud],'open3d')
    callback = o3d.visualization.VisualizerWithKeyCallback()

# 求SVD分解,这里的VT就是转置过后的V
# U,sigma,VT = np.linalg.svd(A)

def ShowPointColorNormal(data, cluster_index, normal=None):
    '''
    功能:根据cluster_index将data涂上不同的颜色
        cluster_index为-1时代表是噪声，默认涂成黑色；
        其余值(0,1,2...)将按照顺序进行上色
    data:nx3的数据
    cluster:(n,)元组的格式
    '''
    cluster_index = np.array(cluster_index)
    [N, d] = data.shape
    colors = [[0,1,0],[1,0,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1]]
    # colors = [[0,1,0],[0,0,1],[0.7,0.7,0.7],[1,0,0],[1,0,0.5],[1,0,1],[1,0.5,0],[1,0.5,0.5],[1,0.5,1],[1,1,0],[1,1,0.5],
    #            [0.5,0,0],[0.5,0,0.5],[0.5,0,1],[0.5,0.5,0],[0.5,0.5,0.5],[0.5,0.5,1],[0.5,1,0],[0.5,1,0.5],[0.5,1,1],
    #            [0,0,0.5],[0,0.5,0],[0,0.5,0.5],[0,0.5,1],[0,1,0.5],[0,1,1]]
    colors = np.array(colors)

    n_cluster = int(np.max(cluster_index)+1) # 一共有多少个聚类
    n_color = len(colors) # 事先设定了多少个颜色
    color = np.zeros((N,3))
    for i in range(n_cluster):
        mask = cluster_index == i
        color[mask] = colors[i%n_color,:]
    color[cluster_index==-1] = np.array([0,0,0]) # 黑色
    # color[cluster_index==-1] = np.array([255,255,255]) # 白色
    
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(data[:,:3])
    point_cloud.colors = o3d.utility.Vector3dVector(color)
    if normal is not None:
        point_cloud.normals = o3d.Vector3dVector(normal)
    o3d.visualization.draw_geometries([point_cloud], "show point & color")

def ShowPointColorAxis(query, axis, neibor, fulldata):
    '''
    除了颜色显示，还通过法向量显示的功能来显示轴向
    '''
    N = fulldata.shape[0] + neibor.shape[0]
    color_green = np.tile(np.array([0,1,0]), (fulldata.shape[0],1))
    color_red = np.tile(np.array([1,0,0]), (neibor.shape[0], 1))

    color = np.concatenate((np.array([1,0,0]).reshape(1,3), color_red, color_green), axis=0)
    data = np.concatenate((query[:3].reshape(1,3), neibor[:,:3], fulldata[:,:3]), axis=0)
    normal = np.tile(axis.reshape(1,3), (N+1,1))
    normal[1+neibor.shape[0] : ] *= 1e-3

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(data)
    point_cloud.colors = o3d.utility.Vector3dVector(color)
    point_cloud.normals = o3d.Vector3dVector(normal) 
    o3d.visualization.draw_geometries([point_cloud], "show point & color")   
    
def read_from_asc(data_path):
    '''
    从asc文件里面读取data数据
    '''
    data = []
    with open(data_path, "r+") as f:
        line = f.readline()
        while line:
            if line[0] == '#':
                line = f.readline()
                continue
            data.append(line[:-1].split(' '))
            line = f.readline()
    
    return np.array(data).astype("float32")


def generate_cylinder(radius=20, tall=100):
    '''
    生成一个半径为radius，高为tall的标准圆柱体
    '''
    data_nx3 = []
    full_cloud_nx6 = []
    for z in range(tall):
        center = np.array([0,0,z])
        for j in range(30):
            thita = np.random.random() * np.pi
            point = np.array([radius*np.cos(thita), radius*np.sin(thita), z])
            normal = point - center
            data_nx3.append(point)
            full_cloud_nx6.append(np.append(point,normal))

    data_nx3 = np.array(data_nx3).astype("float32")
    full_cloud_nx6 = np.array(full_cloud_nx6).astype("float32")
    np.savetxt("cylinder_noNormal.txt", data_nx3)
    np.savetxt("cylinder_normal.txt", full_cloud_nx6)

    return data_nx3, full_cloud_nx6

def voxel_sample_based_o3d(points, voxel_size, show=False):
    '''
    用open3d的voxel_down_sample做了下包装, 就不自己写了
    '''
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.Vector3dVector(points[:,:3])
    downpcd = o3d.geometry.voxel_down_sample(pcd, voxel_size)
    if show:
        o3d.visualization.draw_geometries([pcd], "BEFORE sample") 
        o3d.visualization.draw_geometries([downpcd], "AFTER sample") 
    return np.array(downpcd.points)

def ShowPointColorByList(points_list):
    '''
    除了颜色显示，还通过法向量显示的功能来显示轴向
    '''
    colors = np.array([[0,1,0], [1,0,0], [255/255, 127/255, 80/255], [128/255,0,128/255],[1,1,0],[0,1,1], [0,1,0],[1,0,0]]) # 绿色，红色，橙色，紫色
    color_nx3 = np.array([])
    point_nx3 = np.array([])

    for idx, points in enumerate(points_list):
        i = idx % colors.shape[0]
        color_i = np.tile(colors[i], (points.shape[0],1))
        color_nx3 = np.vstack([color_nx3, color_i]) if color_nx3.size else color_i
        point_nx3 = np.vstack([point_nx3, points[:,:3]]) if point_nx3.size else points[:,:3]

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(point_nx3)
    point_cloud.colors = o3d.utility.Vector3dVector(color_nx3)
    o3d.visualization.draw_geometries([point_cloud], "show point & color")  

def loadFile(name):
    data = np.array([])
    file_type = name.split('.')[-1]
    if file_type == "asc" or file_type=="txt":
        data = read_from_asc(name)
    elif file_type == "npy":
        data = np.load(name)
    else:
        print("文件类型有误")
        data = None
    return data
    