import open3d as o3d 
import numpy as np 

# 点云的显示,利用open3d
def ShowPoint(data:np.ndarray):
    point_cloud = o3d.PointCloud()
    point_cloud.points = o3d.Vector3dVector(data)
    o3d.visualization.draw_geometries([point_cloud],'open3d')
    
def ShowPointCallback(data):
    point_cloud = o3d.PointCloud()
    point_cloud.points = o3d.Vector3dVector(data)
    o3d.visualization.draw_geometries([point_cloud],'open3d')
    callback = o3d.visualization.VisualizerWithKeyCallback()

# 求SVD分解,这里的VT就是转置过后的V
# U,sigma,VT = np.linalg.svd(A)

'''
功能：       根据cluster_index将data涂上不同的颜色
		    cluster_index为-1时代表是噪声，默认涂成黑色；
	    	其余值(0,1,2...)将按照顺序进行上色
data:		nx3的数据
cluster:	(n,)元组的格式
'''
def show_point_color(data, cluster_index):
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
    point_cloud = o3d.PointCloud()
    point_cloud.points = o3d.Vector3dVector(data)
    point_cloud.colors = o3d.Vector3dVector(color)
    o3d.visualization.draw_geometries([point_cloud], "show point & color")
    
