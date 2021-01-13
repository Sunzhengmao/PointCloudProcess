import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

def compute_SPFH_normal(kp_index, eps, data, kd, B=10):
    neighbor = kd.search_radius_vector_3d(data[kp_index, 0:3].reshape(3, 1), radius=eps)
    nbh = np.asarray(neighbor[1])

    dis = np.asarray(neighbor[2]) ** (1 / 2)
    filter = np.logical_and(True, dis != 0)
    index = np.argwhere(filter).flatten()
    nbh = nbh[index]
    dis = dis[index]
    kp = data[kp_index.reshape([-1])]

    # 计算值域，r_th直接当做2π来算吧，虽然arctan会得到-π/2到π/2
    r_a = 2 / B
    r_ph = 2 / B
    r_th = 2 * np.pi / B
    points = data[nbh]
    u = kp[:, 3:6]
    u = u / np.linalg.norm(u)
    p = points[:, 0:3] - kp[:, 0:3]
    p = p / np.linalg.norm(p)
    v = np.cross(p, u.reshape(-1))
    w = np.cross(u, v)

    description = np.zeros([3, 11])
    # 计算triple
    a = np.dot(points[:, 3:6], v.transpose()) + 1
    ph = np.dot(p, u.transpose()) + 1
    th = np.arctan2(np.dot(w, points[:, 3:6].transpose()), np.dot(u, points[:, 3:6].transpose())) + np.pi
    if np.any(th>2*np.pi) or np.any(th<0):
        print("lll")
        th[th>2*np.pi] = 2*np.pi
        th[th<0]     = 0
    # mask = th>np.pi/2
    # while np.any(mask>0):
    #     th[mask] -= np.pi
    #     mask = th>np.pi/2
    # mask = th<-np.pi/2
    # while np.any(mask>0):
    #     th[mask] += np.pi
    #     mask = th<-np.pi/2

    # 给对应的点加1
    num_bin_a = np.floor(a / r_a).astype(int)
    description[0][num_bin_a] += 1
    num_bin_ph = np.floor(ph / r_ph).astype(int)
    description[1][num_bin_ph] += 1
    num_bin_th = np.floor(th / r_th).astype(int)
    description[2][num_bin_th] += 1

    return description, nbh, dis



def compute_FPFH_normal_description(kp_index, eps, data, kd):
    # 1. compute SPFH of query point
    description, nbh, dis = compute_SPFH_normal(kp_index, eps, data, kd=kd)
    description_ = np.zeros_like(description)
    if len(nbh) < 10:
        return np.ones_like(description).flatten() * -1

    # 2. compute SPFH of neighbor points
    for i in range(len(nbh)):
        description_ += compute_SPFH_normal(nbh[i], eps, data, kd=kd)[0] / dis[i] ** 2

    # 3. normalization
    description = description + description_ / len(nbh)
    w = np.sum(description, axis=1, keepdims=True)
    description = description / w
    return description.flatten()


def main():
    data = np.loadtxt('./modelnet40_normal_resampled/chair/chair_0003.txt', delimiter=',')
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    index_1 = np.argwhere(data[:, 0] > 0)
    index = np.argmin(y[index_1])
    index_1 = index_1[index]
    index_2 = np.argwhere(data[:, 0] <= 0)
    index = np.argmin(y[index_2])
    index_2 = index_2[index]

    pcb = o3d.geometry.PointCloud()
    pcb.points = o3d.utility.Vector3dVector(data[:, 0:3])
    kd = o3d.geometry.KDTreeFlann(pcb)
    his_1 = compute_FPFH_normal_description(index_1, eps=0.2, data=data, kd=kd)
    his_2 = compute_FPFH_normal_description(index_2, eps=0.2, data=data, kd=kd)

    fig = plt.figure()
    plt.plot(his_1)
    plt.plot(his_2)
    plt.show()

if __name__ == "__main__":
    main()
