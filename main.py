import numpy as np
import open3d as o3d


if __name__ == '__main__':
    # x1 = "E:/shapenet_benchmark_v0_normal/03624134/1be1aa66cd8674184e09ebaf49b0cb2f.txt"
    x1 = "E:/shapenet_benchmark_v0_normal/03948459/1a640c8dffc5d01b8fd30d65663cfd42.txt"
    # x1 = "E:/shapenet_benchmark_v0_normal/03790512/2d655fc4ecb2df2a747c19778aa6cc0.txt"
    with open(x1, "r") as f:
        pts = np.array([[float(x) for x in l.split(" ")[:6]] for l in f.readlines()])
    # print(set(list(pts[:, -1])))
    pc = o3d.PointCloud()
    pc.points = o3d.Vector3dVector(pts[:, :3])
    # pc.colors = o3d.Vector3dVector([[1, 20/255, 147/255]]*pts.shape[0])
    # pc.colors = o3d.Vector3dVector([[135/255, 206/255, 250/255]] * pts.shape[0])
    pc.colors = o3d.Vector3dVector([[1, 165/255, 0]] * pts.shape[0])
    # pc.normals = o3d.Vector3dVector(pts[:, 3:])
    o3d.draw_geometries([pc], window_name="x1", width=1000, height=800)