import numpy as np
import open3d as o3d
from dataset import SegDataloader
from train_seg_bn import Model


colors = [[255, 20, 147], [218, 112, 214], [138, 43, 226], [65, 105, 225], [0, 191, 255], [0, 206, 209],
          [144, 238, 144], [255, 215, 0], [255, 140, 0], [210, 105, 30], [240, 128, 128]]
colors = np.array(colors)
test_batch_size = 1
test_loader = SegDataloader(batch_size=test_batch_size, mode="test")
f = Model(save_path="../params-seg-bn/")
f.eval()
f.load()
test_loader.reset()


if __name__ == '__main__':
    for i in range(len(test_loader)):
        pc, label, start_idx = test_loader.get(i)
        pc, label = np.concatenate(pc, axis=0), np.concatenate(label, axis=0)
        y = f(pc, start_idx)
        y_pred = np.argmax(y, axis=1)
        # 构造预测的点云和标签点云
        pc_pred, pc_gt = o3d.PointCloud(), o3d.PointCloud()
        pc_pred.points, pc_gt.points = o3d.Vector3dVector(pc), o3d.Vector3dVector(pc)
        pc_pred.colors, pc_gt.colors = o3d.Vector3dVector(colors[y_pred]/255), o3d.Vector3dVector(colors[label]/255)
        o3d.draw_geometries([pc_pred], width=1000, height=800, window_name="pc pred")
        o3d.draw_geometries([pc_gt], width=1000, height=800, window_name="pc gt")