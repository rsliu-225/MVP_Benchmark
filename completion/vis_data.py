import h5py
import open3d as o3d
import numpy as np
import os
import mmcv.ops


def nparray2o3dpcd(nx3nparray_pnts, nx3nparray_nrmls=None, estimate_normals=False):
    o3dpcd = o3d.geometry.PointCloud()
    o3dpcd.points = o3d.utility.Vector3dVector(nx3nparray_pnts[:, :3])
    if nx3nparray_nrmls is not None:
        o3dpcd.normals = o3d.utility.Vector3dVector(nx3nparray_nrmls[:, :3])
    elif estimate_normals:
        o3dpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30))
    return o3dpcd


train_f = h5py.File(f'{os.path.abspath("./")}/data/MVP_Train_CP.h5', 'r')
test_f = h5py.File(f'{os.path.abspath("./")}/data/MVP_Test_CP.h5', 'r')
res_f = h5py.File(f'{os.path.abspath("./")}/log/pcn_cd_debug_2022/results.h5', 'r')
# for k in res_f.keys():
#     print(k)
#     for pcd in res_f[k]:
#         o3dpcd = nparray2o3dpcd(np.asarray(pcd))
#         o3d.visualization.draw_geometries([o3dpcd])
print(train_f.name, train_f.keys())
print(test_f.name, test_f.keys())
print(res_f.name, res_f.keys())

for k in train_f.keys():
    print(k)
    print(train_f[k])
    for pcd in train_f[k][:2]:
        print(pcd, len(np.asarray(pcd)))
        o3dpcd = nparray2o3dpcd(np.asarray(pcd))
        o3d.visualization.draw_geometries([o3dpcd])

for k in test_f.keys():
    print(k)
    for pcd in test_f[k][:2]:
        print(pcd, len(np.asarray(pcd)))
        o3dpcd = nparray2o3dpcd(np.asarray(pcd))
        o3d.visualization.draw_geometries([o3dpcd])

# o3dpcd_i = nparray2o3dpcd()
# o3dpcd_o = nparray2o3dpcd()
#
# o3dpcd_i.paint_uniform_color([0, 0.706, 1])
# o3dpcd_o.paint_uniform_color([0.706, 0, 1])
# o3d.visualization.draw_geometries([o3dpcd_i, o3dpcd_o])
