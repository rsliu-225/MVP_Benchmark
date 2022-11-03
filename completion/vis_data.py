import h5py
import open3d as o3d
import numpy as np
import os
import torch
import utils.metrics.EMD.emd_module as emd_model

ROOT = os.path.abspath("./")
COLOR = np.asarray([[31, 119, 180],  [44, 160, 44], [214, 39, 40]]) / 255


def nparray2o3dpcd(nx3nparray_pnts, nx3nparray_nrmls=None, estimate_normals=False):
    o3dpcd = o3d.geometry.PointCloud()
    o3dpcd.points = o3d.utility.Vector3dVector(nx3nparray_pnts[:, :3])
    if nx3nparray_nrmls is not None:
        o3dpcd.normals = o3d.utility.Vector3dVector(nx3nparray_nrmls[:, :3])
    elif estimate_normals:
        o3dpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30))
    return o3dpcd


def show_dataset_both(path, f_name):
    f = h5py.File(f'{path}/{f_name}.h5', 'r')
    print(f.name, f.keys())
    print(len(range(len(f['complete_pcds']))))
    for i in range(len(f['complete_pcds'])):
        print(f['labels'][i])
        print(np.mean(np.asarray(f['complete_pcds'][i]), axis=0))
        o3dpcd_gt = nparray2o3dpcd(np.asarray(f['complete_pcds'][i]))
        o3dpcd_i = nparray2o3dpcd(np.asarray(f['incomplete_pcds'][i]))
        o3dpcd_gt.paint_uniform_color(COLOR[1])
        o3dpcd_i.paint_uniform_color(COLOR[0])
        o3d.visualization.draw_geometries([o3dpcd_i])
        o3d.visualization.draw_geometries([o3dpcd_gt])


def show_dataset(path, f_name):
    f = h5py.File(f'{path}/{f_name}.h5', 'r')
    print(f.name, f.keys())
    for k in f.keys():
        print(k, f[k].shape)
        for v in f[k]:
            try:
                o3dpcd = nparray2o3dpcd(np.asarray(v))
                o3dpcd.paint_uniform_color([0, 0.7, 1])
                o3d.visualization.draw_geometries([o3dpcd])
            except:
                print(v)


def cham3d(pts1, pts2):
    import utils.metrics.CD.chamfer_python as chamfer_python
    mydist1, mydist2, myidx1, myidx2 = chamfer_python.distChamfer(pts1, pts2)
    return mydist1.cpu().mean().sqrt(), mydist1.cpu().mean().sqrt()


def show_res(result_path, test_path, label=1):
    emd = emd_model.emdModule()
    res_f = h5py.File(result_path, 'r')
    test_f = h5py.File(test_path, 'r')

    print(len(test_f['complete_pcds']), len(test_f['complete_pcds']) / 10)
    # cd_list = []
    # emd_list = []
    # for i in range(int(len(test_f['complete_pcds']) / 500)):
    #     gts = torch.from_numpy(np.asarray(test_f['complete_pcds'][i * 500:(i + 1) * 500]).astype(np.float32)).clone()
    #     results = torch.from_numpy(np.asarray(res_f['results'][i * 500:(i + 1) * 500]).astype(np.float32)).clone()
    #     cd1, cd2 = cham3d(gts, results)
    #     emd_dis, _ = emd(gts, results, 0.05, 3000)
    #     cd_list.append((cd1 + cd2) / 2)
    #     emd_list.append(np.sqrt(emd_dis.cpu()).mean())
    # print('cd', np.mean(np.asarray(cd_list)))
    # print('emd', np.mean(np.asarray(emd_list)))

    for i in range(len(test_f['complete_pcds'])):
        if test_f['labels'][i] == label:
            pts_gt = torch.from_numpy(np.asarray([test_f['complete_pcds'][i]]).astype(np.float32)).clone()
            pts_o = torch.from_numpy(np.asarray([res_f['results'][i]]).astype(np.float32)).clone()
            cd1, cd2 = cham3d(pts_gt, pts_o)
            emd_dis, _ = emd(pts_gt, pts_o, 0.05, 3000)
            print('cd', (cd1 + cd2) / 2)
            print('emd', np.sqrt(emd_dis.cpu()).mean())
            o3dpcd_gt = nparray2o3dpcd(np.asarray(test_f['complete_pcds'][i]))
            o3dpcd_i = nparray2o3dpcd(np.asarray(test_f['incomplete_pcds'][i]))
            o3dpcd_o = nparray2o3dpcd(np.asarray(res_f['results'][i]))
            # o3dpcd_o.estimate_normals()
            o3dpcd_gt.paint_uniform_color(COLOR[1])
            o3dpcd_i.paint_uniform_color(COLOR[0])
            o3dpcd_o.paint_uniform_color(COLOR[2])
            o3d.visualization.draw_geometries([o3dpcd_i, o3dpcd_o])
            o3d.visualization.draw_geometries([o3dpcd_o, o3dpcd_gt])


if __name__ == '__main__':
    # show_dataset_both(f'{ROOT}/data_2048_flat/', 'train')
    # show_res(f'{ROOT}/log/vrcnet_cd_mvp/results.h5', f'{ROOT}/data/MVP_Test_CP.h5')
    show_res(f'{ROOT}/log/pcn_emd_flat_mv/results.h5', f'{ROOT}/data_2048_flat/test.h5', label=-2)

