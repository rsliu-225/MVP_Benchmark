import logging
import os
import sys
import importlib
import argparse
import numpy as np
import h5py
import subprocess

from numpy.lib.index_tricks import AxisConcatenator
import munch
import yaml
from vis_utils import plot_single_pcd
import torch
import open3d as o3d

import warnings

warnings.filterwarnings("ignore")


def inference_sgl(input_data, model_name, load_model):
    args = munch.munchify({'num_points': 2048, 'loss': 'cd', 'eval_emd': False})
    input_data = np.asarray(input_data)
    input_data = torch.from_numpy(input_data)
    model_module = importlib.import_module('.%s' % model_name, 'models')
    net = torch.nn.DataParallel(model_module.Model(args))
    net.cuda()
    net.module.load_state_dict(torch.load(load_model)['net_state_dict'])
    logging.info("%s's previous weights loaded." % model_name)
    net.eval()

    logging.info('Testing...')
    with torch.no_grad():
        inputs_cpu = input_data
        # torch.Size([64, 2048, 3])
        # inputs_cpu = input_data

        inputs = inputs_cpu.float().cuda()
        inputs = torch.unsqueeze(inputs, 0)
        # inputs = inputs.transpose(2, 1).contiguous()
        inputs = inputs.transpose(2, 1).contiguous()
        inputs = inputs.repeat(64, 1, 1)
        # torch.Size([64, 3, 2048])
        print(inputs.shape)
        result_dict = net(inputs, prefix="test")
        output = result_dict['result'].cpu().numpy()
    return output[0]


def nparray2o3dpcd(nx3nparray_pnts, nx3nparray_nrmls=None, estimate_normals=False):
    o3dpcd = o3d.geometry.PointCloud()
    o3dpcd.points = o3d.utility.Vector3dVector(nx3nparray_pnts[:, :3])
    if nx3nparray_nrmls is not None:
        o3dpcd.normals = o3d.utility.Vector3dVector(nx3nparray_nrmls[:, :3])
    elif estimate_normals:
        o3dpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30))
    return o3dpcd


if __name__ == "__main__":
    COLOR = np.asarray([[31, 119, 180], [44, 160, 44], [214, 39, 40]]) / 255
    f_name = 'test'

    # f = h5py.File(f'{GOAL_DATA_PATH}/{f_name}.h5', 'r')
    # print(f.name, f.keys())
    # for i in range(len(f['complete_pcds'])):
    #     if f['labels'][i] == 1:
    #         o3dpcd_i = nparray2o3dpcd(np.asarray(f['incomplete_pcds'][i]))
    #         o3dpcd_gt = nparray2o3dpcd(np.asarray(f['complete_pcds'][i]))
    #         o3dpcd_i.paint_uniform_color(COLOR[0])
    #         o3dpcd_gt.paint_uniform_color(COLOR[1])
    #         result = inference_sgl(np.asarray(f['incomplete_pcds'][i]))
    #         print(result)
    #         o3dpcd_o = nparray2o3dpcd(result)
    #         o3dpcd_o.paint_uniform_color(COLOR[2])
    #         o3d.visualization.draw_geometries([o3dpcd_o, o3dpcd_i])
    #         o3d.visualization.draw_geometries([o3dpcd_o, o3dpcd_gt])
    model_name = 'pcn'
    load_model = 'D:/liu/MVP_Benchmark/completion/log/pcn_emd_prim_mv/best_cd_p_network.pth'

    o3dpcd_1 = o3d.io.read_point_cloud('D:/liu/MVP_Benchmark/completion/data_real/000.pcd')
    o3dpcd_2 = o3d.io.read_point_cloud('D:/liu/MVP_Benchmark/completion/data_real/001.pcd')
    o3dpcd = o3dpcd_1
    o3dpcd = o3dpcd.uniform_down_sample(int(len(np.asarray(o3dpcd.points)) / 2048))
    print(np.asarray(o3dpcd.points).shape)
    result = inference_sgl(np.asarray(o3dpcd.points), model_name, load_model)
    o3dpcd_o = nparray2o3dpcd(result)
    o3dpcd.paint_uniform_color(COLOR[0])
    o3dpcd_o.paint_uniform_color(COLOR[2])
    o3d.visualization.draw_geometries([o3dpcd, o3dpcd_o])
