# 忽略编号，原图与融合结果同号
import os

import cv2
import torch
from torch.autograd import Variable
from net_light import Generator_Nest
import utils_test as utils
from args_fusion import args
import numpy as np
import time
from datetime import datetime


def load_model(path, deepsupervision):
    input_nc = 2
    output_nc = 1
    # nb_filter = [64, 128, 256, 512, 1024]

    G = Generator_Nest(args.nb_filter, input_nc, output_nc, deepsupervision)
    G.load_state_dict(torch.load(path))

    para = sum([np.prod(list(p.size())) for p in G.parameters()])
    type_size = 4
    print('Model {} : params: {:4f}M'.format(G._get_name(), para * type_size / 1000 / 1000))

    G.eval()
    G.cuda()

    return G


def main():
    # run demo
    deepsupervision = False  # true for deeply supervision

    with torch.no_grad():
        if deepsupervision:
            model_path = args.model_deepsuper
        else:
            model_path = args.model_default
        model = load_model(model_path, deepsupervision)

        output_path = './TNO/shengsai_23'

        if not os.path.exists(output_path):
            os.mkdir(output_path)
        output_path = output_path + '/'

        # infrared_dir = "TNO/ir"
        # visible_dir = "TNO/vis"
        infrared_dir = "./test_data/ir"
        visible_dir = "./test_data/vis"
        infrared_files = sorted(os.listdir(infrared_dir))
        visible_files = sorted(os.listdir(visible_dir))

        infrared_path = os.path.join(infrared_dir, infrared_files[0])
        visible_path = os.path.join(visible_dir, visible_files[0])
        img_ir, h, w = utils.get_test_images(infrared_path)
        img_vi, h, w = utils.get_test_images(visible_path)

        if args.cuda:
            img_ir = img_ir.cuda()
            img_vi = img_vi.cuda()
        img_ir = Variable(img_ir, requires_grad=False)
        img_vi = Variable(img_vi, requires_grad=False)

        # fusion
        en_r = torch.cat([img_vi, img_ir], 1).cuda()
        img_fusion = model(en_r)
        start_time = datetime.now()

        print('Processing......  ')
        # Assuming there are equal numbers of infrared and visible images
        for ir_file, vi_file in zip(infrared_files, visible_files):
            infrared_path = os.path.join(infrared_dir, ir_file)
            visible_path = os.path.join(visible_dir, vi_file)

            # Assuming both infrared and visible images have the same filename
            if ir_file != vi_file:
                print(f"Warning: Mismatched files - {ir_file} and {vi_file}")
                continue

            run_demo(model, infrared_path, visible_path, output_path, ir_file)  # Passing the filename directly
        end_time = datetime.now()
        elapsed_time = (end_time - start_time).total_seconds() / 32
        print(f"程序运行时间：{elapsed_time * 32:.4f}avg {elapsed_time}秒")


def run_demo(G, infrared_path, visible_path, output_path_root, filename):
    img_ir, h, w = utils.get_test_images(infrared_path)
    img_vi, h, w = utils.get_test_images(visible_path)

    if args.cuda:
        img_ir = img_ir.cuda()
        img_vi = img_vi.cuda()
    img_ir = Variable(img_ir, requires_grad=False)
    img_vi = Variable(img_vi, requires_grad=False)
    # 启用自动混合精度 (AMP)
    # with torch.cuda.amp.autocast():
    # fusion
    en_r = torch.cat([img_vi, img_ir], 1).cuda()
    img_fusion = G(en_r)

    output_path = os.path.join(output_path_root, filename)  # Using the filename directly for saving
    utils.save_images(img_fusion, output_path)
    print(output_path)


def process_(vis, ir, model):
    with torch.no_grad():
        images_ir = []
        images_vi = []
        infrared_frame = cv2.cvtColor(ir, cv2.COLOR_BGR2GRAY)
        visible_frame = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)
        infrared_frame = np.reshape(infrared_frame, [1, infrared_frame.shape[0], infrared_frame.shape[1]])
        images_ir.append(infrared_frame)
        images_ir = np.stack(images_ir, axis=0)
        images_ir = torch.from_numpy(images_ir).float()
        visible_frame = np.reshape(visible_frame, [1, visible_frame.shape[0], visible_frame.shape[1]])
        images_vi.append(visible_frame)
        images_vi = np.stack(images_vi, axis=0)
        images_vi = torch.from_numpy(images_vi).float()
        if args.cuda:
            img_ir = images_ir.cuda()
            img_vi = images_vi.cuda()
        img_ir = Variable(img_ir, requires_grad=False)
        img_vi = Variable(img_vi, requires_grad=False)
        # 启用自动混合精度 (AMP)
        # with torch.cuda.amp.autocast():
        # fusion
        output_path = os.path.join('./TNO/shengsai_23', "vis.jpg")  # Using the filename directly for saving
        utils.save_images(img_vi, output_path)
        output_path = os.path.join('./TNO/shengsai_23', "ir.jpg")  # Using the filename directly for saving
        utils.save_images(img_ir, output_path)
        en_r = torch.cat([img_vi, img_ir], 1).cuda()
        img_fusion = model(en_r)
        output_path = os.path.join('./TNO/shengsai_23', "now.jpg")  # Using the filename directly for saving
        utils.save_images(img_fusion, output_path)
        if args.cuda:
            # print(img_fusion.data.size())
            # print('img_fusion.data.numpy():', img_fusion.data[0].size())

            img_fusion = img_fusion.cpu().data.numpy()
            # img_fusion = img_fusion.cpu().clamp(0, 255).data[0].numpy()
        else:
            img_fusion = img_fusion.clamp(0, 255).data[0].numpy()

        img_fusion = (img_fusion - np.min(img_fusion)) / (np.max(img_fusion) - np.min(img_fusion))
        img_fusion = img_fusion * 255
        img_fusion = img_fusion.squeeze(0).transpose(1, 2, 0).astype('uint8')
        # cv2.imwrite(output_path, img_fusion)
        if img_fusion.shape[2] == 1:
            img_fusion = img_fusion.reshape([img_fusion.shape[0], img_fusion.shape[1]])

        return img_fusion


if __name__ == '__main__':
    # main()
    ir = cv2.imread('./TNO/ir/32.jpg')
    vis = cv2.imread('./TNO/vis/32.jpg')
    model = load_model("Finally_model.model", False)
    process_(vis, ir, model)
