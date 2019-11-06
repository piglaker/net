import sys
import os
from optparse import OptionParser
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
from skimage import io, transform
from data_loader import SSDataset
from Unet import UNet,Residual_Block


def predict(img, net):
    """

    :param img:
    :param net:
    :return:
    """
    with torch.no_grad():
        output = net(img)


    return output


def get_args():
    parser = OptionParser()
    parser.add_option('-i', '--img', dest='img',
                      default='0.jpg',help='dir of image')
    parser.add_option('-p', '--model_path', dest='model_path',
                      default='checkpoints/unet_CP60.pth', help='load file model')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=True, help='use cuda')
    parser.add_option('-t', '--threshold', dest='threshold',
                      default = 0.8, help='threshold')
    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = get_args()

    net = UNet(in_channels = 3, output_channels = 1, block=Residual_Block)
    #net = UNet(3, 1)
    device_ids = [0, 1, 2, 3, 4, 5]
    os.environ['CUDA_VISIBLE_DEVICES']='4, 5, 6, 7,  8,  9'
    net = nn.DataParallel(net, device_ids=device_ids)

    net.load_state_dict(torch.load(args.model_path))
    print('Model loaded from {}'.format(args.model_path))

    img = io.imread(args.img)
    img = np.array([transform.resize(img, (512, 512)).astype(np.float32)])
    img = torch.from_numpy(img)    
    img = img.permute(0, 3, 1, 2)
    
    if args.gpu:
        net.cuda()
        img = img.cuda()

    output = predict(img, net).cpu().reshape(512, 512, 1)
    print(output.shape)
    output=output*255
    #output = np.where(output > float(args.threshold), 255, 0).astype(np.uint8)
    #output = np.array(output * 255).astype(np.uint8)
    io.imsave('unet_prediction.jpg', output)

