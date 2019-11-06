import sys
import os
from optparse import OptionParser
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim

from data_loader import SSDataset
from Unet import UNet, Residual_Block
import utils
from loss import dice_bce_loss,FocalLoss
import matplotlib.pyplot as plt

def train_net(net,
              epochs=5,
              batch_size=1,
              lr=0.001,
              save_cp=True,
              gpu=False,
              ):

    dir_img = 'img/'
    dir_label = 'label/'
    dir_checkpoint = 'checkpoints/'

    dataset =SSDataset(dir_img, dir_label)[0]

    train_dataset = [dataset[i] for i in range(int(len(dataset) * 0.8))]
    test_dataset = [dataset[i] for i in range(int(len(dataset) * 0.2))]

    print('''
    Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
        Validation size: {}
        Checkpoints: {}
        CUDA: {}
    '''.format(epochs, batch_size, lr, len(train_dataset),
               len(test_dataset), str(save_cp), str(gpu)))

    optimizer = optim.SGD(net.parameters(),
                          lr = lr,
                          momentum = 0.9,
                          weight_decay = 0.0005)
    criterion = dice_bce_loss()
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda:0" if USE_CUDA else "cpu")
    loss_graph = []
    x = []
    count = 0
    for epoch in range(epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        net.train()
        epoch_loss = 0
        batch = utils.batch(train_dataset, batch_size)

        for o in range(len(batch)):

            img = np.array(batch[o][0]).astype(np.float32)
            label = np.array(batch[o][1])

            img = torch.from_numpy(img)
            label = torch.from_numpy(label)

            img = img.permute(0, 3, 1, 2)
            #label = label.reshape(batch_size, 1, 512, 512)

            if gpu:
                img = img.cuda()
                label = label.cuda()
            
            #net = torch.nn.DataParallel(net).cuda()
            net.to(device)
            output = net(img)

            output_final = output.view(-1).float()
            label_final = label.view(-1).float()
            #print(label_final)

            loss = criterion(output_final, label_final)
            epoch_loss = epoch_loss + loss.item()

            optimizer.zero_grad()
            with torch.autograd.set_detect_anomaly(True):
                loss.backward()
            optimizer.step()

        print('Epoch finished ! Loss: {}'.format(epoch_loss / (epoch + 1)))
        loss_graph.append(epoch_loss/ (epoch + 1))
        count += 1
        x.append(count)

        if save_cp:
            torch.save(net.state_dict(),
                       dir_checkpoint + 'unet_CP{}.pth'.format(epoch + 1))
            print('Checkpoint {} saved !'.format(epoch + 1))

    plt.figure()
    plt.plot(x, loss_graph)
    plt.savefig("unet_loss.jpg")


def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default = 5, type = 'int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=2,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.001,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=True, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')


    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = get_args()

    net = UNet(in_channels = 3, output_channels = 1, block=Residual_Block)
    #net = UNet( 3,  1)
    device_ids = [0, 1, 2, 3, 4, 5]
    os.environ['CUDA_VISIBLE_DEVICES']='4, 5, 6, 7,  8,  9'
    net = nn.DataParallel(net, device_ids=device_ids)

    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))

    if args.gpu:
        net.cuda()


    train_net(net=net,
              epochs=args.epochs,
              batch_size=args.batchsize,
              lr=args.lr,
              gpu=args.gpu,)

