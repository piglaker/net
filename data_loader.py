import numpy
import os
import torch
import glob
import cv2
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import utils

class SSDataset(Dataset):
    """semantic dataset."""

    def __init__(self, img_dir, label_dir, in_size = 512, out_size = 512, transform=None):
        """
        Args:

        """
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.in_size = in_size
        self.out_size = out_size

    def __len__(self):
        return len()

    def __getitem__(self, idx):
        data = []

        img_name = os.path.join(self.img_dir, '*.jpg')

        for img_file in glob.glob(img_name):
            image = io.imread(img_file)
            image = transform.resize(image, (self.in_size, self.in_size))
            index = img_file.split("\\")[-1].split('.')[0]
            label_name = os.path.join(self.label_dir, str(index) + '.png')
            label = io.imread(label_name)
            label = cv2.resize(label, (512, 512), interpolation = cv2.INTER_CUBIC)
            label = cv2.cvtColor(label, cv2.COLOR_RGB2GRAY)
            label = np.where(label > 0, 1, 0)
            label = label.astype('float')
            data_dict = {'image': image, 'label': [label]}

            if self.transform:
                data_dict = self.transform(data_dict)

            data.append(data_dict)

        return data


if __name__ == "__main__":
    """
    testing
    """

    dir_img = 'img/'
    dir_label = 'label/'
    dir_checkpoint = 'checkpoints/'

    dataset = SSDataset(dir_img, dir_label)[0]

    train_dataset = [dataset[i] for i in range(int(len(dataset) * 0.8))]
    test_dataset = [dataset[i] for i in range(int(len(dataset) * 0.2))]

    batch = utils.batch(train_dataset, 4)

    img = np.array(batch[0][0]).astype(np.float32)
    label = np.array(batch[0][1])

    print(img, img.shape)
    print(label, label.shape)

    p = img[0]
    l = label[0] * 255




    plt.figure()
    plt.imshow(l[0])
    plt.show()

    plt.figure()
    plt.imshow(p)
    plt.show()

    io.imsave('t.jpg', l[0].astype(np.uint8))

    img = torch.from_numpy(img)
    label = torch.from_numpy(label)

    print(label.shape)
