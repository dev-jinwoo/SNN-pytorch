import numpy as np
import random
from PIL import Image
import torch
import os
from random import Random
import Augmentor

import torchvision.datasets as dset
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

subject = 'train'
test_subject = 'test'


def get_train_valid_loader(data_dir,
                           batch_size,
                           num_train,
                           augment,
                           way,
                           trials,
                           shuffle=False,
                           seed=0,
                           num_workers=4,
                           pin_memory=False):
    train_dir = os.path.join(data_dir, subject)
    valid_dir = os.path.join(data_dir, test_subject)

    train_dataset = dset.ImageFolder(root=train_dir)
    train_dataset = OmniglotTrain(train_dataset, num_train, augment)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    valid_dataset = dset.ImageFolder(root=valid_dir)
    valid_dataset = OmniglotTest(
        valid_dataset, trials=trials, way=way, seed=seed,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=way, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return (train_loader, valid_loader)


def get_test_loader(data_dir,
                    way,
                    trials,
                    seed=0,
                    num_workers=4,
                    pin_memory=False):
    test_dir = os.path.join(data_dir, test_subject)
    test_dataset = dset.ImageFolder(root=test_dir)
    test_dataset = OmniglotTest(
        test_dataset, trials=trials, way=way, seed=seed,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=way, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    return test_loader


# adapted from https://github.com/fangpin/siamese-network
class OmniglotTrain(Dataset):
    def __init__(self, dataset, num_train, augment=False):
        super(OmniglotTrain, self).__init__()
        self.dataset = dataset
        self.num_train = num_train
        self.augment = augment

    def occlude_image(self, src_img, prob=2):
        aug_prob = random.uniform(0, prob)

        np_img = np.array(src_img)

        color = (200 + random.randint(-30, 30), 150 + random.randint(-30, 30), 50 + random.randint(-30, 30))

        width, height = np_img.shape[0], np_img.shape[1]

        if 0 < aug_prob < 0.25:
            np_img[:, :height // 2, :] = color
        elif 0.25 < aug_prob < 0.5:
            np_img[:, height // 2:, :] = color
        elif 0.5 < aug_prob < 0.75:
            np_img[width // 2:, :, :] = color
        elif 0.75 < aug_prob < 1:
            np_img[:width // 2, :, :] = color

        tgt_img = Image.fromarray(np_img)

        return tgt_img

    def __len__(self):
        return self.num_train

    def __getitem__(self, index):
        image1 = random.choice(self.dataset.imgs)

        # get image from same class
        label = None
        if index % 2 == 1:
            label = 1.0
            while True:
                image2 = random.choice(self.dataset.imgs)
                if image1[1] == image2[1]:
                    break
        # get image from different class
        else:
            label = 0.0
            while True:
                image2 = random.choice(self.dataset.imgs)
                if image1[1] != image2[1]:
                    break
        if 'draw' not in image1[0].split('/')[-1]:
            image1 = Image.open(image1[0])
            image1 = self.occlude_image(image1)
        else:
            image1 = Image.open(image1[0])

        image2 = Image.open(image2[0])

        image1 = image1.convert('L')
        image2 = image2.convert('L')

        # apply transformation on the fly
        if self.augment:
            p = Augmentor.Pipeline()
            p.flip_left_right(probability=0.3)
            p.rotate90(probability=0.3)

            trans = transforms.Compose([
                transforms.Grayscale(1),
                transforms.Resize((210, 210)),
                p.torch_transform(),
                transforms.ToTensor(),

            ])

            trans2 = transforms.Compose([
                transforms.Grayscale(1),
                transforms.Resize((210, 210)),
                transforms.ToTensor(),
            ])
        else:
            trans = transforms.Compose([
                transforms.Grayscale(1),
                transforms.Resize((210, 210)),
                transforms.ToTensor()
            ])

        image1 = trans(image1)
        image2 = trans2(image2)
        y = torch.from_numpy(np.array([label], dtype=np.float32))
        return (image1, image2, y)


# adapted from https://github.com/fangpin/siamese-network
class OmniglotTest(Dataset):
    def __init__(self, dataset, trials, way, seed=0):
        super(OmniglotTest, self).__init__()
        self.dataset = dataset
        self.trials = trials
        self.way = way
        self.transform = transforms.Compose([
            transforms.Grayscale(1),
            transforms.Resize((210, 210)),
            transforms.ToTensor()
        ])
        self.seed = seed

    def __len__(self):
        return (self.trials * self.way)

    def __getitem__(self, index):
        self.rng = Random(self.seed + index)

        idx = index % self.way
        label = None
        # generate image pair from same class
        if idx == 0:
            self.img1 = self.rng.choice(self.dataset.imgs)
            while True:
                img2 = self.rng.choice(self.dataset.imgs)
                if self.img1[1] == img2[1]:
                    break
        # generate image pair from different class
        else:
            while True:
                img2 = self.rng.choice(self.dataset.imgs)
                if self.img1[1] != img2[1]:
                    break

        img1 = Image.open(self.img1[0])
        img2 = Image.open(img2[0])
        img1 = img1.convert('L')
        img2 = img2.convert('L')
        img1 = self.transform(img1)
        img2 = self.transform(img2)
        return img1, img2
