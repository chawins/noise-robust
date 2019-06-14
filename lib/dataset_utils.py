'''
Dataset and DataLoader adapted from
https://www.kaggle.com/pinocookie/pytorch-dataset-and-dataloader
'''

import pickle

import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np


def rotate_img(img, rot):
    if rot == 0:  # 0 degrees rotation
        return img
    elif rot == 90:  # 90 degrees rotation
        return np.flipud(np.transpose(img, (1, 0, 2)))
    elif rot == 180:  # 90 degrees rotation
        return np.fliplr(np.flipud(img))
    elif rot == 270:  # 270 degrees rotation / or -90
        return np.transpose(np.flipud(img), (1, 0, 2))
    else:
        raise ValueError('rotation should be 0, 90, 180, or 270 degrees')


class RotateDataset(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]
        rotated_imgs = [
            self.transform(img),
            self.transform(rotate_img(img, 90).copy()),
            self.transform(rotate_img(img, 180).copy()),
            self.transform(rotate_img(img, 270).copy())
        ]
        rotation_labels = torch.LongTensor([0, 1, 2, 3])
        return torch.stack(rotated_imgs, dim=0), rotation_labels


def load_mnist(batch_size,
               data_dir='./data',
               val_size=0.1,
               shuffle=True,
               seed=1):
    """Load MNIST data into train/val/test data loader"""

    num_workers = 4

    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = load_mnist_all(
        data_dir=data_dir, val_size=val_size, shuffle=shuffle, seed=seed)

    trainset = torch.utils.data.TensorDataset(x_train, y_train)
    validset = torch.utils.data.TensorDataset(x_valid, y_valid)
    testset = torch.utils.data.TensorDataset(x_test, y_test)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    validloader = torch.utils.data.DataLoader(
        validset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, validloader, testloader


def load_mnist_all(data_dir='./data', val_size=0.1, shuffle=True, seed=1):
    """Load entire MNIST dataset into tensor"""

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset), shuffle=False)

    x, y = next(iter(trainloader))
    x_test, y_test = next(iter(testloader))

    x_train, x_valid, y_train, y_valid = train_test_split(
        x.numpy(), y.numpy(), test_size=val_size, shuffle=shuffle,
        random_state=seed, stratify=y)

    # scale up
    # scale = 2
    # x_train = x_train.repeat(scale, axis=2).repeat(scale, axis=3)
    # x_valid = x_valid.repeat(scale, axis=2).repeat(scale, axis=3)
    # x_test = x_test.numpy().repeat(scale, axis=2).repeat(scale, axis=3)
    # x_test = torch.tensor(x_test)

    return ((torch.tensor(x_train), torch.tensor(y_train)),
            (torch.tensor(x_valid), torch.tensor(y_valid)), (x_test, y_test))


def load_mnist_rot(batch_size, data_dir='./data', val_size=0.1, shuffle=True,
                   seed=1):

    (x_train, _), (x_valid, _), (x_test, _) = load_mnist_all(
        data_dir, val_size=val_size, seed=seed)

    traindataset = RotateDataset(x_train.numpy().transpose(0, 2, 3, 1))
    trainloader = torch.utils.data.DataLoader(
        traindataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)

    validdataset = RotateDataset(x_valid.numpy().transpose(0, 2, 3, 1))
    validloader = torch.utils.data.DataLoader(
        validdataset, batch_size=batch_size, shuffle=False, num_workers=4)

    testdataset = RotateDataset(x_test.numpy().transpose(0, 2, 3, 1))
    testloader = torch.utils.data.DataLoader(
        testdataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return trainloader, validloader, testloader


def load_cifar10(batch_size,
                 data_dir='./data',
                 val_size=0.1,
                 normalize=True,
                 augment=True,
                 shuffle=True,
                 seed=1):
    """Load CIFAR-10 data into train/val/test data loader"""

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    num_workers = 4

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    if augment:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(
                5, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
            transforms.ColorJitter(brightness=0.1),
            transforms.ToTensor()
        ])
    else:
        transform_train = transform

    if normalize:
        transform = transforms.Compose([
            transform,
            transforms.Normalize(mean, std)
        ])
        transform_train = transforms.Compose([
            transform_train,
            transforms.Normalize(mean, std)
        ])

    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train)
    validset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform)

    # Random split train and validation sets
    num_train = len(trainset)
    indices = list(range(num_train))
    split = int(np.floor(val_size * num_train))

    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers)
    validloader = torch.utils.data.DataLoader(
        validset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, validloader, testloader


def load_cifar10_all(data_dir='./data', val_size=0.1, shuffle=True, seed=1):
    """Load entire CIFAR-10 dataset into tensor"""

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform)
    validset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform)

    # Random split train and validation sets
    num_train = len(trainset)
    indices = list(range(num_train))
    split = int(np.floor(val_size * num_train))

    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=(num_train - split), sampler=train_sampler)
    validloader = torch.utils.data.DataLoader(
        validset, batch_size=split, sampler=valid_sampler)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset), shuffle=False)

    x_train = next(iter(trainloader))
    x_valid = next(iter(validloader))
    x_test = next(iter(testloader))

    return x_train, x_valid, x_test


def load_cifar10_noise(batch_size, data_dir='./data', val_size=0.1, sd=0,
                       shuffle=True, seed=1):

    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = load_cifar10_all(
        data_dir, val_size=val_size, seed=seed)

    x_train += torch.randn_like(x_train) * sd

    trainset = torch.utils.data.TensorDataset(x_train, y_train)
    validset = torch.utils.data.TensorDataset(x_valid, y_valid)
    testset = torch.utils.data.TensorDataset(x_test, y_test)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
    validloader = torch.utils.data.DataLoader(
        validset, batch_size=batch_size, shuffle=False, num_workers=4)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=4)

    return trainloader, validloader, testloader


def load_cifar10_rot(batch_size, data_dir='./data', val_size=0.1, shuffle=True,
                     seed=1):

    (x_train, _), (x_valid, _), (x_test, _) = load_cifar10_all(
        data_dir, val_size=val_size, seed=seed)

    traindataset = RotateDataset(x_train.numpy().transpose(0, 2, 3, 1))
    trainloader = torch.utils.data.DataLoader(
        traindataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)

    validdataset = RotateDataset(x_valid.numpy().transpose(0, 2, 3, 1))
    validloader = torch.utils.data.DataLoader(
        validdataset, batch_size=batch_size, shuffle=False, num_workers=4)

    testdataset = RotateDataset(x_test.numpy().transpose(0, 2, 3, 1))
    testloader = torch.utils.data.DataLoader(
        testdataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return trainloader, validloader, testloader


def load_gtsrb(data_dir='./data', gray=False, train_file_name=None):
    """
    Load GTSRB data as a (datasize) x (channels) x (height) x (width) numpy
    matrix. Each pixel is rescaled to lie in [0,1].
    """

    def load_pickled_data(file, columns):
        """
        Loads pickled training and test data.

        Parameters
        ----------
        file    : string
                          Name of the pickle file.
        columns : list of strings
                          List of columns in pickled data we're interested in.

        Returns
        -------
        A tuple of datasets for given columns.
        """

        with open(file, mode='rb') as f:
            dataset = pickle.load(f)
        return tuple(map(lambda c: dataset[c], columns))

    def preprocess(x, gray):
        """
        Preprocess dataset: turn images into grayscale if specified, normalize
        input space to [0,1], reshape array to appropriate shape for NN model
        """

        if not gray:
            # Scale features to be in [0, 1]
            x = (x / 255.).astype(np.float32)
        else:
            # Convert to grayscale, e.g. single Y channel
            x = 0.299 * x[:, :, :, 0] + 0.587 * x[:, :, :, 1] + \
                0.114 * x[:, :, :, 2]
            # Scale features to be in [0, 1]
            x = (x / 255.).astype(np.float32)
            x = x[:, :, :, np.newaxis]
        return x

    # Load pickle dataset
    if train_file_name is None:
        x_train, y_train = load_pickled_data(
            data_dir + 'train.p', ['features', 'labels'])
    else:
        x_train, y_train = load_pickled_data(
            data_dir + train_file_name, ['features', 'labels'])
    x_val, y_val = load_pickled_data(
        data_dir + 'valid.p', ['features', 'labels'])
    x_test, y_test = load_pickled_data(
        data_dir + 'test.p', ['features', 'labels'])

    # Preprocess loaded data
    x_train = preprocess(x_train, gray)
    x_val = preprocess(x_val, gray)
    x_test = preprocess(x_test, gray)
    return x_train, y_train, x_val, y_val, x_test, y_test


class GtsrbDataset(torch.utils.data.Dataset):

    def __init__(self, x_np, y_np, mean=None, std=None, augment=False):

        self.x_pil = [Image.fromarray(
            (x * 255).astype(np.uint8)) for x in x_np]
        self.y_np = y_np.astype(np.int64)

        if mean is None:
            mean = (0, 0, 0)
            std = (1, 1, 1)

        if augment:
            self.transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4, padding_mode='edge'),
                transforms.RandomAffine(
                    5, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
                transforms.ColorJitter(brightness=0.1),
                transforms.ToTensor(),
                # transforms.Normalize(mean, std),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize(mean, std),
            ])

    def __getitem__(self, index):
        # apply the transformations and return tensors
        return self.transform(self.x_pil[index]), self.y_np[index]

    def __len__(self):
        return len(self.x_pil)


def load_gtsrb_dataloader(data_dir, batch_size, num_workers=4):

    x_train, y_train, x_val, y_val, x_test, y_test = load_gtsrb(
        data_dir=data_dir)

    # Standardization
    mean = np.mean(x_train, (0, 1, 2))
    std = np.std(x_train, (0, 1, 2))

    trainset = GtsrbDataset(x_train, y_train, mean, std, augment=True)
    validset = GtsrbDataset(x_val, y_val, mean, std, augment=False)
    testset = GtsrbDataset(x_test, y_test, mean, std, augment=False)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    validloader = torch.utils.data.DataLoader(
        validset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, validloader, testloader


def create_planes(d=1000, k=10, num_total=10000, bound=(0, 1), test_size=0.2,
                  val_size=0.1, seed=1):
    """
    Create plane dataset: two planes with dimension k in space of dimension d.
    The first k dimensions are random numbers within the bound, dimensions
    k + 1 to d - 1 are 0, and d-th dimension is bound[0] or bound[1] which
    determines the class.
    """

    assert bound[0] < bound[1]

    np.random.seed(seed)

    planes = torch.zeros((num_total, d))
    planes[:, :k] = torch.rand(num_total, k) * (bound[1] - bound[0]) + bound[0]
    # planes[:num_total // 2, -1] = bound[0]
    # planes[num_total // 2:, -1] = bound[1]
    planes[:num_total // 2, -1] = 0.3
    planes[num_total // 2:, -1] = 0.7

    indices = np.arange(num_total)
    np.random.shuffle(indices)

    train_idx = int(num_total * (1 - test_size - val_size))
    test_idx = int(num_total * (1 - test_size))
    x_train = planes[indices[:train_idx]]
    x_valid = planes[indices[train_idx:test_idx]]
    x_test = planes[indices[test_idx:]]

    y_train = torch.tensor(
        (indices[:train_idx] >= num_total // 2).astype(np.int64))
    y_valid = torch.tensor(
        (indices[train_idx:test_idx] >= num_total // 2).astype(np.int64))
    y_test = torch.tensor(
        (indices[test_idx:] >= num_total // 2).astype(np.int64))

    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)


def load_planes(batch_size, d=1000, k=10, num_total=10000, bound=(0, 1),
                test_size=0.2, val_size=0.1, shuffle=True, seed=1):

    num_workers = 4

    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = create_planes(
        d=d, k=k, num_total=num_total, bound=bound, test_size=test_size,
        val_size=val_size, seed=seed)

    trainset = torch.utils.data.TensorDataset(x_train, y_train)
    validset = torch.utils.data.TensorDataset(x_valid, y_valid)
    testset = torch.utils.data.TensorDataset(x_test, y_test)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    validloader = torch.utils.data.DataLoader(
        validset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, validloader, testloader
