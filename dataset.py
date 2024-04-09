import numpy as np
import torch
from torchvision.transforms import transforms, autoaugment
import torch.utils.data as data
import logging
from torchvision import datasets, transforms
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# from datasets import CIFAR10_truncated, SVHN_truncated, CIFAR100_truncated
import pandas as pd
from utils import CheXpertDataset, TransformTwice


def load_cifar10_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    cifar10_train_ds = datasets.CIFAR10(datadir, train=True, download=True, transform=transform)
    cifar10_test_ds = datasets.CIFAR10(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar10_train_ds.data, np.array(cifar10_train_ds.targets)
    X_test, y_test = cifar10_test_ds.data, np.array(cifar10_test_ds.targets)

    return (X_train, y_train, X_test, y_test)


def load_cifar100_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    cifar100_train_ds = datasets.CIFAR100(datadir, train=True, download=True, transform=transform)
    cifar100_test_ds = datasets.CIFAR100(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar100_train_ds.data, np.array(cifar100_train_ds.targets)
    X_test, y_test = cifar100_test_ds.data, np.array(cifar100_test_ds.targets)

    return (X_train, y_train, X_test, y_test)


def load_SVHN_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    SVHN_train_ds = datasets.SVHN(datadir, split='train', download=True, transform=transform)
    SVHN_test_ds = datasets.SVHN(datadir, split='test', download=True, transform=transform)

    X_train, y_train = SVHN_train_ds.data, np.array(SVHN_train_ds.labels)
    X_test, y_test = SVHN_test_ds.data, np.array(SVHN_test_ds.labels)

    return (X_train, y_train, X_test, y_test)


def partition_data(args, dataset, datadir, partition='noniid', n_parties=10, n_classes=10, gama=0.4):
    if dataset == 'cifar10':
        X_train, y_train, X_test, y_test = load_cifar10_data(datadir)
    elif dataset == 'SVHN':
        X_train, y_train, X_test, y_test = load_SVHN_data(datadir)
    elif dataset == 'cifar100':
        X_train, y_train, X_test, y_test = load_cifar100_data(datadir)
        
    n_train = y_train.shape[0]
    val_idxs = np.random.permutation(n_train)
    val_data_map = np.random.choice(range(len(val_idxs)), 15000, replace=False)
    train_idx = list(set(val_idxs) - set(val_data_map))
    if partition == "homo" or partition == "iid":
        if args.diff_labeled:
            net_dataidx_map = {}
            # idx = np.random.permutation(n_train)
            net_dataidx_map[0] = np.random.choice(range(len(train_idx)), args.labeled_nums, replace=False)
            idx = list(set(train_idx) - set(net_dataidx_map[0]))
            batch_idxs = np.array_split(idx, n_parties - 1)
            for u in range(1,n_parties):
                net_dataidx_map[u] = batch_idxs[u-1]
        else:
            # idxs = np.random.permutation(n_train)
            batch_idxs = np.array_split(train_idx, n_parties)
            net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}

    elif partition == "noniid-labeldir" or partition == "noniid":
        min_size = 0
        min_require_size = 10
        K = 10

        N = len(train_idx)
        # print(N)
        net_dataidx_map = {}

        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_parties)]
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                remaining_idx = np.setdiff1d(idx_k, val_data_map)
                # print(len(remaining_idx))
                np.random.shuffle(remaining_idx)
                proportions = np.random.dirichlet(np.repeat(gama, n_parties))
                proportions = np.array(
                    [p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(remaining_idx)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(remaining_idx, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_parties):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    return X_train, y_train, X_test, y_test, net_dataidx_map, val_data_map
        
    
def get_dataloader(args, data_np, label_np, dataset_type, datadir, train_bs, is_labeled=None, data_idxs=None,
                   is_testing=False, pre_sz=40, input_sz=32):
    if dataset_type == 'SVHN':
        normalize = transforms.Normalize(mean=[0.4376821, 0.4437697, 0.47280442],
                                         std=[0.19803012, 0.20101562, 0.19703614])
        assert pre_sz == 40 and input_sz == 32, 'Error: Wrong input size for 32*32 dataset'
        
    elif dataset_type == 'cifar100':
        normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                         std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
        assert pre_sz == 40 and input_sz == 32, 'Error: Wrong input size for 32*32 dataset'
        
    elif dataset_type == 'cifar10':
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        assert pre_sz == 40 and input_sz == 32, 'Error: Wrong input size for 32*32 dataset'

    if not is_testing:
        if is_labeled:
            trans = transforms.Compose(
                [transforms.RandomCrop(size=(input_sz, input_sz)),
                 transforms.RandomHorizontalFlip(p=0.5),
                 transforms.ToTensor(),
                 normalize
                 ])
            ds = CheXpertDataset(dataset_type, data_np, label_np, pre_sz, pre_sz, lab_trans=trans,
                                         is_labeled=True, is_testing=False)
        else:
            # weak augmentation
            weak_trans = transforms.Compose([
                transforms.RandomCrop(size=(input_sz, input_sz)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                normalize
            ])
            # strong augmentations
            strong_trans = transforms.Compose([
                transforms.RandomResizedCrop(size=(input_sz, input_sz)),
                transforms.RandomHorizontalFlip(p=0.5),
                autoaugment.RandAugment(),
                transforms.ToTensor(),
                transforms.ConvertImageDtype(torch.float),
                normalize
            ])
            ds = CheXpertDataset(dataset_type, data_np, label_np, pre_sz, pre_sz,
                                         un_trans=TransformTwice(weak_trans, strong_trans),
                                         data_idxs=data_idxs,
                                         is_labeled=False,
                                         is_testing=False)
        dl = data.DataLoader(dataset=ds, batch_size=train_bs, drop_last=False, shuffle=True, num_workers=0)
    else:
        ds = CheXpertDataset(dataset_type, data_np, label_np, input_sz, input_sz, lab_trans=transforms.Compose([
            transforms.ToTensor(),
            normalize
        ]), is_labeled=True, is_testing=True)
        dl = data.DataLoader(dataset=ds, batch_size=train_bs, drop_last=False, shuffle=False, num_workers=0)
    return dl, ds