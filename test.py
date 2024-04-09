CUDA_VISIBLE_DEVICES="4"
from evaluate import test, epochVal_metrics_test
import numpy as np
import torch.backends.cudnn as cudnn
import random
from dataset import get_dataloader, partition_data
import copy
import torch
import torch.optim
import torch.nn.functional as F
from para_options import args_parser
from network import ModelFedCon
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
# import json

args = args_parser()
save_mode_path='final_model/' + args.dataset + '/' + args.model_path + '.pth'

if __name__ == "__main__":
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    X_train, y_train, X_test, y_test, _, _ = partition_data(args,
        args.dataset, args.datadir, partition=args.partition, n_parties=args.num_users, gama=args.gama)

    if args.dataset == 'SVHN':
        X_train = X_train.transpose([0, 2, 3, 1])
        X_test = X_test.transpose([0, 2, 3, 1])

    if args.dataset == 'SVHN' or args.dataset == 'cifar10':
        n_classes = 10
    
    elif args.dataset == 'cifar100':
        n_classes = 100

    # all_client = np.zeros([n_classes, n_classes], dtype=int)
    # traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)

    checkpoint_path = save_mode_path
    checkpoint = torch.load(checkpoint_path)

    net = ModelFedCon(args.model, args.out_dim, n_classes=n_classes, dataset=args.dataset)
    model = net.cuda()
    model.load_state_dict(checkpoint['state_dict'])
    print(checkpoint['epoch'])            

    test_dl, test_ds = get_dataloader(args, X_test, y_test,
                                        args.dataset, args.datadir, args.sup_bs,
                                        is_labeled=True, is_testing=True)

    AUROCs, Accus = epochVal_metrics_test(model, test_dl,args.model, n_classes=n_classes)
    AUROC_avg = np.array(AUROCs).mean()
    Accus_avg = np.array(Accus).mean()
    print(Accus_avg, AUROC_avg)