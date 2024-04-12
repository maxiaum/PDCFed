import numpy as np
import torch
import os
import logging
import random
import copy
import datetime
import sys
import pandas as pd
from torchvision import transforms
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from para_options import args_parser
from local_update import SupervisedLocalUpdate, UnsupervisedLocalUpdate
from dataset import get_dataloader, partition_data
from FedAvg import FedAvg
from evaluate import test
from network import ModelFedCon


if __name__ == '__main__':
    args = args_parser()
    
    sup_user_id = [0]
    unsup_user_id = list(range(len(sup_user_id), len(sup_user_id) + args.unsup_num)) 
    sup_num = len(sup_user_id)
    unsup_num = len(unsup_user_id)
    total_num = sup_num + unsup_num
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    if args.log_file_name is None:
        args.log_file_name = 'log-%s' % (datetime.datetime.now().strftime("%m-%d-%H%M-%S"))
    log_path = args.log_file_name + '.log'
    logging.basicConfig(filename=os.path.join(args.logdir, log_path), level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.info(str(args))
    logger.info(args.time_current)
    
    # set_seed
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    #set tensoboard path
    
    if not os.path.isdir('tensorboard/'):
        os.mkdir('tensorboard/')
        
    if args.dataset == 'SVHN':
        if not os.path.isdir('tensorboard/SVHN/' + args.time_current):
            os.mkdir('tensorboard/SVHN/' + args.time_current)
        writer = SummaryWriter('tensorboard/SVHN/' + args.time_current)
    
    elif args.dataset == 'cifar10':
        if not os.path.isdir('tensorboard/cifar10/' + args.time_current):
            os.mkdir('tensorboard/cifar10/' + args.time_current)
        writer = SummaryWriter('tensorboard/cifar10/' + args.time_current)
    
    elif args.dataset == 'cifar100':
        if not os.path.isdir('tensorboard/cifar100/'+ args.time_current):
            os.mkdir('tensorboard/cifar100/' + args.time_current)
        writer = SummaryWriter('tensorboard/cifar100/' + args.time_current)
    
    #set saved model path
        
    if not os.path.isdir('model/'):
        os.mkdir('model/')
    shot_path = 'model/'
    
    if args.dataset == 'SVHN':
        if not os.path.isdir('model/SVHN/'):
            os.mkdir('model/SVHN/')
        shot_path = shot_path + 'SVHN/'
    
    elif args.dataset == 'cifar10':
        if not os.path.isdir('model/cifar10/'):
            os.mkdir('model/cifar10/')
        shot_path = shot_path + 'cifar10/'
            
    elif args.dataset == 'cifar100':
        if not os.path.isdir('model/cifar100/'):
            os.mkdir('model/cifar100/') 
        shot_path = shot_path + 'cifar100/'  
        
    print('==> Reloading data partitioning strategy..')
    
    assert os.path.isdir('partition_strategy/'), 'Error: no partition_strategy directory found!'
    
    # data partition
    if args.dataset == 'SVHN':
        partition = torch.load('partition_strategy/SVHN_noniid_Training_10%labeled.pth')
        val_partition = torch.load('partition_strategy/SVHN_noniid_Validation.pth')
    
    elif args.dataset == 'cifar10':
        partition = torch.load('partition_strategy/cifar10_noniid_Training_beta0.8.pth')
        val_partition = torch.load('partition_strategy/cifar10_noniid_Validation.pth')
    
    elif args.dataset == 'cifar100':
        partition = torch.load('partition_strategy/cifar100_noniid_Training_10%labeled.pth')
        val_partition = torch.load('partition_strategy/cifar100_noniid_Validation.pth')
    
    net_dataidx_map = partition['data_partition']
    val_data_map = val_partition['val_data']
    
    if args.partition == 'iid':
        X_train, y_train, X_test, y_test, net_dataidx_map, val_data_map = partition_data(args,
        args.dataset, args.datadir, partition=args.partition, n_parties=total_num, gama=args.gama)
    
    else:
        X_train, y_train, X_test, y_test, _, _ = partition_data(args,
        args.dataset, args.datadir, partition=args.partition, n_parties=total_num, gama=args.gama)
    
    each_lenth = [len(net_dataidx_map[i]) for i in range(len(net_dataidx_map))]
    total_lenth = sum(each_lenth)
    
    if args.dataset == 'SVHN':
        X_train = X_train.transpose([0, 2, 3, 1])
        X_test = X_test.transpose([0, 2, 3, 1])
        
    if args.dataset == 'SVHN' or args.dataset == 'cifar10':
        n_classes = 10
    
    elif args.dataset == 'cifar100':
        n_classes = 100
        
    net_glob = ModelFedCon(args.model, args.out_dim, n_classes=n_classes, dataset=args.dataset)
    if len(args.gpu.split(',')) > 1:
        use_gpu = args.gpu.split(',')
        print(use_gpu)
        net_glob = torch.nn.DataParallel(net_glob, device_ids=[int(i) for _, i in enumerate(use_gpu)])
    
    net_glob.train()
    w_glob = net_glob.state_dict()
    
    sup_trainer_locals = []
    unsup_trainer_locals = []
    sup_net_locals = []
    unsup_net_locals = []
    sup_optim_locals =[]
    unsup_optim_locals = []
    
    for i in sup_user_id:
        sup_trainer_locals.append(SupervisedLocalUpdate(args, net_dataidx_map[i], n_classes))
        sup_net_locals.append(copy.deepcopy(net_glob))
        
        optimizer = torch.optim.SGD(sup_net_locals[i].parameters(), lr=args.base_lr, momentum=0.9, weight_decay=5e-4)
        sup_optim_locals.append(copy.deepcopy(optimizer.state_dict()))
    
    for i in unsup_user_id:
        unsup_trainer_locals.append(UnsupervisedLocalUpdate(args, net_dataidx_map[i], n_classes))
        unsup_net_locals.append(copy.deepcopy(net_glob))
        
        optimizer = torch.optim.SGD(unsup_net_locals[i - sup_num].parameters(), lr=args.unsup_lr, momentum=0.9, weight_decay=5e-4)
        unsup_optim_locals.append(copy.deepcopy(optimizer.state_dict()))
    
    curr_result, curr_round = 0.0, 0
    
    for com_round in range(args.rounds):
        print("************* Comm round %d begins *************" % com_round)
        
        loss_locals = []
        clt_this_round = []
        w_locals_this_round = []
        curr_pseu_ratio = []
        sampling_ratio = []
        chosen_sup = []
        
        clt_this_round = random.sample(list(range(0, args.num_users)), args.sampled_client)
        logger.info(f'Comm round {com_round} chosen client: {clt_this_round}')
        for client_idx in clt_this_round:
            if client_idx in sup_user_id:
                # supervised client local update
                chosen_sup.append(client_idx)
                local = sup_trainer_locals[client_idx]
                optimizer = sup_optim_locals[client_idx]
                train_dl_local, train_ds_local = get_dataloader(args, X_train[net_dataidx_map[client_idx]],
                                                                y_train[net_dataidx_map[client_idx]],
                                                                args.dataset, args.datadir, args.sup_bs,
                                                                is_labeled=True,
                                                                data_idxs=net_dataidx_map[client_idx],
                                                                pre_sz=args.pre_sz, input_sz=args.input_sz)
                w, loss, op = local.train(args, sup_net_locals[client_idx].state_dict(), optimizer,
                                            train_dl_local, n_classes, com_round)
                writer.add_scalar('Supervised loss on sup client %d' % client_idx, loss, global_step=com_round)
                w_locals_this_round.append(copy.deepcopy(w))
                loss_locals.append(copy.deepcopy(loss))
                sup_optim_locals[client_idx] = copy.deepcopy(op)
                logger.info(
                    'Labeled client {} sample num: {} training loss : {} lr : {}'.format(client_idx,
                                                                                            len(train_ds_local),
                                                                                            loss,
                                                                                            sup_optim_locals[
                                                                                                client_idx][
                                                                                                'param_groups'][0][
                                                                                                'lr']))
            else:
                # unsuperised client local update
                local = unsup_trainer_locals[client_idx - sup_num]
                optimizer = unsup_optim_locals[client_idx - sup_num]
                train_dl_local, train_ds_local = get_dataloader(args, X_train[net_dataidx_map[client_idx]],
                                                                y_train[net_dataidx_map[client_idx]],
                                                                args.dataset, args.datadir, args.unsup_bs,
                                                                is_labeled=False,
                                                                data_idxs=net_dataidx_map[client_idx],
                                                                pre_sz=args.pre_sz, input_sz=args.input_sz)
                w, loss, op = local.train(args,
                    unsup_net_locals[client_idx - sup_num].state_dict(),
                    optimizer,
                    client_idx,
                    train_dl_local, n_classes, com_round, len(train_ds_local))
                writer.add_scalar('Unsupervised loss on unsup client %d' % client_idx, loss, global_step=com_round)
                w_locals_this_round.append(copy.deepcopy(w))
                loss_locals.append(copy.deepcopy(loss))
                unsup_optim_locals[client_idx - sup_num] = copy.deepcopy(op) 
                logger.info(
                    'Unlabeled client {} sample num: {} Training loss: {}, lr {}'.format(
                        client_idx, len(train_ds_local), loss,
                        unsup_optim_locals[
                            client_idx - sup_num][
                            'param_groups'][
                            0]['lr'],
                    ))
        each_lenth_this_round = [each_lenth[i] for i in clt_this_round]
        
        # sup client weight adjustment
        if args.w_mul_times != 1 and len(chosen_sup) != 0:
            for idx in chosen_sup:
                each_lenth_this_round[clt_this_round.index(idx)] *= args.w_mul_times
        
        total_lenth_this_round = sum(each_lenth_this_round)
        freq_this_round = [length / total_lenth_this_round for length in each_lenth_this_round]
        
        with torch.no_grad():
            w_glob = FedAvg(w_locals_this_round, freq_this_round)
            
        net_glob.load_state_dict(w_glob)    
        
        for idx in sup_user_id:
            sup_net_locals[idx] = copy.deepcopy(net_glob)
            
        for idx in unsup_user_id:
            unsup_net_locals[idx - sup_num] = copy.deepcopy(net_glob)
        
        loss_avg = sum(loss_locals) / len(loss_locals)
            
        # save tensorboard file
        assert args.st != 0, 'Error: the divisor is invalidate!'
        if com_round % args.st == 0:
            if not os.path.isdir(shot_path + args.time_current):
                os.mkdir(shot_path + args.time_current)
            save_mode_path = os.path.join(shot_path + args.time_current, 'epoch_' + str(com_round) + '.pth')
            if len(args.gpu) != 1:
                torch.save({
                    'state_dict': net_glob.module.state_dict(),
                    'sup_optimizers': sup_optim_locals,
                    'unsup_optimizers': unsup_optim_locals,
                    'start_epoch': com_round
                }
                    , save_mode_path
                )
            else:
                torch.save({
                    'state_dict': net_glob.state_dict(),
                    'sup_optimizers': sup_optim_locals,
                    'unsup_optimizers': unsup_optim_locals,
                    'start_epoch': com_round
                }
                    , save_mode_path
                )
        
        #test model
        AUROC_avg, Accus_avg = test(args, net_glob.state_dict(), X_train[val_data_map], y_train[val_data_map], n_classes)
        
        # select the best model
        weighted_acc_auc = args.elpi * Accus_avg + (1 - args.elpi) * AUROC_avg
        if com_round > 0 and weighted_acc_auc >= curr_result:
            curr_model = copy.deepcopy(w_glob)
            curr_result = round(weighted_acc_auc, 4)
            curr_round = com_round
        if com_round + 1 == args.rounds:
            if not os.path.isdir('final_model/' + args.dataset + '/'):
                os.mkdir('final_model/' + args.dataset + '/')
            torch.save({'state_dict': curr_model, "epoch": curr_round}, 'final_model/{}/{}.pth'.format(args.dataset, args.time_current))
            
        writer.add_scalar('AUC', AUROC_avg, global_step=com_round)
        writer.add_scalar('Acc', Accus_avg, global_step=com_round)
        # writer.add_scalar('sample_ratio', round(sample_avg, 4), global_step=com_round)
        # writer.add_scalar('pseu_ratio', round(pseu_avg, 4), global_step=com_round)
        logger.info("\nTEST AUROC: {:6f}, TEST Accus: {:6f}"
                    .format(AUROC_avg, Accus_avg))
