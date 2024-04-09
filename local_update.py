import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from torchvision import transforms
from para_options import args_parser
from scipy.special import entr
from network import ModelFedCon

args = args_parser()

class SupervisedLocalUpdate(object):
    def __init__(self, args, idxs, n_classes):
        self.base_lr = args.base_lr
        self.data_idx = idxs
        self.max_grad_norm = args.max_grad_norm

        net = ModelFedCon(args.model, args.out_dim, n_classes=n_classes, dataset=args.dataset)
        if len(args.gpu.split(',')) > 1:
            use_gpu = args.gpu.split(',')
            net = torch.nn.DataParallel(net, device_ids=[int(i) for _, i in enumerate(use_gpu)])
        self.model = net.cuda()

    def train(self, args, net_w, op_dict, dataloader, n_classes, round):
        self.model.load_state_dict(copy.deepcopy(net_w))
        self.model.cuda().train()
        # l_rate = args.base_lr * np.exp(- round / (2.0 * args.rounds))
        
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                             lr=args.base_lr, momentum=0.9,
                                             weight_decay=5e-4)
        
        param_collector = copy.deepcopy(list(self.model.parameters()))
        self.optimizer.load_state_dict(op_dict)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.base_lr

        ce_loss = torch.nn.CrossEntropyLoss()
        
        epoch_loss = []
        
        logging.info('Begin supervised training')
        
        for e in range(args.su_epoch):
            batch_loss = []
            for i, (_, image_batch, label_batch) in enumerate(dataloader):

                image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
                label_batch = label_batch.long().squeeze()
                inputs = image_batch
                _, activations, outputs = self.model(inputs, model=args.model)
                
                if len(label_batch.shape) == 0:
                    label_batch = label_batch.unsqueeze(dim=0)
                if len(outputs.shape) != 2:
                    outputs = outputs.unsqueeze(dim=0)
                
                self.optimizer.zero_grad()

                loss = ce_loss(outputs, label_batch)
                
                loss.backward()
                    
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               max_norm=self.max_grad_norm)
                self.optimizer.step()
                
                batch_loss.append(loss.item())

            epoch_loss.append(np.array(batch_loss).mean())

        self.model.cpu()
        return self.model.state_dict(), sum(epoch_loss) / len(epoch_loss), copy.deepcopy(
            self.optimizer.state_dict())
    
class UnsupervisedLocalUpdate(object):
    def __init__(self, args, idxs, n_classes):
        net = ModelFedCon(args.model, args.out_dim, n_classes=n_classes, dataset=args.dataset)
        if len(args.gpu.split(',')) > 1:
            use_gpu = args.gpu.split(',')
            net = torch.nn.DataParallel(net, device_ids=[int(i) for _, i in enumerate(use_gpu)])
        self.model = net.cuda()
        
        self.data_idxs = idxs
        self.unsup_lr = args.unsup_lr
        self.max_grad_norm = args.max_grad_norm
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.prob_max_mu_t = 1 / n_classes
        self.prob_max_var_t = 1.0
        self.ema_beta = 0.999
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.unsup_lr, momentum=0.9,
                                             weight_decay=5e-4)
    
    def update(self, freq):
        prob_max_mu_t = torch.mean(freq)
        prob_max_var_t = torch.var(freq, unbiased=True)
        self.prob_max_mu_t = self.ema_beta * self.prob_max_mu_t + (1.0 - self.ema_beta) * prob_max_mu_t
        self.prob_max_var_t = self.ema_beta * self.prob_max_var_t + (1.0 - self.ema_beta) * prob_max_var_t
        
    def calculate_frequent(self, freq):
        mu = self.prob_max_mu_t
        var = self.prob_max_var_t
        div = torch.where(freq <= mu, torch.full_like(freq, args.L_divisor), torch.full_like(freq, args.R_divisor))
        frequent = args.lambda_max * torch.exp(- (freq - mu) ** 2 / (2 * var / div))
        return frequent
    
    def train(self, args, net_w, op_dict, unlabeled_idx, train_dl_local, n_classes, r, samples):
        self.model.load_state_dict(copy.deepcopy(net_w))
        
        self.model.train()
        self.model.cuda()
        self.optimizer.load_state_dict(op_dict)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.unsup_lr
        # self.epoch = epoch

        epoch_loss = []
        
        logging.info('Unlabeled client %d begin unsupervised training' % unlabeled_idx)
        
        for e in range(args.local_ep):
            batch_loss = []
            for i, (_, aug_batch, label_batch) in enumerate(train_dl_local):
                aug_batch = [aug_batch[version].cuda() for version in range(len(aug_batch))]
                
                label = label_batch.squeeze().cuda()
                if len(label.shape) == 0:
                    label = label.unsqueeze(dim=0)
                # print(len(label))
                # with torch.no_grad():
                _, pro_weak, weak_logits = self.model(aug_batch[0], model=args.model)
                probs_weak = F.softmax(weak_logits, dim=1)
                max_probs, target_u = torch.max(probs_weak, dim=-1)
                    
                mask = max_probs.ge(args.tau_upper).float()
                mask_2 = max_probs.lt(args.tau_upper).float()
                mask_3 = max_probs.gt(args.tau_lower).float()
                    
                _, pro_strong, strong_logits = self.model(aug_batch[1], model=args.model)
           
                probs_strong = F.softmax(strong_logits, dim=1)
                probs_dist = F.softmax(strong_logits / args.temper, dim=1)

                increased = probs_strong.gt(probs_weak).float()
                frequent = torch.sum(increased, dim=1) / n_classes
                
                self.update(frequent)
                Freq = self.calculate_frequent(frequent)
                
                # dist_q = torch.sigmoid(- torch.cosine_similarity(pro_weak, pro_strong, dim=1))
                dist_q = torch.norm(probs_weak - probs_strong, dim=1)

                weight = torch.exp(- args.lambda_scale * dist_q / Freq)

                mask_4 = weight.ge(args.sigma).float()
                
                # sample_nums =  sample_nums + torch.sum(mask, dim=-1) + torch.sum((mask_2 * mask_3 * mask_4).float(), dim=-1)
                # pseu1 = torch.sum(label.eq(target_u).float() * mask, dim=-1)
                # pseu2 = torch.sum(label.eq(target_u).float() * mask_2 * mask_3 * mask_4, dim=-1)
                # pseu_nums = pseu_nums + pseu1 + pseu2
                
                Lu_1 = (self.ce_loss(probs_strong, target_u) * mask).mean()
                Lu_2 = (self.ce_loss(probs_strong, target_u) * mask_2 * mask_3 * mask_4).mean()
    
                L_e = args.lambda_e * (- torch.sum(probs_dist * F.log_softmax(strong_logits / args.temper, dim=1), dim=1) * mask_3 * mask_4).mean()
                
                loss_u = args.lambda_u * (Lu_1 + Lu_2)
        
                self.optimizer.zero_grad()

                (loss_u + L_e).backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               max_norm=self.max_grad_norm)
                self.optimizer.step()

                batch_loss.append(loss_u.item())

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        
        self.model.cpu()
        # print(pesu_nums, sample_nums, samples)
        return self.model.state_dict(), sum(epoch_loss) / len(epoch_loss), copy.deepcopy(
            self.optimizer.state_dict())