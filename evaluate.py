from network import ModelFedCon
from dataset import get_dataloader
import torch
import numpy as np
from torch.nn import functional as F
from utils import  compute_metrics_test

def epochVal_metrics_test(model, dataLoader, model_type, n_classes):
    training = model.training
    model.eval()

    gt = torch.FloatTensor().cuda()
    pred = torch.FloatTensor().cuda()
    
    gt_study   = {}
    pred_study = {}
    studies    = []

    with torch.no_grad():
        for i, (study, image, label) in enumerate(dataLoader):
            image, label = image.cuda(), label.cuda()
            _, feature, output = model(image, model=model_type)
            study=study.tolist()
            output = F.softmax(output, dim=1)

            for i in range(len(study)):
                if study[i] in pred_study:
                    assert torch.equal(gt_study[study[i]], label[i])
                    pred_study[study[i]] = torch.max(pred_study[study[i]], output[i])
                else:
                    gt_study[study[i]] = label[i]
                    pred_study[study[i]] = output[i]
                    studies.append(study[i]) 
        
        for study in studies:
            gt = torch.cat((gt, gt_study[study].view(1, -1)), 0)
            pred = torch.cat((pred, pred_study[study].view(1, -1)), 0)
        #gt=F.one_hot(gt.to(torch.int64).squeeze())
        #AUROCs, Accus, Senss, Specs, pre, F1 = compute_metrics_test(gt, pred,  thresh=thresh, competition=True)
        AUROCs, Accus = compute_metrics_test(gt, pred,n_classes=n_classes)

    model.train(training)

    return AUROCs, Accus

def test(args, checkpoint, data_test, label_test, n_classes):
    net = ModelFedCon(args.model, args.out_dim, n_classes=n_classes, dataset=args.dataset)
    
    if len(args.gpu.split(',')) > 1:
        use_gpu = args.gpu.split(',')
        net = torch.nn.DataParallel(net, device_ids=[int(i) for _, i in enumerate(use_gpu)])
        
    model = net.cuda()
    model.load_state_dict(checkpoint)

    val_dl, _ = get_dataloader(args, data_test, label_test, args.dataset, args.datadir, args.sup_bs,
                                        is_labeled=True, is_testing=True)
    

    AUROCs, Accus = epochVal_metrics_test(model, val_dl, args.model, n_classes=n_classes)
    AUROC_avg = np.array(AUROCs).mean()
    Accus_avg = np.array(Accus).mean()

    return AUROC_avg, Accus_avg