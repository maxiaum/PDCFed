# encoding: utf-8
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from para_options import args_parser
# from sklearn.metrics._ranking import roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  # , sensitivity_score
from imblearn.metrics import sensitivity_score, specificity_score
import pdb
from sklearn.metrics._ranking import roc_auc_score

N_CLASSES = 10

args = args_parser()

def compute_metrics_test(gt, pred, n_classes=10):
    """
    Computes accuracy, precision, recall and F1-score from prediction scores.
    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.
        competition: whether to use competition tasks. If False,
          use all tasks
    Returns:
        List of AUROCs of all classes.
    """

    gt_np = gt.cpu().detach().numpy()
    pred_np = pred.cpu().detach().numpy()
    indexes = range(n_classes)

    AUROCs = roc_auc_score(gt_np, pred_np, multi_class='ovr')
    Accus = accuracy_score(gt_np, np.argmax(pred_np, axis=1))
    # Pre = precision_score(gt_np, np.argmax(pred_np, axis=1), average='macro')
    # Recall = recall_score(gt_np, np.argmax(pred_np, axis=1), average='macro')
    return AUROCs, Accus  # , Senss, Specs, Pre, F1


def compute_pred_matrix(gt, pred, n_classes):
    matrix = np.zeros([n_classes, n_classes])
    for idx_gt in range(len(gt)):
        matrix[int(gt[idx_gt])][pred[idx_gt]] += 1
    return matrix

class CheXpertDataset(Dataset):
    def __init__(self, dataset_type, data_np, label_np, pre_w, pre_h, lab_trans=None, un_trans=None, data_idxs=None,
                 is_labeled=False,
                 is_testing=False):
        """
        Args:
            data_dir: path to image directory.
            csv_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        super(CheXpertDataset, self).__init__()

        self.images = data_np
        self.labels = label_np
        self.is_labeled = is_labeled
        self.dataset_type = dataset_type
        self.is_testing = is_testing

        self.resize = transforms.Compose([transforms.Resize((pre_w, pre_h))])
        if not is_testing:
            if is_labeled == True:
                self.transform = lab_trans
            else:
                self.data_idxs = data_idxs
                self.weak_trans = un_trans
        else:
            self.transform = lab_trans

        print('Total # images:{}, labels:{}'.format(len(self.images), len(self.labels)))

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        if self.dataset_type == 'skin':
            img_path = self.images[index]
            image = Image.open(img_path).convert('RGB')
        else:
            image = Image.fromarray(self.images[index]).convert('RGB')

        image_resized = self.resize(image)
        label = self.labels[index]

        if not self.is_testing:
            if self.is_labeled == True:
                if self.transform is not None:
                    image = self.transform(image_resized).squeeze()
                    # image=image[:,:224,:224]
                    return index, image, torch.FloatTensor([label])
            else:
                if self.weak_trans and self.data_idxs is not None:
                    weak_aug = self.weak_trans(image_resized)
                    idx_in_all = self.data_idxs[index]

                    for idx in range(len(weak_aug)):
                        weak_aug[idx] = weak_aug[idx].squeeze()
                    return index, weak_aug, torch.FloatTensor([label])
        else:
            image = self.transform(image_resized)
            return index, image, torch.FloatTensor([label])
            # return index, weak_aug, strong_aug, torch.FloatTensor([label])

    def __len__(self):
        return len(self.labels)

class TransformTwice:
    def __init__(self, transform1, transform2):
        self.transform1 = transform1
        self.transform2 = transform2

    def __call__(self, inp):
        out1 = self.transform1(inp)
        out2 = self.transform2(inp)
        
        return [out1, out2]