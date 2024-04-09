import torch.nn as nn
import torch.nn.functional as F
from para_options import args_parser

args = args_parser()


class ModelFedCon(nn.Module):

    def __init__(self, base_model, out_dim, n_classes, dataset='SVHN'):
        super(ModelFedCon, self).__init__()
        if dataset == 'cifar100':
            self.features = SimpleCNN_h2(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=n_classes)
            num_ftrs = 84
        else:
            self.features = SimpleCNN_h1(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=n_classes)
            num_ftrs = 84

        # projection MLP
        self.l1 = nn.Linear(num_ftrs, num_ftrs)
        self.dropout = nn.Dropout(p=0.5)
        self.l2 = nn.Linear(num_ftrs, out_dim)

        # last layer
        self.l3 = nn.Linear(out_dim, n_classes)
        # self.l4 = nn.Linear(num_ftrs, n_classes)

    def _get_basemodel(self, model_name):
        try:
            model = self.model_dict[model_name]
            # print("Feature extractor:", model_name)
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward(self, x, model=None):
        h = self.features(x)
        # print("h before:", h)
        # print("h size:", h.size())
        if model == 'densenet':
            out = F.relu(h, inplace=True)
            h = F.adaptive_avg_pool2d(out, (1, 1)).view(h.size(0), -1)
        h = h.squeeze()
        if len(h.shape) == 1:
            h = h.unsqueeze(dim=0)
        # print("h after:", h)
        x = self.l1(h)
        x = F.relu(x)
        x = self.l2(x)

        y = self.l3(x)
        
        # z = self.l4(h)
        return h, x, y
        # base-encoder out, projection out, classifier out (logits)


class SimpleCNN_h1(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=10):
        super(SimpleCNN_h1, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        # self.fc3 = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.relu(x)
        x = self.pool(self.conv2(x))
        x = self.relu(x)
        x = x.view(-1, 16 * 5 * 5)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x
    
class SimpleCNN_h2(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=10):
        super(SimpleCNN_h2, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm1d(120)
        self.bn4 = nn.BatchNorm1d(84)
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        # self.fc3 = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.relu(x)
        x = self.bn1(x)
        x = self.pool(self.conv2(x))
        x = self.relu(x)
        x = self.bn2(x)
        x = x.view(-1, 16 * 5 * 5)
        x = self.relu(self.fc1(x))
        x = self.bn3(x)
        x = self.relu(self.fc2(x))
        x = self.bn4(x)
        return x