# PDCFed
The code for paper "Two-stage Sampling with Predicted Distribution Changes in Federated Semi-supervised Learning"
## Preparation
1. Create conda environment:

		conda create -n PDCFed python=3.8
		conda activate PDCFed

2. Install dependencies:

		pip install -r requirements.txt
		
SVHN, CIFAR-100 and CIFAR-10 dataset will be downloaded automatically once training started.

## Parameters
Parameter     | Description
-------- | -----
dataset  | dataset used
model | backbone structure
unsup_num  | number of unlabeled clients
sup_bs | batch size of supervised client
unsup_bs | batch size of unsupervised client
lambda_u | ratio of loss on unlabeled clients
su_epoch | supervised client local epoch
base_lr | lr on labeled clients
unsup_lr | lr on unlabeled clients
max_grad_norm | limit maximum gradient
rounds | maximum global communication rounds
sampled_client | number of clients in each subset
w_mul_times | scaling times for labeled clients
tau_upper | lower bound for first stage sampling and upper bound for second stage sampling
tau_lower | lower bound for second stage sampling
sigma | the threshold for sample selection
gama | heterogeneity (Dirichlet partition)
time_current | file name

## Run the code
1. Train model for each dataset:

    sh run.sh

## Evaluation
1. SVHN
   
    python test.py --dataset=SVHN --model_path=pdc_noniid

2. cifar10
   
    python test.py --dataset=cifar10 --model_path=pdc_noniid
   
3. cifar100

    python test.py --dataset=cifar100 --model_path=pdc_noniid
