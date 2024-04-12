#!/bin/bash


# Training on the SVHN dataset under non-iid condition

python main.py --dataset=SVHN \
--model=simple-cnn \
--unsup_num=9 \
--lambda_u=0.05 \
--opt=sgd \
--base_lr=0.03 \
--unsup_lr=0.021 \
--sampled_client=5 \
--rounds=1000 \
--w_mul_times=6 \
--tau_upper=0.8 \
--tau_lower=0.6 \
--sigma=0.4 \
--su_epoch=3 \
--lambda_e=1e-3 \
--seed=1337 \
--time_current=pdc_noniid

# Training on the cifar10 dataset under non-iid condition

python main.py --dataset=cifar10 \
--model=simple-cnn \
--unsup_num=9 \
--lambda_u=0.027 \
--opt=sgd \
--base_lr=0.01 \
--unsup_lr=0.007 \
--sampled_client=5 \
--rounds=500 \
--w_mul_times=6 \
--tau_upper=0.85 \
--tau_lower=0.6 \
--sigma=0.6 \
--su_epoch=2 \
--lambda_e=1e-3 \
--seed=1337 \
--time_current=pdc_noniid

# Training on the cifar100 dataset under non-iid condition

python main.py --dataset=cifar100 \
--model=simple-cnn \
--unsup_num=9 \
--lambda_u=0.027 \
--opt=sgd \
--base_lr=0.003 \
--unsup_lr=0.001 \
--sup_bs=64 \
--unsup_bs=64 \
--sampled_client=5 \
--rounds=500 \
--w_mul_times=6 \
--tau_upper=0.85 \
--tau_lower=0.6 \
--sigma=0.6 \
--su_epoch=2 \
--lambda_e=1e-4 \
--seed=1337 \
--time_current=pdc_noniid

