from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import deepgate
import torch
import os
import config

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
DATA_DIR = './data/lcm_train/'

if __name__ == '__main__':
    num_epochs = 60
    args = config.get_parse_args()
    train_npz_path = '../../npz/train'
    test_npz_path = '../../npz/test/test_0_16630.npz'
    
    print('[INFO] Parse Dataset')
    dataset = deepgate.NpzParser(DATA_DIR, train_npz_path, test_npz_path)
    train_dataset, val_dataset = dataset.get_dataset()
    print('[INFO] Create Model and Trainer')
    model = deepgate.Model()
    
    trainer = deepgate.Trainer(
        model, training_id='train_1104_dg2', 
        distributed=args.distributed, device='cuda:1', batch_size=args.batch_size
    )
    trainer.set_training_args(prob_rc_func_weight=[3.0, 1.0, 0.0], lr=1e-4, lr_step=50)
    print('[INFO] Stage 1 Training ...')
    trainer.train(num_epochs, train_dataset, val_dataset)
    
    print('[INFO] Stage 2 Training ...')
    trainer.set_training_args(prob_rc_func_weight=[3.0, 1.0, 2.0], lr=1e-4, lr_step=50)
    trainer.train(num_epochs, train_dataset, val_dataset)
    