import torch 
import os 
import numpy as np
import utils.dataset_utils as dataset_utils

npz_list = [
    # './npz/train_0_30000.npz',
    # './npz/train_30000_60000.npz',
    # './npz/train_60000_100000.npz',
    './npz/train_0_20000.npz',
    './npz/train_20000_40000.npz',
    './npz/train_40000_100000.npz',
]

output_npz_path = './npz/train.npz'

if __name__ == '__main__':
    graphs = {}
    
    for npz_k, npz_path in enumerate(npz_list):
        print('Loading ... {}'.format(npz_path))
        data = np.load(npz_path, allow_pickle=True)['circuits'].item()
        for cir_name in data.keys():
            aig = data[cir_name]
            graphs[cir_name] = aig
            print('Loaded {} with {} nodes'.format(cir_name, aig.num_nodes))
    
    
    print('Total number of AIGs: {}'.format(len(graphs)))
    np.savez(output_npz_path, circuits=graphs)
    print('Saved to {}'.format(output_npz_path))

