import argparse
import glob
import os
import sys
import platform
import time
import numpy as np
import deepgate as dg
import torch

from utils.logger import Logger
import utils.aiger_utils as aiger_utils
import utils.circuit_utils as circuit_utils
import utils.dataset_utils as dataset_utils 

input_npz_path = './npz/test.npz'
output_dir = './npz/split_test'

if __name__ == '__main__': 
    print('Loading ... : {}'.format(input_npz_path))
    circuits = np.load(input_npz_path, allow_pickle=True)['circuits'].item()
    graphs = {}
    
    for cir_idx, cir_name in enumerate(circuits):
        ckt = circuits[cir_name]
        x = ckt["x"]
        edge_index = ckt["edge_index"]
        forward_level = ckt["forward_level"]
        backward_level = ckt["backward_level"]
        prob = ckt["prob"]
        gate = ckt["gate"]
        tt_pair_index = ckt["tt_pair_index"]
        tt_sim = ckt["tt_sim"]
        tt_dis = 1 - tt_sim
        
        graphs[cir_name] = {
            'x': x.numpy(),
            'edge_index': edge_index.numpy(),
            'forward_level': forward_level.numpy(),
            'backward_level': backward_level.numpy(),
            'prob': prob.numpy(),
            'gate': gate.numpy(),
            'tt_pair_index': tt_pair_index.numpy(),
            'tt_sim': tt_sim.numpy(),
            'tt_dis': tt_dis.numpy(), 
            'no_nodes': len(x),
            'no_edges': len(edge_index[0])  
        }
        if len(graphs) % 1000 == 0:
            print('Processed: {}'.format(len(graphs)))
    
    print('Total number of graphs: {}'.format(len(graphs)))
    # Divide the graphs into smaller chunks
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    ckt_id = 0
    npz_id = 0
    steps = 20000
    graph_keys = list(graphs.keys())
    while ckt_id < len(graphs):
        tmp_graphs = {}
        for i in range(ckt_id, min(ckt_id + steps, len(graphs))):
            tmp_graphs[graph_keys[i]] = graphs[graph_keys[i]]
        ckt_id += len(tmp_graphs)
        output_npz_path = os.path.join(output_dir, 'train_{}_{}.npz'.format(npz_id, len(tmp_graphs)))
        npz_id += 1
        np.savez(output_npz_path, circuits=tmp_graphs)
        print('Save: {}'.format(output_npz_path))
    print('Done')
    