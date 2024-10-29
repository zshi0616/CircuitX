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

def get_args():
    parser = argparse.ArgumentParser(description='Parse AIG')
    parser.add_argument('--aig_dir', type=str, default='./data/sub_aig', help='AIG directory')
    parser.add_argument('--start_idx', type=int, default=0, help='Start index')
    parser.add_argument('--end_idx', type=int, default=100000, help='End index')
    parser.add_argument('--npz_path', type=str, default='', help='NPZ path')
    
    # Modes 
    parser.add_argument('--outlist', action='store_true', help='Output list of AIGs')
    args = parser.parse_args()
    
    # Output path 
    if args.npz_path == '':
        args.npz_path = './npz/aig_{:}_{:}.npz'.format(args.start_idx, args.end_idx)
    
    return args

if __name__ == '__main__':
    args = get_args()       # Get arguments
    # Check if AIG directory exists
    if args.outlist:
        aig_list = glob.glob(os.path.join(args.aig_dir, '*/*.aig'))
        f = open('./tmp/aig_list.txt', 'w')
        for aig_path in aig_list:
            f.write('{}\n'.format(aig_path))
        f.close()
        print('List saved to aig_list.txt')
        exit()
    else:
        f = open('./tmp/aig_list.txt', 'r')
        aig_list = f.readlines()
        f.close()
        aig_list = [aig.replace('\n', '') for aig in aig_list]
        no_aigs = min(args.end_idx - args.start_idx, len(aig_list))
    logger = Logger()       # Initialize logger
    
    # Main
    tot_time = 0
    graphs = {}
    for aig_k, aig_path in enumerate(aig_list):
        if aig_k < args.start_idx or aig_k >= args.end_idx:
            continue
        if not os.path.exists(aig_path):
            logger.write('File not found: {}'.format(aig_path))
            continue
        start_time = time.time()
        aig_name = os.path.basename(aig_path)[:-4]
        # Parse 
        x_data, edge_index = aiger_utils.aig_to_xdata(aig_path)
        logger.write('Parse: {} ({:} / {:}), Size: {:}, Time: {:.2f}s, ETA: {:.2f}s, Succ: {:}'.format(
            aig_name, aig_k, no_aigs, len(x_data), 
            tot_time, tot_time / ((aig_k + 1) / no_aigs) - tot_time, 
            len(graphs)
        ))
        fanin_list, fanoutlist = circuit_utils.get_fanin_fanout(x_data, edge_index)
        
        #############################################
        # Circuit features 
        #############################################
        edge_index = []
        for idx in range(len(x_data)):
            for fanin_idx in fanin_list[idx]:
                edge_index.append([fanin_idx, idx])
        x_data, edge_index = circuit_utils.remove_unconnected(x_data, edge_index)
        if len(edge_index) == 0 or len(x_data) == 0:
            logger.write('Empty: {}'.format(aig_name))
            continue
        x_one_hot = dg.construct_node_feature(x_data, 3)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        forward_level, forward_index, backward_level, backward_index = dg.return_order_info(edge_index, x_one_hot.size(0))
        
        graph = dataset_utils.OrderedData()
        graph.x = x_one_hot
        graph.edge_index = edge_index
        graph.name = aig_name
        graph.gate = torch.tensor(x_data[:, 1], dtype=torch.long).unsqueeze(1)
        graph.forward_index = forward_index
        graph.backward_index = backward_index
        graph.forward_level = forward_level
        graph.backward_level = backward_level
        graph.no_nodes = len(x_data)
        graph.no_edges = len(edge_index[0]) 
                       
        #############################################
        # Node-level features 
        #############################################
        prob, tt_pair_index, tt_sim, con_index, con_label = circuit_utils.prepare_dg2_labels_cpp(graph, 15000)
        
        assert max(prob).item() <= 1.0 and min(prob).item() >= 0.0
        assert len(prob) == len(x_data)
        if len(tt_pair_index) == 0:
            tt_pair_index = torch.zeros((2, 0), dtype=torch.long)
        else:
            tt_pair_index = tt_pair_index.t().contiguous()
        graph.prob = prob
        graph.tt_pair_index = tt_pair_index
        graph.tt_sim = tt_sim
        graph.connect_pair_index = con_index.T
        graph.connect_label = con_label
        
        
        end_time = time.time()
        tot_time += end_time - start_time
    
    np.savez_compressed(args.npz_path, circuits=graphs)
    logger.write(args.npz_path)
    logger.write(len(graphs))