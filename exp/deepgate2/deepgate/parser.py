from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Optional, Callable, List
import os.path as osp

import numpy as np 
import torch
import shutil
import os
import copy
import glob 
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader

from .utils.data_utils import read_npz_file
from .utils.aiger_utils import aig_to_xdata
from .utils.circuit_utils import get_fanin_fanout, read_file, add_node_index, feature_gen_connect
from .parser_func import *

class NpzParser():
    '''
        Parse the npz file into an inmemory torch_geometric.data.Data object
    '''
    def __init__(self, data_dir, train_npz_dir, test_npz_path, \
                 random_shuffle=True): 
        self.data_dir = data_dir
        if not os.path.exists(train_npz_dir):
            raise ValueError('The data directory does not exist.')
        
        if os.path.isdir(train_npz_dir):
            train_npz_list = []
            input_npz_list = glob.glob(os.path.join(train_npz_dir, '*.npz'))
            for npz_path in input_npz_list:
                train_npz_list.append(npz_path)
            all_train_dataset = self.inmemory_dataset(data_dir, train_npz_list, 'train')
        else:
            assert train_npz_dir.endswith('.npz'), 'The input file should be a npz file.'
            all_train_dataset = self.inmemory_dataset(data_dir, [train_npz_dir], 'train')
            
        if random_shuffle:
            perm = torch.randperm(len(all_train_dataset))
            all_train_dataset = all_train_dataset[perm]
        test_dataset = self.inmemory_dataset(data_dir, [test_npz_path], 'test')
                
        data_len = len(all_train_dataset)
        self.train_dataset = all_train_dataset
        self.val_dataset = test_dataset
        
        # self.train_dataset = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
        # self.val_dataset = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
    def get_dataset(self):
        return self.train_dataset, self.val_dataset
    
    class inmemory_dataset(InMemoryDataset):
        def __init__(self, root, circuit_path_list, inm_header, transform=None, pre_transform=None, pre_filter=None):
            self.name = 'npz_inmm_dataset'
            self.root = root
            self.circuit_path_list = circuit_path_list
            self.inm_header = inm_header
            super().__init__(root, transform, pre_transform, pre_filter)
            self.data, self.slices = torch.load(self.processed_paths[0])
        
        @property
        def raw_dir(self):
            return self.root

        @property
        def processed_dir(self):
            name = 'inmemory_{}'.format(self.inm_header)
            return osp.join(self.root, name)

        @property
        def raw_file_names(self) -> List[str]:
            return [self.circuit_path_list[0]]

        @property
        def processed_file_names(self) -> str:
            return ['data.pt']

        def download(self):
            pass
        
        def process(self):
            data_list = []
            
            for npz_k, npz_path in enumerate(self.circuit_path_list):
                
                circuits = read_npz_file(npz_path)['circuits'].item()
                
                for cir_idx, cir_name in enumerate(circuits):
                    print('[{:}/{:}] Parse circuit: {}, {:} / {:} = {:.2f}%'.format(
                        npz_k, len(self.circuit_path_list),
                        cir_name, cir_idx, len(circuits), cir_idx / len(circuits) * 100
                    ))
                    ckt = circuits[cir_name]
                    x = ckt["x"]
                    edge_index = ckt["edge_index"]

                    tt_dis = ckt['tt_dis']
                    tt_pair_index = ckt['tt_pair_index']
                    prob = ckt['prob']
                    
                    if 'rc_pair_index' not in ckt:
                        rc_pair_index = torch.zeros((0, 2), dtype=torch.long)
                        is_rc = []

                    # check the gate types
                    # assert (x[:, 1].max() == (len(self.args.gate_to_index)) - 1), 'The gate types are not consistent.'
                    graph = parse_pyg_mlpgate(
                        x, edge_index, tt_dis, tt_pair_index, 
                        prob, rc_pair_index, is_rc, 
                        ckt['forward_level'], ckt['backward_level']
                    )
                    graph.name = cir_name
                    data_list.append(graph)
                    
            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])
            print('[INFO] Inmemory dataset save: ', self.processed_paths[0])
            print('Total Circuits: {:}'.format(len(data_list)))

        def __repr__(self) -> str:
            return f'{self.name}({len(self)})'

class AigParser():
    def __init__(self):
        pass
    
    def read_aiger(self, aig_path):
        circuit_name = os.path.basename(aig_path).split('.')[0]
        # tmp_aag_path = os.path.join(self.tmp_dir, '{}.aag'.format(circuit_name))
        x_data, edge_index = aig_to_xdata(aig_path)
        # os.remove(tmp_aag_path)
        # Construct graph object 
        x_data = np.array(x_data)
        edge_index = np.array(edge_index)
        tt_dis = []
        tt_pair_index = []
        prob = [0] * len(x_data)
        rc_pair_index = []
        is_rc = []
        graph = parse_pyg_mlpgate(
            x_data, edge_index, tt_dis, tt_pair_index, prob, rc_pair_index, is_rc
        )
        graph.name = circuit_name
        graph.PIs = graph.forward_index[(graph['forward_level'] == 0) & (graph['backward_level'] != 0)]
        graph.POs = graph.backward_index[(graph['backward_level'] == 0) & (graph['forward_level'] != 0)]
        graph.no_connect = graph.forward_index[(graph['forward_level'] == 0) & (graph['backward_level'] == 0)]
        
        return graph        
        
class BenchParser():
    def __init__(self, gate_to_index={'PI': 0, 'AND': 1, 'NOT': 2}):
        self.gate_to_index = gate_to_index
        pass
    
    def read_bench(self, bench_path):
        circuit_name = os.path.basename(bench_path).split('.')[0]
        x_data = read_file(bench_path)
        x_data, num_nodes, _ = add_node_index(x_data)
        x_data, edge_index = feature_gen_connect(x_data, self.gate_to_index)
        for idx in range(len(x_data)):
            x_data[idx] = [idx, int(x_data[idx][1])]
        # os.remove(tmp_aag_path)
        # Construct graph object 
        x_data = np.array(x_data)
        edge_index = np.array(edge_index)
        tt_dis = []
        tt_pair_index = []
        prob = [0] * len(x_data)
        rc_pair_index = []
        is_rc = []
        graph = parse_pyg_mlpgate(
            x_data, edge_index, tt_dis, tt_pair_index, prob, rc_pair_index, is_rc
        )
        graph.name = circuit_name
        graph.PIs = graph.forward_index[(graph['forward_level'] == 0) & (graph['backward_level'] != 0)]
        graph.POs = graph.backward_index[(graph['backward_level'] == 0) & (graph['forward_level'] != 0)]
        graph.no_connect = graph.forward_index[(graph['forward_level'] == 0) & (graph['backward_level'] == 0)]
        
        return graph       
