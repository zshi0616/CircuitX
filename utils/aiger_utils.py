import aiger
import os

def aig_to_xdata(aig_filename, gate_to_index={'PI': 0, 'AND': 1, 'NOT': 2}):
    aig = aiger.load(aig_filename)
    f = str(aiger.BoolExpr(aig))
    lines = f.split('\n')
    header = lines[0].strip().split(" ")
    # “M”, “I”, “L”, “O”, “A” separated by spaces.
    n_variables = eval(header[1])
    n_inputs = eval(header[2])
    n_outputs = eval(header[4])
    n_and = eval(header[5])
    no_latch = eval(header[3])
    assert no_latch == 0, 'The AIG has latches.'
    # if n_outputs != 1 or n_variables != (n_inputs + n_and) or n_variables == n_inputs:
    #     return [], []
    # assert n_outputs == 1, 'The AIG has multiple outputs.'
    # assert n_variables == (n_inputs + n_and), 'There are unused AND gates.'
    # assert n_variables != n_inputs, '# variable equals to # inputs'
    # Construct AIG graph
    x_data = []
    edge_index = []
    # node_labels = []
    
    # PI 
    for i in range(n_inputs):
        x_data.append([len(x_data), gate_to_index['PI']])
    # AND 
    for i in range(n_and):
        x_data.append([len(x_data), gate_to_index['AND']])
    
    # AND Connections
    has_not = [-1] * (len(x_data) + 1)
    for (i, line) in enumerate(lines[1+n_inputs+n_outputs: ]):
        arr = line.replace('\n', '').split(' ')
        if len(arr) != 3:
            continue
        and_index = int(int(arr[0]) / 2) - 1
        fanin_1_index = int(int(arr[1]) / 2) - 1
        fanin_2_index = int(int(arr[2]) / 2) - 1
        fanin_1_not = int(arr[1]) % 2
        fanin_2_not = int(arr[2]) % 2
        if fanin_1_not == 1:
            if has_not[fanin_1_index] == -1:
                x_data.append([len(x_data), gate_to_index['NOT']])
                not_index = len(x_data) - 1
                edge_index.append([fanin_1_index, not_index])
                has_not[fanin_1_index] = not_index
            fanin_1_index = has_not[fanin_1_index]
        if fanin_2_not == 1:
            if has_not[fanin_2_index] == -1:
                x_data.append([len(x_data), gate_to_index['NOT']])
                not_index = len(x_data) - 1
                edge_index.append([fanin_2_index, not_index])
                has_not[fanin_2_index] = not_index
            fanin_2_index = has_not[fanin_2_index]
        edge_index.append([fanin_1_index, and_index])
        edge_index.append([fanin_2_index, and_index])

    # PO NOT check 
    for (i, line) in enumerate(lines[1+n_inputs: 1+n_inputs+n_outputs]):
        arr = line.replace('\n', '').split(' ')
        if len(arr) != 1:
            continue
        po_index = int(int(arr[0]) / 2) - 1
        if po_index < 0:
            continue
        po_not = int(arr[0]) % 2
        if po_not == 1:
            if has_not[po_index] == -1:
                x_data.append([len(x_data), gate_to_index['NOT']])
                not_index = len(x_data) - 1
                edge_index.append([po_index, not_index])
                has_not[po_index] = not_index

    return x_data, edge_index