import os 
import random

train_test_ratio = 0.8

if __name__ == '__main__':
    f = open('./tmp/aig_list.txt', 'r')
    aig_list = f.readlines()
    f.close()
    aig_list = [aig.replace('\n', '') for aig in aig_list]
    
    train_path = './tmp/aig_train_list.txt'
    test_path = './tmp/aig_test_list.txt'
    
    train_flag = [0] * len(aig_list)
    for i in range(int(len(aig_list) * train_test_ratio)):
        train_flag[i] = 1
    random.shuffle(train_flag)
    
    f_train = open(train_path, 'w')
    f_test = open(test_path, 'w')
    for i, aig_path in enumerate(aig_list):
        if train_flag[i] == 1:
            f_train.write('{}\n'.format(aig_path))
        else:
            f_test.write('{}\n'.format(aig_path))
    f_train.close()
    f_test.close()