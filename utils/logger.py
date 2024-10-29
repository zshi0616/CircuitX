from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import sys
import torch

class Logger(object):
    def __init__(self, log_dir = './log', log_name = ''):
        # MM-DD-HH-MM-SS-LOG_NAME.log
        log_path = os.path.join(log_dir, time.strftime('%m-%d-%H-%M-%S') + '-' + log_name + '.log')
        self.log = open(log_path, 'w')

    def write(self, txt):
        self.log.write(txt + '\n')
        print(txt)
        self.log.flush()
