import os; import sys; 
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(ROOT_DIR)

import torch
import time

from models.ik_network import IKNet
from ik_net.settings import config
from utils.rotations import axis_angle_to_quaternion



def main():     
    model = IKNet(config).cuda()
    model.eval()

    x = torch.randn(2, 21, 3).cuda()
    y = torch.randn(2, 6).cuda()

    while True:
        total_time = 0
        for i in range(100):
            c = time.time() 
            out = model(x, y)
            total_time += time.time() - c
        
        avg_time = total_time / 100
        print()

        
        print(round(1 / (avg_time)), out.keys())

        # print(out['hms'].shape, out['d3d'].shape)

        # global_orient = out['global_orient']

        # axis_angle_to_quaternion(global_orient)
        # print(global_orient.shape)

        

if __name__ == '__main__':
    main()