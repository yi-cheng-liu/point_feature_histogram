#!/usr/bin/env python
import utils
import numpy
import matplotlib.pyplot as plt
###YOUR IMPORTS HERE###
import numpy as np

def main():
    #Import the cloud
    pc_source = utils.load_pc('cat.csv')

    # target = 3
    # pc_target_filename = f'cloud_icp_target{target}.csv'
    # pc_target = utils.load_pc(pc_target_filename) # Change this to load in a different target
    
    fig1 = utils.view_pc([pc_source], None, ['b'])
    fig1.axes[0].set_xlim(-150, 200)
    fig1.axes[0].set_ylim(-150, 200)
    fig1.axes[0].set_zlim(-150, 200)
    plt.title("Initial Point Clouds")

    plt.show()
    # raw_input("Press enter to end:")


if __name__ == '__main__':
    main()