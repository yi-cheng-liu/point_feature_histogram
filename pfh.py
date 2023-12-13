import numpy as np
import matplotlib.pyplot as plt
import utils

def pfh(pc, radius):
    pass

def main():
    # Import the cloud
    pc_source = utils.load_pc('data/pcl_data_csv/horse.csv')

    target = 'cat'
    pc_target_filename = f'cloud_icp_target{target}.csv'
    pc_target = utils.load_pc(pc_target_filename)
    
    fig1 = utils.view_pc([pc_source], None, ['b'])
    fig1.axes[0].set_xlim(-150, 200)
    fig1.axes[0].set_ylim(-150, 200)
    fig1.axes[0].set_zlim(-150, 200)

    plt.show()

if __name__ == '__main__':
    main()