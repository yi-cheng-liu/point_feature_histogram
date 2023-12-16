import time
import numpy as np
import matplotlib.pyplot as plt
import utils
from icp_template import icp
from pfh import icp_pfh, random_trans_matrix

# Include your ICP and PFH function definitions here

def compare_icp_pfh(pc_source, pc_target):
    # Measure ICP
    start_time_icp = time.time()
    transformed_icp, errors_icp = icp(pc_source, pc_target)  # Assuming icp() returns the transformed point cloud and errors
    duration_icp = time.time() - start_time_icp
    
    # ICP result
    fig1 = utils.view_pc([transformed_icp, pc_target], None, ['b', 'r'], ['o', '^'])
    plt.title("ICP Result")
    # plt.axis([-0.15, 0.15, -0.15, 0.15, -0.15, 0.15])

    # Measure PFH
    start_time_pfh = time.time()
    transformed_pfh, errors_pfh = icp_pfh(pc_source, pc_target, 100, 3000, 400)
    duration_pfh = time.time() - start_time_pfh
    
    # PFH result
    fig1 = utils.view_pc([transformed_pfh, pc_target], None, ['b', 'r'], ['o', '^'])
    plt.title("PFH Result")
    # plt.axis([-0, 0.15, -0.05, 0.1, -0.1, 0.05])

    # Compare and possibly visualize results
    print(f"ICP Time: {duration_icp} seconds, Final Error: {errors_icp[-1]}")
    print(f"PFH Time: {duration_pfh} seconds, Final Error: {errors_pfh[-1]}")

    # Plot error vs. iteration    
    plt.figure()
    plt.plot(errors_icp, label='ICP Error')
    plt.plot(errors_pfh, label='PFH Error')
    plt.title("Error vs. Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Error")
    plt.grid(True)
    plt.legend()
    plt.show()

def main():
    # Mug
    # pc_source = utils.load_pc('source/course_data/cloud_icp_source.csv')

    # target = 3
    # pc_target_filename = f'source/course_data/cloud_icp_target{target}.csv'
    # pc_target = utils.load_pc(pc_target_filename)
    
    # Horse
    pc_source = utils.load_pc('source/pcl_data_csv/horse.csv')
    pc_source_matrix = utils.convert_pc_to_matrix(pc_source)
    M = random_trans_matrix()
    pc_source_np_homogenous = np.vstack((pc_source_matrix, np.ones((1, pc_source_matrix.shape[1]))))
    pc_target_matrix = np.dot(M, pc_source_np_homogenous)[:3, :]
    pc_target = utils.convert_matrix_to_pc(pc_target_matrix)
    
    # Initial state
    fig1 = utils.view_pc([pc_source, pc_target], None, ['b', 'r'], ['o', '^'])
    plt.title("Initial Point Clouds")
    # plt.axis([-0.15, 0.15, -0.15, 0.15, -0.15, 0.15])

    compare_icp_pfh(pc_source, pc_target)

if __name__ == '__main__':
    main()
