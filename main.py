import numpy as np
import matplotlib.pyplot as plt
import utils
import time
from pfh import PFH

def main():
    # Import the source point cloud
    pc_source = utils.load_pc('source/pcl_data_csv/horse.csv')
    pc_source_original = utils.load_pc('source/pcl_data_csv/horse.csv')
    pc_source_matrix = utils.convert_pc_to_matrix(pc_source)
    
    # Transform point cloud to get target point cloud
    # M = random_trans_matrix()
    M = [
        [-0.11398497,  0.43457496,  0.89339355,  0.11882465],
        [-0.63446844,  0.6601462,  -0.40206567,  0.62941794],
        [-0.76449803, -0.61265945,  0.20047735,  0.16695979],
        [ 0.,          0.,          0.,          1.        ]
    ]
    M = np.array(M)
    pc_source_np_homogenous = np.vstack((pc_source_matrix, np.ones((1, pc_source_matrix.shape[1]))))
    pc_target_matrix = np.dot(M, pc_source_np_homogenous)[:3, :] # (3, 3400) matrix
    pc_target = utils.convert_matrix_to_pc(pc_target_matrix)
    
    # Create PFH instance for source point cloud
    pfh_source = PFH(pc_source)

    # Create PFH instance for target point cloud
    pfh_target = PFH(pc_target)

    # set parameter
    max_iteration = 10
    epsilon = 30000
    num_selected = 400
    
    # Perform ICP using PFH
    start = time.time()
    pc_source_transformed, _ = pfh_source.icp_pfh(pc_source, pc_target, max_iteration, epsilon, num_selected)
    duration = time.time() - start
    print(f"Total time of Point Feature Histogram ICP is {duration:.5f} sec.")

    #########################    Visualization    #########################
    # View point cloud 
    fig1 = utils.view_pc([pc_source_original, pc_target, pc_source_transformed], None, ['b', 'r', 'g'], ['o', '^', 'v'])
    fig1.axes[0].set_xlim(-150, 200)
    fig1.axes[0].set_ylim(-150, 200)
    fig1.axes[0].set_zlim(-150, 200)
    labels = ['Source Original', 'Target', 'Source Transformed']
    fig1.legend(labels, loc='upper right')
    plt.title("Horse Result")
    
    # # Plot normals
    # for i in range(len(pc_source)):
    #     point = pc_source_array[i]
    #     normal = source_normals[i]
    #     fig1.axes[0].quiver(point[0], point[1], point[2], normal[0], normal[1], normal[2], length=10, color='r')

    plt.show()

    #######################################################################

if __name__ == '__main__':
    main()