import numpy as np
import matplotlib.pyplot as plt
import utils
import time

# Done
def random_trans_matrix():
    """ Generate random transformation matrix for target point """
    # Rotational matrix
    roll = np.random.rand() * np.pi
    pitch = np.random.rand() * np.pi
    yaw = np.random.rand() * np.pi
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    rot_mat = Rz @ Ry @ Rx
    # Translation matrix
    translation_mat = np.random.rand(3).reshape((3, ))

    transformation_mat = np.eye(4)
    transformation_mat[:3, :3] = rot_mat
    transformation_mat[:3, 3] = translation_mat

    return transformation_mat
# Done
def find_rigid_transformation(P_np, Q_np):
    """ Helper function for finding the tranformation matrix between two point cloud lists"""
    # Compute means and centered vectors
    p_mean = np.mean(P_np, axis=1)
    q_mean = np.mean(Q_np, axis=1)

    X = P_np - p_mean
    Y = Q_np - q_mean

    # Compute matrix S
    S = np.dot(X, Y.T)

    # SVD
    U, _, Vt = np.linalg.svd(S)
    V = Vt.T

    # Correct for reflection
    M = np.diag([1, 1, np.linalg.det(np.dot(V, U.T))])

    # Compute R and t
    R = np.dot(V, np.dot(M, U.T))
    t = q_mean - np.dot(R, p_mean)

    return R, t
# Done
def calculate_normal_vec(pc, k=20):
    normals = []

    for i, point in enumerate(pc):
        # calculate distance 
        distances = np.linalg.norm(pc - point, axis=1)

        # Find k nearest neighbors
        idx = np.argsort(distances)[1 : k + 1]
        neighbors = pc[idx[1:]]  # Exclude the point itself from its neighbors

        # Perform PCA to find the normal vector
        cov_mat = np.cov(neighbors, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_mat)
        normal = eigenvectors[:, np.argmin(eigenvalues)]

        # Ensure the normal is outward
        if np.dot(normal, point - np.mean(neighbors, axis=0)) < 0:
            normal = -normal

        normals.append(normal)

    # print("Normals shape: ",np.array(normals).shape)

    return np.array(normals)  # Transpose back to match the original pc shape
# Done
def find_closest_histogram(hist_source, hists_target):
    dists = np.linalg.norm(hists_target - hist_source, axis=1)
    closest_idx = np.argmin(dists)

    return closest_idx

def get_neighbors_radius(point, pc, radius=20):
    neighbor_indices = []
    for i, other_point in enumerate(pc):
        if other_point.all() != point.all() and np.linalg.norm(point - other_point) < radius:
            neighbor_indices.append(i)
    return neighbor_indices

def calculate_feature(center, neighbor, center_normal, neighbor_normal):
    u = center_normal
    v = np.cross(u, (neighbor - center) / np.linalg.norm(neighbor - center))
    w = np.cross(u, v)

    alpha = np.dot(v, neighbor_normal)
    phi = np.dot(u, (neighbor - center) / np.linalg.norm(neighbor - center))
    theta = np.arctan2(np.dot(w, neighbor_normal), np.dot(u, neighbor_normal))
    return np.array([alpha, phi, theta])

def get_bin_index(feature, bins_number):

    # normalize the fuature
    normalized_feature = feature / (2 * np.pi)

    # calculate the bin index
    bin_index = normalized_feature * bins_number

    int_bin_index = np.array([int(i) for i in bin_index])

    return int_bin_index



def pfh(pc, normals, bins_number=125, radius=20):
    """
    pc: point cloud, (3400, 3) np.array
    return: closest list
    """
    points_count = pc.shape[0]
    histograms = []
    
    for i in range(points_count):
        # Find neighbor
        neighbor_indices = get_neighbors_radius(pc[i], pc, radius)
        histogram = np.zeros([bins_number])

        # Feature calculation
        for j in neighbor_indices:
            if i != j:
                feature = calculate_feature(pc[i], pc[j], normals[i], normals[j])

                # Histogram bin
                bin_index = get_bin_index(feature, bins_number)
                histogram[bin_index] += 1

        histograms.append(histogram)

    return np.array(histograms)

def icp_pfh(pc_source, pc_target):
    # Compute centroid of source and target
    iter = 0
    finished = False
    epsilon = 0.01
    errors = []
    
    pc_target_matrix = utils.convert_pc_to_matrix(pc_target)     # (3, 3400) matrix
    pc_target_array = np.asarray(pc_target_matrix.T)             # (3400, 3) array
    pc_target_normals = calculate_normal_vec(pc_target_array)
    target_histograms = pfh(pc_target_array, pc_target_normals)  # (3400, 125)
    
    while not finished:
        if iter == 100:
            finished = True
            return P, errors
        
        print(f"Iteration: {iter}")
        pc_source_matrix = utils.convert_pc_to_matrix(pc_source)     # (3, 3400) matrix
        pc_source_array = np.asarray(pc_source_matrix.T)             # (3400, 3) array
        
        # Compute Correspondences
        P = [] # source point
        Q = [] # closest point in target

        # Calculate normal vector
        normal_vec_start = time.time()
        pc_source_normals = calculate_normal_vec(pc_source_array)
        normal_vec_duration = time.time() - normal_vec_start
        print(f"Normal Vector time: {normal_vec_duration:.5f} sec. ")

        # source and target histogram
        pfh_start = time.time()
        score_histograms = pfh(pc_source_array, pc_source_normals)  # (3400, 125)
        pfh_duration = time.time() - pfh_start
        print(f"Point Feature Histogram time: {pfh_duration:.5f} sec. ")

        # print(f'pfh done')

        find_closest_start = time.time()
        for i, point in enumerate(pc_source):
            # compare histogram, and find closest
            closest_index = find_closest_histogram(score_histograms[i], target_histograms)
            q_np = pc_target_array[closest_index]
            q_np = q_np.reshape((3, 1)).T
            q = utils.convert_matrix_to_pc(q_np)  # this is a list of 3 floats
            
            P.append(point)
            Q.append(q)
        find_closest_duration = time.time() - find_closest_start
        print(f"Find Closest Histogram time: {find_closest_duration:.5f} sec. ")
        
        P_np = utils.convert_pc_to_matrix(P)
        Q_np = utils.convert_pc_to_matrix(Q)
        R, t = find_rigid_transformation(P_np, Q_np)

        dist = (np.dot(R, pc_source_matrix) + t) - pc_target_matrix
        error = np.sum(np.linalg.norm(dist, axis=0))
        errors.append(error)
        if error < epsilon:
            finished = True
            return P, errors
        
        # Update all P
        for i in range(len(P)):
            pc_source[i] = np.dot(R, pc_source_matrix[:, i]) + t
            
        iter += 1
            
    return pc_source, errors

def main():
    # Import the source point cloud
    pc_source = utils.load_pc('source/pcl_data_csv/horse.csv')
    pc_source_matrix = utils.convert_pc_to_matrix(pc_source)
    
    # Transform point cloud to get target point cloud
    M = random_trans_matrix()
    pc_source_np_homogenous = np.vstack((pc_source_matrix, np.ones((1, pc_source_matrix.shape[1]))))
    pc_target_matrix = np.dot(M, pc_source_np_homogenous)[:3, :] # (3, 3400) matrix
    pc_target = utils.convert_matrix_to_pc(pc_target_matrix)
    
    # Iterate with Point Feature Histogram
    start = time.time()

    icp_pfh(pc_source, pc_target)
    duration = time.time() - start
    print(f"Total time of Point Feature Histogram is {duration:.5f} sec. ")

    #########################    Visualization    #########################
    # View point cloud 
    fig1 = utils.view_pc([pc_source, pc_target], None, ['b', 'r'], ['o', '^'])
    fig1.axes[0].set_xlim(-150, 200)
    fig1.axes[0].set_ylim(-150, 200)
    fig1.axes[0].set_zlim(-150, 200)
    
    # # Plot normals
    # for i in range(len(pc_source)):
    #     point = pc_source_array[i]
    #     normal = source_normals[i]
    #     fig1.axes[0].quiver(point[0], point[1], point[2], normal[0], normal[1], normal[2], length=10, color='r')

    plt.show()

    #######################################################################

if __name__ == '__main__':
    main()