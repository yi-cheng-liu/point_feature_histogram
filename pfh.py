import numpy as np
import matplotlib.pyplot as plt
import utils
import time

class PFH:
    def __init__(self, pc, bins_number=125, radius=5):
        """
        Initialize the PFH object.

        :param pc: The point cloud data, assumed to be an (N, 3) numpy array.
        :param normals: The normals for each point in the point cloud, also an (N, 3) numpy array.
        :param bins_number: The number of bins to use for the histogram.
        :param radius: The radius within which to search for neighboring points.
        """
        self.pc = pc
        self.pc_matrix = utils.convert_pc_to_matrix(pc)
        # self.normals = self.calculate_normal_vec(pc)
        self.bins_number = bins_number
        self.radius = radius
    
    def random_trans_matrix(self):
        """ Generate random transformation matrix for target point """
        # Rotational matrix
        roll = np.random.rand() * 2 * np.pi
        pitch = np.random.rand() * 2 * np.pi
        yaw = np.random.rand() * 2 * np.pi
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

    def find_rigid_transformation(self, P_np, Q_np):
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

    def calculate_normal_vec(self, pc, k=20):
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

        return np.array(normals)  # Transpose back to match the original pc shape

    def find_closest_histogram(self, hist_source, hists_target):
        # find the point idx which has the most silimar histogram
        dists = np.linalg.norm(hists_target - hist_source, axis=1)
        closest_idx = np.argmin(dists)

        return closest_idx

    def get_neighbors_radius(self, point, pc, radius=30):
        # get neighbors based on radius
        neighbor_indices = []
        for i, other_point in enumerate(pc):
            if np.linalg.norm(point - other_point) < radius:
                neighbor_indices.append(i)
        return neighbor_indices

    def calculate_feature(self, center, neighbor, center_normal, neighbor_normal):
        # claculate feature used for histogram
        u = center_normal
        v = np.cross(u, (neighbor - center) / np.linalg.norm(neighbor - center))
        w = np.cross(u, v)

        alpha = np.dot(v, neighbor_normal)
        phi = np.dot(u, (neighbor - center) / np.linalg.norm(neighbor - center))
        theta = np.arctan2(np.dot(w, neighbor_normal), np.dot(u, neighbor_normal))
        return np.array([alpha, phi, theta])

    def get_bin_index(self, feature, bins_number):
        # normalize the fuature
        normalized_feature = feature / (2 * np.pi)

        # calculate the bin index
        bin_index = normalized_feature * bins_number
        int_bin_index = np.array([int(i) for i in bin_index])

        return int_bin_index

    def pfh(self, pc, normals, selected_points, bins_number=125, radius=5):
        """
        pc: point cloud, (3400, 3) np.array
        return: closest list
        """
        histograms = np.zeros((len(pc), bins_number))
        
        for idx in selected_points:
            # Find neighbor
            neighbor_indices = self.get_neighbors_radius(pc[idx], pc, radius)
            histogram = np.zeros([bins_number])

            # Feature calculation
            for j in neighbor_indices:
                for k in neighbor_indices:
                    if j != k:
                # if idx != j:
                        feature = self.calculate_feature(pc[j], pc[k], normals[j], normals[k])

                        # Histogram bin
                        bin_index = self.get_bin_index(feature, bins_number)
                        histogram[bin_index] += 1

            histograms[idx] = histogram

        return histograms

    def icp_pfh(self, pc_source, pc_target, max_iteration, epsilon, num_selected):
        # Compute centroid of source and target
        iter = 0
        finished = False
        errors = []
        
        pc_target_matrix = utils.convert_pc_to_matrix(pc_target)     # (3, 3400) matrix
        pc_target_array = np.asarray(pc_target_matrix.T)             # (3400, 3) array
        pc_target_normals = self.calculate_normal_vec(pc_target_array)
        target_histograms = self.pfh(pc_target_array, pc_target_normals, np.arange(len(pc_source)))  # (3400, 125)
        
        while not finished:
            # set max iterations
            if iter == max_iteration:
                finished = True
                return pc_source, errors
            
            print(f"Iteration: {iter}")
            pc_source_matrix = utils.convert_pc_to_matrix(pc_source)     # (3, 3400) matrix
            pc_source_array = np.asarray(pc_source_matrix.T)             # (3400, 3) array
            
            # Compute Correspondences
            P = [] # source point
            Q = [] # closest point in target

            # Calculate normal vector
            normal_vec_start = time.time()
            pc_source_normals = self.calculate_normal_vec(pc_source_array)
            normal_vec_duration = time.time() - normal_vec_start
            print(f"Normal Vector time: {normal_vec_duration:.5f} sec. ")

            # randomly selected points for histogram to reduce computing time
            selected_points = np.random.choice(len(pc_source), num_selected, replace=False)

            # source histogram
            start = time.time()
            score_histograms = self.pfh(pc_source_array, pc_source_normals, selected_points) # (3400, 125)
            find_closest_start = time.time()
            print(f'PFH time: {find_closest_start - start}')

            # iterate seleted points rather than all points
            for i in selected_points:
                # compare histogram, and find closest
                closest_index = self.find_closest_histogram(score_histograms[i], target_histograms)
                q_np = pc_target_array[closest_index]
                q_np = q_np.reshape((3, 1)).T
                q = utils.convert_matrix_to_pc(q_np)  # this is a list of 3 floats
                
                # add to list used to compute transformation
                P.append(pc_source[i])
                Q.append(q)

            find_closest_duration = time.time() - find_closest_start
            print(f"Find Closest Histogram time: {find_closest_duration:.5f} sec. ")
            
            # compute transformation
            P_np = utils.convert_pc_to_matrix(P)
            Q_np = utils.convert_pc_to_matrix(Q)
            R, t = self.find_rigid_transformation(P_np, Q_np)

            # Update all P
            for i in range(len(pc_source)):
                pc_source[i] = np.dot(R, pc_source_matrix[:, i]) + t

            # compute icp error
            dist = (np.dot(R, pc_source_matrix) + t) - pc_target_matrix
            error = np.sum(np.linalg.norm(dist, axis=0))
            errors.append(error)
            print(f'Error: {error}')

            # return when error small enough
            if error < epsilon:
                finished = True
                return pc_source, errors
        
            iter += 1
                
        return pc_source, errors
