import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import utils
import time

class PFH:
    def __init__(self, pc, bins_number=125, radius=5, k=20):
        """
        Initialize the PFH object.

        :param pc:            The point cloud data, assumed to be a list of (3, 1) np.matrix with n length.
        :param pc_matrix:     The point cloud data, an (3, N) np.matrix.
        :param pc_array:      The point cloud data, an (N, 3) np.array.
        :param normals:       The normals for each point in the point cloud, an (N, 3) np.array.
        :param bins_number:   The number of bins to use for the histogram, default if 125.
        :param radius:        The radius within which to search for neighboring points, default if 5.
        """
        self.pc = pc
        self.pc_matrix = utils.convert_pc_to_matrix(pc)
        self.pc_array = np.asarray(self.pc_matrix.T)
        self.k = k
        self.bins_number = bins_number
        self.radius = radius
        self.normals = self.calculate_normal_vec(self.pc_array)

    def random_trans_matrix(self):
        """ Generate random transformation matrix for target point """
        # Generate random rotation and translation
        rotation = R.random()
        rot_mat = rotation.as_matrix()
        trans_mat = np.random.rand(3)

        # Combine into a Transformation matrix
        transformation_mat = np.eye(4)
        transformation_mat[:3, :3] = rot_mat
        transformation_mat[:3, 3] = trans_mat

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

    def calculate_normal_vec(self, pc):
        """
        Calculate surface normals for each point in the point cloud using Principal Component Analysis (PCA) on its k nearest neighbors.
        
        :param pc:   The point cloud data as an (N, 3) numpy array, where N is the number of points.
        :return:     A numpy array of shape (N, 3) containing the calculated normals for each point in the point cloud.
        """
        normals = []

        for i, point in enumerate(pc):
            # calculate distance
            distances = np.linalg.norm(pc - point, axis=1)

            # Find k nearest neighbors
            idx = np.argsort(distances)[1 : self.k + 1]
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
        """ find the point idx which has the most silimar histogram """
        dists = np.linalg.norm(hists_target - hist_source, axis=1)
        closest_idx = np.argmin(dists)

        return closest_idx

    def get_neighbors_radius(self, point, pc):
        """ Find neighbors based on radius """
        neighbor_indices = []
        for i, other_point in enumerate(pc):
            if np.linalg.norm(point - other_point) < self.radius:
                neighbor_indices.append(i)
        return neighbor_indices

    def calculate_feature(self, center, neighbor, center_normal, neighbor_normal):
        # claculate feature used for histogram
        u = center_normal
        v = np.cross((neighbor - center) / np.linalg.norm(neighbor - center), u)
        w = np.cross(u, v)

        alpha = np.dot(v, neighbor_normal)
        phi = np.dot(u, (neighbor - center) / np.linalg.norm(neighbor - center))
        theta = np.arctan2(np.dot(w, neighbor_normal), np.dot(u, neighbor_normal))
        
        return np.array([alpha, phi, theta])

    def get_bin_index(self, feature):
        # normalize the fuature
        normalized_feature = feature / (2 * np.pi)

        # calculate the bin index
        bin_index = normalized_feature * self.bins_number
        int_bin_index = np.array([int(i) for i in bin_index])

        return int_bin_index

    def pfh(self, pc, normals, selected_points):
        """
        Computes Point Feature Histograms (PFH) for selected points in a 3D point cloud.
        This method identifies the k-nearest neighbors within a specified radius for each selected point, then calculates PFH features by examining the geometric relationships between each point and its neighbors. The calculated features are binned into histograms, which serve as robust descriptors for the local geometry around each selected point.

        Parameters:
            pc (np.array): The point cloud as an (N, 3) array, where N is the number of points.
            normals (np.array): The normals corresponding to each point in the point cloud, structured as an (N, 3) array.
            selected_points (list[int]): Indices of points within the point cloud for which PFHs are to be computed.

        Returns:
            np.array: An array of PFH histograms for the selected points, each histogram corresponding to the local geometry descriptor of a point.
        """
        histograms = np.zeros((len(pc), self.bins_number))
        
        for idx in selected_points:
            # 1. Find neighbor within the radius
            neighbor_indices = self.get_neighbors_radius(pc[idx], pc)
            histogram = np.zeros([self.bins_number])

            # 2. Compute PFH Features and Update Histogram
            for j in neighbor_indices:
                for k in neighbor_indices:
                    if j != k:
                        feature = self.calculate_feature(pc[j], pc[k], normals[j], normals[k])

                        # Histogram bin
                        bin_index = self.get_bin_index(feature)
                        histogram[bin_index] += 1

            histograms[idx] = histogram

        return histograms

    def icp_pfh(self, pc_source, pc_target, max_iteration=10, epsilon=30000, num_selected=400):
        # Compute centroid of source and target
        iter = 0
        finished = False
        errors = []
        
        # pc target
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
            print(f"Normal Vector time: {time.time() - normal_vec_start:.5f} sec. ")

            # randomly selected points for histogram to reduce computing time
            selected_points = np.random.choice(len(pc_source), num_selected, replace=False)

            # source histogram
            start = time.time()
            score_histograms = self.pfh(pc_source_array, pc_source_normals, selected_points) # (3400, 125)
            find_closest_start = time.time()
            print(f'PFH time: {time.time() - start}')

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
