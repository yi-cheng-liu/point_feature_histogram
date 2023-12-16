#!/usr/bin/env python
import utils
import numpy
import matplotlib.pyplot as plt
###YOUR IMPORTS HERE###
import numpy as np


def find_closest_point(point, pc):
    dists = np.linalg.norm(pc - point, axis=0)
    closest_idx = np.argmin(dists)

    return (pc[:, closest_idx])

def find_rigid_transformation(P_np, Q_np):
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

def icp(pc_source, pc_target):
    # Compute centroid of source and target
    iter = 0
    finished = False
    epsilon = 0.01
    errors = []
    
    while not finished:
        if iter == 100:
            finished = True
            return P, errors
        
        # print(f"Iteration: {iter}")
        pc_source_np = utils.convert_pc_to_matrix(pc_source)
        pc_target_np = utils.convert_pc_to_matrix(pc_target)
        
        # Compute Correspondences
        P = []
        Q = []
        
        for point in pc_source:
            q_np = find_closest_point(point, pc_target_np)
            q = utils.convert_matrix_to_pc(q_np)  # this is a list of 3 floats
            P.append(point)
            Q.append(q[0])
            
        P_np = utils.convert_pc_to_matrix(P)
        Q_np = utils.convert_pc_to_matrix(Q)
        R, t = find_rigid_transformation(P_np, Q_np)

        dist = (np.dot(R, pc_source_np) + t) - pc_target_np
        error = np.sum(np.linalg.norm(dist, axis=0))
        errors.append(error)
        if error < epsilon:
            finished = True
            return P, errors
        
        # Update all P
        for i in range(len(P)):
            pc_source[i] = np.dot(R, pc_source_np[:, i]) + t
            
        iter += 1
            
    return pc_source, errors

###YOUR IMPORTS HERE###

def main():
    #Import the cloud
    pc_source = utils.load_pc('source/course_data/cloud_icp_source.csv')

    ###YOUR CODE HERE###
    target = 3
    pc_target_filename = f'source/course_data/cloud_icp_target{target}.csv'
    pc_target = utils.load_pc(pc_target_filename) # Change this to load in a different target
    
    fig1 = utils.view_pc([pc_source, pc_target], None, ['b', 'r'], ['o', '^'])
    plt.title("Initial Point Clouds")
    plt.axis([-0.15, 0.15, -0.15, 0.15, -0.15, 0.15])
    
    pc_source, errors = icp(pc_source, pc_target)
    
    plt.figure()
    plt.plot(errors)
    plt.title("Error vs. Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Error")
    plt.grid(True)
    
    fig2 = utils.view_pc([pc_source, pc_target], None, ['b', 'r'], ['o', '^'])
    plt.title("ICP Point Clouds")
    plt.axis([-0.15, 0.15, -0.15, 0.15, -0.15, 0.15])

    
    ###YOUR CODE HERE###

    plt.show()
    # raw_input("Press enter to end:")


if __name__ == '__main__':
    main()