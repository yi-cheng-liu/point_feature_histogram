#!/usr/bin/env python
import utils
import numpy
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
###YOUR IMPORTS HERE###
import numpy as np
import copy

def pca(pc, mode='no_plot'):
    # -----------------------   (a) Rotation   -----------------------
    # Rotate the points to align with the XY plane
    pc_np = utils.convert_pc_to_matrix(pc)
    mean = np.mean(pc_np, axis=1)
    pc_centered_np = pc_np - mean
    pc_count = pc_centered_np.shape[1]

    # Compute the covariance matrix
    cov = pc_centered_np @ pc_centered_np.T / (pc_count - 1)
    U, sigma, Vt = np.linalg.svd(cov)
    V = Vt.T
    print(f"V: {V}")

    # Compute the rotated point cloud
    pc_rotated_np = Vt.T @ pc_centered_np
    pc_rotated = utils.convert_matrix_to_pc(pc_rotated_np)

    # Rotated point cloud
    if mode == 'plot':
        fig2 = utils.view_pc([pc_rotated])
        plt.title("Rotated Point Cloud")
    
    # -----------------------   (b) Reduce Dimension   -----------------------
    # Compute the variance of each principle component
    S = np.diag(sigma) ** 2
    threshold = 0.01
    
    Vs = np.zeros((V.shape[0], V.shape[1]))
    for i in range(len(sigma)):
        if sigma[i] >= threshold:
            Vs[:, i] = V[:, i].reshape((3, ))
        else:
            Vs[:, i] = np.zeros(3).reshape((3, ))
    
    # Rotate the points to align with the XY plane AND eliminate the noise
    Vs = np.array(Vs)
    print(f"Vs: {Vs}")
    pc_reduced_np = Vs.T @ pc_rotated_np
    pc_reduced = utils.convert_matrix_to_pc(pc_reduced_np)
    
    if mode == 'plot':
        fig3 = utils.view_pc([pc_reduced])
        plt.title("Noise Reduced Point Cloud")
    
    return Vt, mean

def plot_pca(Vt, mean_point, pc):
    fig = utils.view_pc([pc])
    normal_vector = Vt[-1].reshape(3, 1)
    pt = mean_point
    color_green = (0, 1.0, 0, 0.3)
    x_min, x_max = -0.5, 1.0
    y_min, y_max = -0.5, 1.0
    utils.draw_plane(fig, normal_vector, pt, color=color_green, 
                     length=[x_min, x_max], width=[y_min, y_max])
    plt.title("Point Cloud with Plane")
    plt.show()
    
###YOUR IMPORTS HERE###


def main():

    # Import the cloud
    pc = utils.load_pc('cloud_pca.csv')

    ###YOUR CODE HERE###
    # Show the input point cloud
    fig1 = utils.view_pc([pc])
    plt.title("Original Point Cloud")

    # Principle Component Analysis (PCA) algorithm
    Vt, mean_point = pca(pc, mode='plot')
    
    # Draw the plane
    plot_pca(Vt, mean_point, pc)
    
    ###YOUR CODE HERE###

if __name__ == '__main__':
    main()
