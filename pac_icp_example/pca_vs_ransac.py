#!/usr/bin/env python
import utils
import numpy
import time
import random
import matplotlib.pyplot as plt
###YOUR IMPORTS HERE###
import numpy as np
from pca_template import pca
from ransac_template import ransac, get_inliers_outliers, plot_ransac, compute_residuals
###YOUR IMPORTS HERE###

def add_some_outliers(pc,num_outliers):
    pc = utils.add_outliers_centroid(pc, num_outliers, 0.75, 'uniform')
    random.shuffle(pc)
    return pc

def main():
    #Import the cloud
    pc = utils.load_pc('cloud_pca.csv')

    num_tests = 10
    fig = None
    
    pca_total_time = 0
    ransac_total_time = 0
    
    pca_errors = []
    ransac_errors = []
    
    for i in range(0,num_tests):
        print(f"================  Iteration: {i}  ================")
        pc = add_some_outliers(pc,10) #adding 10 new outliers for each test

        ###YOUR CODE HERE###
        
        threshold = 0.2
        
        # PCA
        start_pca = time.time()
        Vt, best_point = pca(pc)
        end_pca = time.time()
        pca_total_time += end_pca - start_pca
        
        # Find parameters of the plane
        normal_vector = Vt[-1]
        a, b, c = normal_vector[0, 0], normal_vector[0, 1], normal_vector[0, 2]
        mean_point = np.mean(utils.convert_pc_to_matrix(pc), axis=1)
        d = -np.dot(normal_vector, mean_point)
        d = d.tolist()[0][0]
        plane_params = [a, b, c, d]
        
        # Find Inliers and Outliers
        inliers, outliers = get_inliers_outliers(pc, plane_params, threshold=threshold)
        pca_error = np.sum(compute_residuals(np.array(outliers), plane_params))
        pca_errors.append(pca_error)
        print("PCA Errors", pca_errors)

        # Draw the plane
        if i == num_tests - 1:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            utils.view_pc([inliers], fig, 'r')
            utils.view_pc([outliers], fig, 'b')
            normal_vector = Vt[-1].reshape(3, 1)
            pt = best_point
            color_green = (0, 1.0, 0, 0.3)
            x_min, x_max = -0.5, 1.0
            y_min, y_max = -0.5, 1.0
            ax.set(xlabel='x', ylabel='y', zlabel='z',
                   title='PCA Plane: {0:.4f}x + {1:.4f}y + {2:.4f}z = {3:.4f}'.format(a, b, c, d))
            utils.draw_plane(fig, normal_vector, pt, color=color_green, 
                            length=[x_min, x_max], width=[y_min, y_max])
        
        # RANSAC
        pc_np = utils.convert_pc_to_matrix(pc)
        iterations = 500
        threshold = 0.2
        min_inliers = 120
        
        
        # RANSAC
        start_ransac = time.time()
        _, plane_params_best, best_point = ransac(pc_np, iterations, threshold, min_inliers)
        end_ransac = time.time()
        ransac_total_time += end_ransac - start_ransac

        # Show the resulting point cloud
        
        inliers, outliers = get_inliers_outliers(pc, plane_params_best, threshold=threshold)
        ransac_error = np.sum(compute_residuals(np.array(outliers), plane_params_best))
        ransac_errors.append(ransac_error)
        print("RANSAC Errors", ransac_errors)
        
        if i == num_tests - 1:
            plot_ransac(inliers, outliers, plane_params_best, best_point)
            errors = np.sum(compute_residuals(np.array(outliers), plane_params_best))
            print(errors)
            plt.show()
        
        if i == num_tests - 1:
            print("PCA total time: ", pca_total_time)
            print("RANSAC total time: ", ransac_total_time)
            
            # plot of the outliers
            fig, ax = plt.subplots() 
            ax.plot(pca_errors, label='PCA')
            ax.plot(ransac_errors, label='RANSAC')
            ax.set_xlabel('Iteration or Number of Outliers')
            ax.set_ylabel('Error')
            ax.set_title('Error Comparison between PCA and RANSAC')
            ax.legend()
            plt.show()
        
        ###YOUR CODE HERE###

    input("Press enter to end")


if __name__ == '__main__':
    main()
