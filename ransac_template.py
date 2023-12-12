#!/usr/bin/env python
import utils
import numpy
import matplotlib.pyplot as plt
###YOUR IMPORTS HERE###
import numpy as np

def is_collinear(p1, p2, p3, epsilon=1e-6):
    vector1 = np.subtract(p2, p1).reshape(3, )
    vector2 = np.subtract(p3, p1).reshape(3, )
    cross_product = np.cross(vector1, vector2)
    
    # If magnitude of cross product is close to zero, points are collinear
    return np.linalg.norm(cross_product) < epsilon

def fit_plane(points):
    # Compute centroid of points
    centroid = np.mean(points, axis=1)  # mean of [[x], [y], [z]]
    
    # If more than three points are available, randomly select three to fit the plane
    if points.shape[1] > 3:
        # Randomly select three indices without replacement
        indices = np.random.choice(points.shape[1], 3, replace=False)
        sampled_points = points[:, indices]
    else:
        # If there are exactly three points, use them
        sampled_points = points
    
    centered_points = sampled_points - centroid.reshape((3, 1))
    
    U, sigma, Vt = np.linalg.svd(centered_points)
    
    # Normal vector is the last column of vh (corresponding to smallest singular value)
    normal_vector = Vt[-1, :]
    
    # Plane parameters: ax + by + cz + d = 0
    a, b, c = normal_vector.tolist()[0]
    d = -np.dot(normal_vector, centroid)
    d = d.tolist()[0][0]
        
    return a, b, c, d

def compute_residuals(point, plane_params):
    a, b, c, d = plane_params
    
    num = np.abs(a * point[0, :] + b * point[1, :] + c * point[2, :] + d)
    den = np.sqrt(a**2 + b**2 + c**2)
    distance = num / den
    
    return distance

def ransac(pc_np, iterations, threshold, min_inliers):
    e_best = float('inf')
    plane_params_best = None
    best_point = None
    
    for i in range(iterations):
        # Pick random subset - random sample 3 points in 200 point clouds
        indices = np.random.choice(pc_np.shape[1], 3, replace=False)
        random_points = pc_np[:, indices]
        
        # Check whether collinear, if so, then resample 3 points 
        while is_collinear(random_points[:, 0], random_points[:, 1], random_points[:, 2]):
            print("Points collinear, resampling")
            indices = np.random.choice(pc_np.shape[1], 3, replace=False)
            random_points = pc_np[:, indices]
        
        # Fit the model
        plane_params = fit_plane(random_points)
                
        # Make consensus set
        C = []
        
        for j in range(pc_np.shape[1]):
            if j not in indices:
                point = pc_np[:, j]
                error = compute_residuals(point.reshape((3, 1)), plane_params)
                
                if error[0] <= threshold:
                    C.append(point)
                    
        # Refit the model if concensus is large enough
        if len(C) >= min_inliers:
            C_np = utils.convert_pc_to_matrix(C)
            union = np.hstack((C_np, random_points))
            plane_params = fit_plane(union)
            # print(f"Refitting: {len(C)} points")

            # Compute the total error for the new model
            e_new = np.sum(compute_residuals(union, plane_params))

            # Update the best model if necessary
            if e_new < e_best:
                print(f"Iteration: {i}, Error: {e_best} to {e_new}. \n")
                e_best = e_new
                plane_params_best = plane_params
                best_point = np.mean(random_points, axis=1)
    
    print(f"Best plane parameters: {plane_params_best}")
    
    return e_best, plane_params_best, best_point

def get_inliers_outliers(pc, plane_params_best, threshold=0.2):
    inliers = []
    outliers = []
    for point in pc:
        error = compute_residuals(point.reshape((3, 1)), plane_params_best)
        if error[0] <= threshold:
            inliers.append(point)
        else:
            outliers.append(point)
    print()
    print(f"Inliners: {len(inliers)}")
    print(f"Outliners: {len(outliers)}")
    
    return inliers, outliers

def plot_ransac(inliers, outliers, plane_params_best, best_point):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x_min, x_max = -0.5, 1.0
    y_min, y_max = -0.5, 1.0
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])

    a, b, c, d = plane_params_best

    normal_vector = np.matrix([a, b, c]).reshape(3, 1)
    pt = best_point
    color_green = (0.0, 1.0, 0.0, 0.5)
    ax.set(xlabel='x', ylabel='y', zlabel='z',
           title='RANSAC Plane: {0:.4f}x + {1:.4f}y + {2:.4f}z = {3:.4f}'.format(a, b, c, d))

    fig = utils.draw_plane(fig, normal_vector, pt, color=color_green, 
                           length=[x_min, x_max], width=[y_min, y_max])
    
    utils.view_pc([inliers], fig, 'r')
    utils.view_pc([outliers], fig, 'b')
    print('The equation of the plane is {0}x + {1}y + {2}z = {3}'.format(a, b, c, d))
    
###YOUR IMPORTS HERE###


def main():
    #Import the cloud
    pc = utils.load_pc('cloud_ransac.csv')


    ###YOUR CODE HERE###
    # Show the input point cloud
    utils.view_pc([pc])
    plt.title("Original point cloud")

    #Fit a plane to the data using ransac
    pc_np = utils.convert_pc_to_matrix(pc)
    iterations = 500
    threshold = 0.2
    theta = 0.01
    min_inliers = 200
    
    
    # RANSAC
    _, plane_params_best, best_point = ransac(pc_np, iterations, threshold, min_inliers)
    print(best_point)

    # Show the resulting point cloud
    inliers, outliers = get_inliers_outliers(pc, plane_params_best, threshold=threshold)
    plot_ransac(inliers, outliers, plane_params_best, best_point)
    
    print("RANSAC Finished!")

    ###YOUR CODE HERE###
    plt.show()
    #input("Press enter to end:")


if __name__ == '__main__':
    main()
