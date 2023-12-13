#!/usr/bin/env python

import utils
import csv
import numpy
import matplotlib.pyplot as plt
def main():

    #load in the point cloud
    pc = utils.load_pc('cloud_pca.csv')

    #convert the cloud to a 3 x N matrix
    mpc = utils.convert_pc_to_matrix(pc)
    
    #add some stuff to the matrix (move the points by (1,1,1))
    mpc = mpc + numpy.ones(mpc.shape)

    #convert back to point cloud for plotting
    pc2 = utils.convert_matrix_to_pc(mpc)

    #plot the original point cloud
    fig = utils.view_pc([pc])

    #plot the moved point cloud in red ^ markers
    fig = utils.view_pc([pc2],fig,['r'], ['^'])

    #draw a plane
    fig = utils.draw_plane(fig, numpy.matrix([0,0,1]).T, numpy.matrix([0.5, 0.5, 0.5]).T, (0.1, 0.7, 0.1, 0.5), [-1, 1], [-1, 1])
   
    plt.show()
    #input("Press enter to end")


if __name__ == '__main__':
    main()
