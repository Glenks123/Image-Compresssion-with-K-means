# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 13:23:40 2022

@author: HP
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# def load_data(filename, plot=False):
#     X = np.load(os.path.join('data', filename))
    
#     if plot:
#         plt.scatter(X[:, 0], X[:, 1], marker='x')
#         plt.plot()
#         plt.show()
        
#     return X

def load_image(filename, visualize=False):
    image = plt.imread(os.path.join('images', filename))
    
    if visualize:
        plt.imshow(image)
    return image

def transform_image(image):
    """
        Transform image (3 dimensional matrix) into a 2 dimensional matrix
        1. Divide by 255 so values are between 0 - 1
        2. Transform such that each row contains a red, green and blue pixel
            m * 3; where m -> no. of pixels in image
    """
    
    image = image / 255
    m = image.shape[0] * image.shape[1]
    image = np.reshape(image, (m, 3))
    return image

def intialize_clusters(X, K):
    intial_clusters = np.random.permutation(X)[:K]
    return intial_clusters

def cluster_assignment(X, centroids):
    """
        1. Find distance between clusters and data points
        2. Assign data points to the closest cluster
        3. Return index of cluster to which x_i has been assigned to
    """
    
    m, n = X.shape
    idx = np.zeros((m, 1))
    f_distances = []
    
    for i in range(m):
        distances = []
        for centroid in centroids:
            L2_norm = np.sqrt(sum((X[i] - centroid) ** 2))
            squared_distance = L2_norm ** 2
            distances.append(squared_distance)
        
        idx[i] = distances.index(min(distances))
        f_distances.append(min(distances))
        
    return idx, f_distances


def move_centroids(X, idx, K):
    """
        1. Find all data points assigned to a particular cluster
        2. Calculate mean position of the data points
        3. Move cluster centroid to mean position
    """
    
    m, n = X.shape
    centroids = np.zeros((K, n))
    
    for k in range(K):
        data_points = X[np.where(idx == k)[0], :]
        cluster_pos = sum(data_points) / data_points.shape[0]
        centroids[k] = cluster_pos

    return centroids

def compute_distortion(X, distances):
    m, n = X.shape
    distortion = (1/m) * sum(distances)
    return distortion

def KMeans(X, K, max_iters=15, plot_progess=False): 
    
    m, n = X.shape
    centroids = intialize_clusters(X, K)
    previous_centroids = centroids 
    idx = np.zeros(m)
    
    for i in range(max_iters):
        idx, distances = cluster_assignment(X, centroids)
        centroids = move_centroids(X, idx, K)
        distortion = compute_distortion(X, distances)
        print(f'Iter: {i+1} | Cost: {distortion}')
        
        if plot_progess:
            plt.scatter(X[:, 0], X[:, 1], c=idx)
            plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', c='k', linewidths=3)

            # Plot history of the centroids with lines
            for j in range(centroids.shape[0]):
                plt.plot([centroids[j, :][0], previous_centroids[j, :][0]], 
                         [centroids[j, :][1], previous_centroids[j, :][1]], 
                         "-k", linewidth=1)

            plt.title(f"Iteration number {i}")
            plt.show()
        

    return centroids, idx

def display_images(image1, image2, K):
    fig, ax = plt.subplots(1,2, figsize=(8,8))
    plt.axis('off')
    
    # Display original image
    ax[0].imshow(image1)
    ax[0].set_title('Original')
    ax[0].set_axis_off()
    
    
    # Display compressed image
    ax[1].imshow(image2)
    ax[1].set_title(f'Compressed with {K} colours')
    ax[1].set_axis_off()
    
    plt.savefig(os.path.join('images', 'comparison.png'), bbox_inches='tight')


original_image = load_image('parrots.jpeg', visualize=True)
transformed_image = transform_image(original_image)

# Running K-means on pre-processed image
K = 5 # represents the no. of colours to represent the image
centroids, idx = KMeans(transformed_image, K, max_iters=10, plot_progess=True)

# Representing image in terms of indices
X_recovered = centroids[idx.astype(int), :]
# Reshaping recovered image
X_recovered = np.reshape(X_recovered, original_image.shape)

display_images(original_image, X_recovered, K)


