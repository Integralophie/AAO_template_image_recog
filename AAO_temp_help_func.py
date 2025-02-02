
import cv2
import math
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from scipy.spatial import KDTree


# Cut scale bar from the image and make it binary
def cut_scale_bar(img):
    '''
    Returns a row of binary number representing scale bar
    '''
    scale_bar = img[1800:-1, 1800:-1]
    _,thresh1 = cv2.threshold(scale_bar,80,255,cv2.THRESH_BINARY)
    plt.imshow(thresh1,cmap="gray")
    return thresh1[1].tolist()


# Calculate the translation between scale bar & pixels
def dist_btw_scale_bar(lst, value):
    '''
    Args:
        lst: a row in the image that cuts thru the scale bar
        value: usually = 255 if white

    Returns:
        length of distance between scale bar in pixels
    '''
    # find first occurance of white scale bar
    j = lst.index(value)
    # find last occurance of white scale bar
    lst.reverse()
    i = lst.index(value)

    return (len(lst) - i - 1) - j

def translate_pixel_to_nm(length_pixel, scale_bar_length):
    '''
    Args:
        length_pixel: from `dist_btw_scale_bar`, the number of pixels between scale bar
        scale_bar_length: actual scale bar length from SEM

    Returns:
        number of pixels per nanometer
    '''
    return length_pixel / scale_bar_length


def find_clusters(img, eps=8, min_samples=30):
    '''
    Finds clusters with DBSCAN algorithm.
    Args: img=thresholded binary image
    Returns: a dictionary with keys=#cluster, values=all pixel coord within the cluster

    '''
    
    # Find coordinates of dark pixels
    dark_spots_coords = np.column_stack(np.where(img == 0))
    
    # Apply DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(dark_spots_coords)
    
    # Initialize dictionary to store cluster centroids
    cluster_centers = {}
    
    # Save label and coord info to the dictionary
    for label, coord in zip(dbscan.labels_, dark_spots_coords):
        if label != -1:  # Ignore noise points
            if label not in cluster_centers:
                cluster_centers[label] = [coord]

            else:
                cluster_centers[label].append(coord)

    return cluster_centers

def remove_small_holes(cluster_centers, min_area=500):
    '''
    Hard coded to remove all holes with diameters smaller than 40nm, as they are most
    likely artifects from adaptive thresholding.

    Returns a dictionary similar to previous func, but removes small holes
    '''

    # Make a copy of the dictionary so dict doesn't complain
    cluster_center_copy = {}

    # getting rid of holes with effective diamter < 40nm
    for label,coords in cluster_centers.items():
        if len(coords) >= min_area:
            cluster_center_copy[label] = coords

    return cluster_center_copy, len(cluster_center_copy)


def annotate_image_with_labels(cluster_large_holes,original_img):
    '''
    centroid_x, centroid_y: two lists with x and y coordinates of the centroid for each
    hole
    '''

    centroids_x = []
    centroids_y = []

    for label, coords in cluster_large_holes.items():

        center = np.mean(coords, axis=0).astype(int)
        x, y = center[0], center[1]

        cv2.putText(original_img, str(label), (y, x),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=255, thickness=2)
        
        centroids_x.append(center[0])
        centroids_y.append(center[1])

    # Display the annotated image
    # cv2.imshow('Annotated Image', original_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return centroids_x, centroids_y


def nearest_neighbour_dist(cen_x,cen_y,pixel_len):
    dots = np.column_stack((cen_x, cen_y))
    tree = KDTree(dots)
    distances, _ = tree.query(dots, k=7)
    nearest_neighbor_distances = distances[:, 1]/pixel_len

    return np.average(nearest_neighbor_distances), nearest_neighbor_distances

def switch_filename_to_coord_tuple(mydict):
    '''
    Takes average value of the five images from one grid, and save the avg
    value in a dictionary. The key of the return dict is a tuple of the
    coordinates of that grid.
    
    Args: a dict contains the map of data from previous step
    
    '''
    temp_sum_dict = {}
    count_img = {}
    for img, value in mydict.items():
        if (int(img[6]),int(img[4])) not in temp_sum_dict:
            temp_sum_dict[(int(img[6]),int(img[4]))] = value
            count_img[(int(img[6]),int(img[4]))] = 1
        else:
            temp_sum_dict[(int(img[6]),int(img[4]))] += value
            count_img[(int(img[6]),int(img[4]))] += 1


    dict_norm = {}
    for img, value in temp_sum_dict.items():
        dict_norm[img] = temp_sum_dict[img] / count_img[img]

    return dict_norm


def switch_dict_to_ndarray(mydict):
    '''
    Args: dict_norm from previous function
    Returns: a np 2d array for later heatmap ploting
    '''

    my_array = np.zeros((4,5))
    for i in range(len(my_array)):
        for j in range(len(my_array[0])):
            if (i+1,j+1) not in mydict:
                print("no")
                pass
            else:
                my_array[i][j] = mydict[(i+1,j+1)]

    return my_array
            
def flatten_concatenation(matrix):
    flat_list = []    
    for row in matrix:
        flat_list += row
        
    return flat_list