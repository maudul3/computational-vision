import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.cm as cm
from random import random


dir_path = Path(__file__).parent

class Coordinate:
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y

class Centroid:
    # Centroid class that keeps track of a mean and indexes of its datapoints
    def __init__(self, centroid = None):
        if centroid:
            self.mean = Coordinate(centroid.mean.x, centroid.mean.y)
            self.datapoints_idx = list(centroid.datapoints_idx)
        else:
            self.mean = Coordinate(2*random() - 1, 2*random() - 1)
            self.datapoints_idx = []


    def reset_datapoints(self):
        '''Reset datapoints associated with centroid'''
        self.datapoints_idx = []

    def calculate_new_mean(self, data):
        '''Calculate new mean given data that is associated with centroid'''
        sum_x = 0
        sum_y = 0
        length = len(self.datapoints_idx)

        for idx in self.datapoints_idx:
            sum_x += data[idx].x
            sum_y += data[idx].y

        if length != 0:
            self.mean = Coordinate(sum_x/length, sum_y/length)

    def distance(self, coord):
        '''Determine the distance of a coordinate from centroid mean'''
        return (coord.x - self.mean.x)**2 + (coord.y - self.mean.y)**2

    def calculate_mean_squared_error(self, data):
            '''Calculate the mean squared error for this centroid'''
            mse = 0
            for idx in self.datapoints_idx:
                mse += (data[idx].x - self.mean.x)**2 + (data[idx].y - self.mean.y)**2 
            return mse

def plot_data_and_centroids(data, centroids, title):
    # Plot all of the clusters
        colors = cm.rainbow(np.linspace(0, 1, len(centroids)))

        for centroid_idx, (centroid, color) in enumerate(zip(centroids, colors)):
            current_x = []
            current_y = []
            for data_idx in centroid.datapoints_idx:
                current_x.append(data[data_idx].x)
                current_y.append(data[data_idx].y)

            plt.scatter(x=current_x, y=current_y, label=centroid_idx, s=2, c=color)

        mean_x = []
        mean_y = []

        for centroid in centroids:
            mean_x.append(centroid.mean.x)
            mean_y.append(centroid.mean.y)

        plt.scatter(x=mean_x, y=mean_y, label="means", s=20, c="black", marker="x")
        plt.title(title)
        plt.legend()
        plt.show()
        plt.clf()

def k_means(data, k, r):
    centroids = [Centroid() for _ in range(k)]
    best_centroids = None
    best_mse = 100000000
    for iteration in range(r):
        
        '''At beginning of each run reset datapoints associated with centroid'''
        for centroid in centroids:
            centroid.reset_datapoints()

        # Determine which centroid is closest to the given datapoint
        for data_idx, coordinate in enumerate(data):
            min_distance = 10000
            min_idx = -1
            for centroid_idx, centroid in enumerate(centroids):
                distance = centroid.distance(coordinate)
                if (distance < min_distance):
                    min_distance = distance
                    min_idx = centroid_idx
            centroids[min_idx].datapoints_idx.append(data_idx)
        
        # Calculate the total mean squared error 
        total_mse = 0
        for centroid in centroids:
            centroid.calculate_new_mean(data)
            total_mse += centroid.calculate_mean_squared_error(data)
        
        plot_data_and_centroids(data, centroids, "Iteration #{} with MSE: {}".format(iteration + 1, total_mse))

        if total_mse < best_mse:
            best_centroids = list([Centroid(cent) for cent in centroids])
            best_mse = total_mse
    
    return (best_centroids, best_mse)
if __name__ == '__main__':

    gaussian_data = []
    with open(dir_path / '510_cluster_dataset.txt', 'r') as f:
        for line in f.readlines():
            x, y = line.split()
            gaussian_data.append(Coordinate(float(x),float(y)))

    '''Run the k-means algorithm r times'''
    r = 20
    k = 5
    best_c, best_mse = k_means(gaussian_data, k, r)

    plot_data_and_centroids(gaussian_data, best_c, "The best of the best with MSE: " + str(best_mse))