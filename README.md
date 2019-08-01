# collegePython
Python code I did in college.
You'll need to change the filepath for raw data if you want to run the programs.

# Data Visualization
Uses matplotlib to visualize the data, such as histograms and box plots of the data with variable bin size. It can also create a Feature-by-Feature correlation matrix as well as a heatmap of the Data Point x Data Point matrix of the distance using lp norm.

Here's the source of the data:

https://archive.ics.uci.edu/ml/datasets/Iris


https://archive.ics.uci.edu/ml/datasets/Wine

# KNN
K Nearest Neighbors
Question 1 uses the knn classifier with an 80/20 split of the data. Then it calculates the accuracy, sensitivity, and specificity of the results. Question 2 compares different p-values for the distance function using the same metrics. See the write up for more detail.

# K Means Clustering
Randomly picked points are chosen for the cluster centroids, then data points are assigned to these clusters. Questions 1 and 2 are differentiated by the Question 2 having multiple iterations and error bars on the graph. Question 3 uses kmeans++, which spreads out the initial centroids as opposed to just choosing them randomly, ensuring a more opitmal clustering. Question 4 prints out the labels of the 3 nearest neighbors of the centroid to show the inter and intra class similarity.


