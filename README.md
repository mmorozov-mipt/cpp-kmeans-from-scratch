# cpp-kmeans-ml

Minimal K-means clustering implementation in C++ for basic machine learning tasks.

The goal of this project is to demonstrate:
- implementation of K-means algorithm from scratch
- K-means++ initialization
- work with std::vector for numeric computations
- clustering of 2D points into several groups
- export of clustering result to CSV for further analysis and visualization in Python

Contents:
- kmeans.h - KMeans class declaration
- kmeans.cpp - KMeans implementation
- main.cpp - example: clustering synthetic 2D data into 3 clusters and saving result to CSV

Build (macOS / Linux) with g++:

g++ -std=c++17 -O2 main.cpp kmeans.cpp -o kmeans_demo

Run:

./kmeans_demo

You will see:
- training log (iteration and max centroid shift)
- final centroids printed to console
- file clusters.csv created in the current directory with columns: x,y,label

You can visualize clusters.csv in Python, for example using matplotlib or any other plotting tool.

Disclaimer:

This project is for educational and portfolio purposes only.
The author is not responsible for any misuse of the code.
