#ifndef KMEANS_H
#define KMEANS_H

#include <vector>
#include <cstddef>

using Point = std::vector<double>;
using Dataset = std::vector<Point>;

class KMeans {
public:
    KMeans(std::size_t k, std::size_t max_iters = 100, double tol = 1e-4);

    // Run K-means on given dataset
    void fit(const Dataset& data);

    // Predict cluster index for a single point
    std::size_t predict(const Point& p) const;

    // Get centroids after training
    const Dataset& centroids() const { return centroids_; }

    // Get labels for training data (same order as in fit input)
    const std::vector<std::size_t>& labels() const { return labels_; }

private:
    std::size_t k_;
    std::size_t max_iters_;
    double tol_;

    Dataset centroids_;
    std::vector<std::size_t> labels_;

    void init_kmeans_pp(const Dataset& data);
    static double squared_distance(const Point& a, const Point& b);
};

#endif
