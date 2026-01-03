#include "kmeans.h"

#include <random>
#include <limits>
#include <stdexcept>
#include <cmath>
#include <iostream>

KMeans::KMeans(std::size_t k, std::size_t max_iters, double tol)
    : k_(k), max_iters_(max_iters), tol_(tol) {
    if (k_ == 0) {
        throw std::runtime_error("KMeans: k must be > 0");
    }
}

double KMeans::squared_distance(const Point& a, const Point& b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("squared_distance: dimension mismatch");
    }
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        double d = a[i] - b[i];
        s += d * d;
    }
    return s;
}

void KMeans::init_kmeans_pp(const Dataset& data) {
    if (data.empty()) {
        throw std::runtime_error("KMeans: empty dataset");
    }

    std::mt19937 gen(42);
    std::uniform_int_distribution<std::size_t> dist_index(0, data.size() - 1);

    centroids_.clear();
    centroids_.reserve(k_);

    // first centroid is random point from data
    centroids_.push_back(data[dist_index(gen)]);

    // next centroids
    while (centroids_.size() < k_) {
        std::vector<double> distances(data.size());
        double total = 0.0;

        for (std::size_t i = 0; i < data.size(); ++i) {
            double min_dist = std::numeric_limits<double>::max();
            for (const auto& c : centroids_) {
                double d = squared_distance(data[i], c);
                if (d < min_dist) {
                    min_dist = d;
                }
            }
            distances[i] = min_dist;
            total += min_dist;
        }

        if (total == 0.0) {
            // all points are identical or already covered
            // just pick random point
            centroids_.push_back(data[dist_index(gen)]);
            continue;
        }

        std::uniform_real_distribution<double> dist_prob(0.0, total);
        double r = dist_prob(gen);

        double cumulative = 0.0;
        std::size_t chosen_index = 0;
        for (std::size_t i = 0; i < data.size(); ++i) {
            cumulative += distances[i];
            if (cumulative >= r) {
                chosen_index = i;
                break;
            }
        }

        centroids_.push_back(data[chosen_index]);
    }
}

void KMeans::fit(const Dataset& data) {
    if (data.empty()) {
        throw std::runtime_error("KMeans: empty dataset");
    }

    std::size_t n = data.size();
    std::size_t dim = data[0].size();
    for (const auto& p : data) {
        if (p.size() != dim) {
            throw std::runtime_error("KMeans: all points must have same dimension");
        }
    }

    if (k_ > n) {
        throw std::runtime_error("KMeans: k cannot be greater than number of points");
    }

    labels_.assign(n, 0);
    init_kmeans_pp(data);

    Dataset new_centroids(k_, Point(dim, 0.0));
    std::vector<std::size_t> counts(k_);

    for (std::size_t iter = 0; iter < max_iters_; ++iter) {
        // assignment step
        for (std::size_t i = 0; i < n; ++i) {
            double best_dist = std::numeric_limits<double>::max();
            std::size_t best_j = 0;
            for (std::size_t j = 0; j < k_; ++j) {
                double d = squared_distance(data[i], centroids_[j]);
                if (d < best_dist) {
                    best_dist = d;
                    best_j = j;
                }
            }
            labels_[i] = best_j;
        }

        // update step
        for (std::size_t j = 0; j < k_; ++j) {
            std::fill(new_centroids[j].begin(), new_centroids[j].end(), 0.0);
            counts[j] = 0;
        }

        for (std::size_t i = 0; i < n; ++i) {
            std::size_t cluster = labels_[i];
            counts[cluster] += 1;
            for (std::size_t d = 0; d < dim; ++d) {
                new_centroids[cluster][d] += data[i][d];
            }
        }

        for (std::size_t j = 0; j < k_; ++j) {
            if (counts[j] == 0) {
                // empty cluster, reinitialize centroid as random data point
                new_centroids[j] = data[j % n];
            } else {
                for (std::size_t d = 0; d < dim; ++d) {
                    new_centroids[j][d] /= static_cast<double>(counts[j]);
                }
            }
        }

        // compute max shift
        double max_shift = 0.0;
        for (std::size_t j = 0; j < k_; ++j) {
            double d = squared_distance(centroids_[j], new_centroids[j]);
            if (d > max_shift) {
                max_shift = d;
            }
        }

        centroids_ = new_centroids;

        double shift = std::sqrt(max_shift);
        std::cout << "Iteration " << iter
                  << " max centroid shift = " << shift << "\n";

        if (shift < tol_) {
            std::cout << "Converged after " << iter << " iterations\n";
            break;
        }
    }
}

std::size_t KMeans::predict(const Point& p) const {
    if (centroids_.empty()) {
        throw std::runtime_error("KMeans: model is not fitted");
    }
    double best_dist = std::numeric_limits<double>::max();
    std::size_t best_j = 0;
    for (std::size_t j = 0; j < k_; ++j) {
        double d = squared_distance(p, centroids_[j]);
        if (d < best_dist) {
            best_dist = d;
            best_j = j;
        }
    }
    return best_j;
}
