#include <iostream>
#include <random>
#include <fstream>
#include "kmeans.h"

int main() {
    // Generate synthetic 2D data: three clusters
    std::size_t n_per_cluster = 100;
    std::size_t k = 3;
    Dataset data;
    data.reserve(n_per_cluster * k);

    std::mt19937 gen(123);
    std::normal_distribution<double> noise(0.0, 0.5);

    // cluster centers
    Point c1{0.0, 0.0};
    Point c2{5.0, 5.0};
    Point c3{-5.0, 5.0};

    auto add_cluster = [&](const Point& center) {
        for (std::size_t i = 0; i < n_per_cluster; ++i) {
            Point p(2);
            p[0] = center[0] + noise(gen);
            p[1] = center[1] + noise(gen);
            data.push_back(p);
        }
    };

    add_cluster(c1);
    add_cluster(c2);
    add_cluster(c3);

    // Run K-means
    KMeans km(k, 100, 1e-3);
    km.fit(data);

    const auto& centroids = km.centroids();
    std::cout << "\nFinal centroids:\n";
    for (std::size_t j = 0; j < centroids.size(); ++j) {
        std::cout << "Cluster " << j << ": ("
                  << centroids[j][0] << ", "
                  << centroids[j][1] << ")\n";
    }

    const auto& labels = km.labels();

    // Save to CSV
    std::ofstream out("clusters.csv");
    if (!out) {
        std::cerr << "Failed to open clusters.csv for writing\n";
        return 1;
    }

    out << "x,y,label\n";
    for (std::size_t i = 0; i < data.size(); ++i) {
        out << data[i][0] << "," << data[i][1] << "," << labels[i] << "\n";
    }

    std::cout << "\nSaved clusters to clusters.csv\n";
    std::cout << "You can visualize it in Python or any plotting tool.\n";

    return 0;
}
