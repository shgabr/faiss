/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#ifndef INDEX_FLAT_H
#define INDEX_FLAT_H

#include <vector>

#include <faiss/IndexFlatCodes.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/distances.h>

namespace faiss {

/** Index that stores the full vectors and performs exhaustive search */
struct IndexFlat : IndexFlatCodes {
    explicit IndexFlat(idx_t d, MetricType metric = METRIC_L2);

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    void range_search(
            idx_t n,
            const float* x,
            float radius,
            RangeSearchResult* result,
            const SearchParameters* params = nullptr) const override;

    void reconstruct(idx_t key, float* recons) const override;

    /** compute distance with a subset of vectors
     *
     * @param x       query vectors, size n * d
     * @param labels  indices of the vectors that should be compared
     *                for each query vector, size n * k
     * @param distances
     *                corresponding output distances, size n * k
     */
    void compute_distance_subset(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            const idx_t* labels) const;

    // get pointer to the floating point data
    float* get_xb() {
        return (float*)codes.data();
    }
    const float* get_xb() const {
        return (const float*)codes.data();
    }

    IndexFlat() {}

    FlatCodesDistanceComputer* get_FlatCodesDistanceComputer() const override;

    /* The stanadlone codec interface (just memcopies in this case) */
    void sa_encode(idx_t n, const float* x, uint8_t* bytes) const override;

    void sa_decode(idx_t n, const uint8_t* bytes, float* x) const override;
};

struct IndexFlatIP : IndexFlat {
    explicit IndexFlatIP(idx_t d) : IndexFlat(d, METRIC_INNER_PRODUCT) {}
    IndexFlatIP() {}
};

struct IndexFlatL2 : IndexFlat {
    explicit IndexFlatL2(idx_t d) : IndexFlat(d, METRIC_L2) {}
    IndexFlatL2() {}
};

struct IndexFlatFusion : IndexFlat {
    explicit IndexFlatFusion(idx_t d) : IndexFlat(d, METRIC_FUSION) {}
    explicit IndexFlatFusion(idx_t d, size_t num_filters, size_t filter_dim)
            : IndexFlat(d, METRIC_FUSION), filter_size(sizeof(float)*num_filters*filter_dim) {}
    IndexFlatFusion() {}

    size_t filter_size;
    std::vector<uint8_t> filters;
    const float* get_xf() const {
        return (const float*)filters.data();
    }
    void add(idx_t n, const float* x, const float* filters) {
        FAISS_THROW_IF_NOT(is_trained);
        if (n == 0) {
            return;
        }
        this->codes.resize((ntotal + n) * code_size);
        this->filters.resize((ntotal + n) * filter_size);
        sa_encode(n, x, codes.data() + (ntotal * code_size));
        memcpy(this->filters.data() + (ntotal * filter_size), filters, n * filter_size);

        ntotal += n;
    }
    // override the search method to use the fusion distance
    void search(
            idx_t n,                // number of queries
            const float* x,         // input queries data
            const float* x_filters, // input queries filters
            int nf,                 // number of filters
            int filter_dimension,   // size of each filter
            idx_t k,            // Number of neihbors to retrieve for each query
            float* distances,   // output: L2 distances
            float* f_distances, // output: Fusion distances
            idx_t* labels,      // output: labels
            const SearchParameters* params = nullptr) {
        float_maxheap_array_t res = {
                size_t(n), size_t(k), labels, distances, f_distances};
        knn_fusion(
                x,
                x_filters,
                get_xb(),
                get_xf(),
                d,
                n,
                nf,
                filter_dimension,
                ntotal,
                &res);
    }
};

/// optimized version for 1D "vectors".
struct IndexFlat1D : IndexFlatL2 {
    bool continuous_update = true; ///< is the permutation updated continuously?

    std::vector<idx_t> perm;       ///< sorted database indices

    explicit IndexFlat1D(bool continuous_update = true);

    /// if not continuous_update, call this between the last add and
    /// the first search
    void update_permutation();

    void add(idx_t n, const float* x) override;

    void reset() override;

    /// Warn: the distances returned are L1 not L2
    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;
};

} // namespace faiss

#endif
