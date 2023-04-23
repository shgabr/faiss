/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdio>
#include <cstdlib>
#include <random>

#include <faiss/IndexFlat.h>

// 64-bit int
using idx_t = faiss::idx_t;

int main() {
    int d = 64; // dimension
    int fd = 1;
    int num_filters = 3;
    int nb = 100000; // database size
    int nq = 10000;  // nb of queries

    std::mt19937 rng;
    std::uniform_real_distribution<> distrib;

    float* xb = new float[d * nb];
    float* xq = new float[d * nq];
    float* xbf = new float[fd * nb];
    float* xqf = new float[fd * nq];

    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < d; j++)
            xb[d * i + j] = distrib(rng);

        for (int f = 0; f < num_filters; f++) {
            for (int j = 0; j < fd; j++) {
                xb[fd * i + j + f * i * fd] = distrib(rng);
            }
        }

        xb[d * i] += i / 1000.;
        xbf[fd * i] += i / 1000.;
    }

    for (int i = 0; i < nq; i++) {
        for (int j = 0; j < d; j++)
            xq[d * i + j] = distrib(rng);

        for (int f = 0; f < num_filters; f++) {
            for (int j = 0; j < fd; j++) {
                xq[fd * i + j + f * i * fd] = distrib(rng);
            }
        }

        xq[d * i] += i / 1000.;
        xqf[fd * i] += i / 1000.;
    }

    faiss::IndexFlatFusion index(d, num_filters, fd);  // call constructor
    printf("is_trained = %s\n", index.is_trained ? "true" : "false");
    index.add(nb, xb, xbf);              // add vectors to the index
    printf("ntotal = %zd\n", index.ntotal);

    int k = 4;

    { // sanity check: search 5 first vectors of xb
        idx_t* I = new idx_t[k * 5];
        float* D = new float[k * 5];
        float* FD = new float[k * 5];

        index.search(5, xb, xbf, num_filters, fd, k, D, FD, I);

        // print results
        printf("I=\n");
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < k; j++)
                printf("%5zd ", I[i * k + j]);
            printf("\n");
        }

        printf("D=\n");
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < k; j++)
                printf("%7g ", D[i * k + j]);
            printf("\n");
        }

        delete[] I;
        delete[] D;
    }

    { // search xq
        idx_t* I = new idx_t[k * nq];
        float* D = new float[k * nq];
        float* FD = new float[k * nq];

        index.search(nq, xq, xqf, num_filters, fd, k, D, FD, I);

        // print results
        printf("I (5 first results)=\n");
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < k; j++)
                printf("%5zd ", I[i * k + j]);
            printf("\n");
        }

        printf("I (5 last results)=\n");
        for (int i = nq - 5; i < nq; i++) {
            for (int j = 0; j < k; j++)
                printf("%5zd ", I[i * k + j]);
            printf("\n");
        }

        delete[] I;
        delete[] D;
    }

    delete[] xb;
    delete[] xq;

    return 0;
}
