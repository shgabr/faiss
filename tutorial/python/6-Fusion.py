# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

d = 64  # dimension
num_filters = 4
fd = 1
nb = 100000  # database size
nq = 10000  # nb of queries
np.random.seed(1234)  # make reproducible
xb = np.random.random((nb, d)).astype("float32")
xb[:, 0] += np.arange(nb) / 1000.0
xq = np.random.random((nq, d)).astype("float32")
xq[:, 0] += np.arange(nq) / 1000.0

xbf = np.random.random((nb, num_filters, fd)).astype("float32")
xqf = np.random.random((nq, num_filters, fd)).astype("float32")

print("Data shape: ", xb.shape)
print("Filter shape: ", xbf.shape)
print("Query shape: ", xq.shape)
print("Query filter shape: ", xqf.shape)

import faiss  # make faiss available

index = faiss.IndexFlatFusion(d, num_filters, fd)  # build the index
print(index.is_trained)
index.add(xb, xbf)  # add vectors to the index
print(index.ntotal)

k = 4  # we want to see 4 nearest neighbors
D, I = index.search(xq[:5], k, xqf[:5])  # sanity check
print(I)
print(D)
D, I = index.search(xq, k, xqf)  # actual search
print(I[:5])  # neighbors of the 5 first queries
print(I[-5:])  # neighbors of the 5 last queries
