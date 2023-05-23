#!/bin/bash

make -C build -j faiss swigfaiss
(cd build/faiss/python && python setup.py install)
