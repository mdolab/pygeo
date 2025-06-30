#!/bin/bash
# this file will download the input files for pyGeo regression tests
# and examples and extract them to the right place.

DIR=$(dirname $0)
wget -O $DIR/input_files.tar.gz https://websites.umich.edu/~mdolaboratory/repo_files/pyGeo/pygeo_input_files.tar.gz
tar -xzf $DIR/input_files.tar.gz -C $DIR/../
