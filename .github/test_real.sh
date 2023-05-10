#!/bin/bash
set -e
./input_files/get-input-files.sh

# No tests should be skipped on GCC and non Intel MPI
if [[ $COMPILERS == "gcc" ]] && [[ -z $I_MPI_ROOT ]]; then
    EXTRA_FLAGS='--disallow_skipped'
fi

cd tests
testflo -v -n 1 --coverage --coverpkg pygeo $EXTRA_FLAGS
