#!/bin/bash
set -e
./input_files/get-input-files.sh

# No tests should be skipped on GCC, non Intel MPI, and x86 arch
if [[ $COMPILERS == "gcc" ]] && [[ -z $I_MPI_ROOT ]] && [[ "$(arch)" == "x86_64" ]]; then
    EXTRA_FLAGS='--disallow_skipped'
fi

cd tests
testflo -v -n 1 --coverage --coverpkg pygeo $EXTRA_FLAGS
