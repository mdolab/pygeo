#!/bin/bash
set -e
./input_files/get-input-files.sh

# No tests should be skipped on GCC, non Intel MPI, and x86 arch
if [[ $COMPILERS == "gcc" ]] && [[ -z $I_MPI_ROOT ]] && [[ "$(arch)" == "x86_64" ]]; then
    EXTRA_FLAGS='--disallow_skipped'
fi

export OMPI_MCA_rmaps_base_oversubscribe=1
export PRTE_MCA_rmaps_default_mapping_policy=:oversubscribe

cd tests
testflo -v -n 1 --coverage --coverpkg pygeo $EXTRA_FLAGS
