#!/bin/bash
set -e
./input_files/get-input-files.sh

# No tests should be skipped on GCC, non Intel MPI, and x86 arch
if [[ $COMPILERS == "gcc" ]] && [[ -z $I_MPI_ROOT ]] && [[ "$(arch)" == "x86_64" ]]; then
    EXTRA_FLAGS='--disallow_skipped'
fi

# Set OpenMPI env variables only on non-Intel MPI
if [[ -z $I_MPI_ROOT ]]; then
    # Set these to allow MPI oversubscription because the tests need to run on specific number of procs but the test runner may have fewer
    export OMPI_MCA_rmaps_base_oversubscribe=1 # This works for OpenMPI <= 4
    export PRTE_MCA_rmaps_default_mapping_policy=:oversubscribe # This works from OpenMPI >= 5
fi

cd tests
testflo -v -n 1 --coverage --coverpkg pygeo $EXTRA_FLAGS
