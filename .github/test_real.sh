#!/bin/bash
set -e
./input_files/get-input-files.sh

# All tests should pass on Ubuntu
if [[ $OS == "ubuntu" ]]; then
    EXTRA_FLAGS='--disallow_skipped'
fi

cd tests
testflo -v -n 1 --coverage --coverpkg pygeo $EXTRA_FLAGS
