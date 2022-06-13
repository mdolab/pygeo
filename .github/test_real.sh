#!/bin/bash
set -e
./input_files/get-input-files.sh

# all tests should pass on private
if [[ $IMAGE == "private" ]] && [[ $OS == "ubuntu" ]]; then
    EXTRA_FLAGS='--disallow_skipped'
fi

cd tests
testflo -v -n 1 --coverage --coverpkg pygeo $EXTRA_FLAGS
