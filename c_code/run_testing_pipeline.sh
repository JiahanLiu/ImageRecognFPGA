#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: ./run_testing_pipeline.sh <jpg image>"
    exit 1
fi

echo "Converting image to C-style array..."
python create_test_image.py $1

echo "Compiling C code..."
gcc *.h *.c

echo "Running C code..."
./a.out

echo "Comparing C preprocessing to Python preprocessing..."
python display_preprocessed_image.py output_image_test.txt $1

