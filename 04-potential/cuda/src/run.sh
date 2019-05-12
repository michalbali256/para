#!/bin/bash

RADIUS=5

srun -p volta-hp make

echo "----------"
srun -p volta-hp --gres=gpu:1 ./cuda-blur-stencil serial $RADIUS ../data/lenna.pbm ../data/result-serial.pbm

echo "----------"
srun -p volta-hp --gres=gpu:1 ./cuda-blur-stencil cuda $RADIUS ../data/lenna.pbm ../data/result-cuda.pbm

echo "----------"
echo "Comaring results..."
if diff ../data/result-serial.pbm ../data/result-cuda.pbm; then
	echo "OK"
fi

