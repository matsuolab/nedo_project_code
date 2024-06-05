#!/bin/bash
set -e

echo "=====squeue====="
squeue --json

echo "=====sinfo====="
sinfo --json

echo "=====disk====="
df -h

echo "=====git====="
cd /storage3/GENIAC_haijima
git branch
git status

echo "=====who====="
who
