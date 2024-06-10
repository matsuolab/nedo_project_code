#!/bin/bash

set -e
echo ""

###copy common file to megatron-deepspeed
cp ../common/deepspeed_checkpoint.py ../../Megatron-DeepSpeed/tools/convert_checkpoint/deepspeed_checkpoint.py 
cp ../common/deepspeed_to_megatron.py ../../Megatron-DeepSpeed/tools/convert_checkpoint/deepspeed_to_megatron.py 