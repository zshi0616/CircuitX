NUM_PROC=4
python3 -m torch.distributed.launch --nproc_per_node=$NUM_PROC debug.py \
 --gpus 1,2,4,5 \
 --batch_size 64 