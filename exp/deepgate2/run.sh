NUM_PROC=4
python3 -m torch.distributed.launch --nproc_per_node=$NUM_PROC --master_port 29566 train_dg2.py \
 --gpus 2,4,5,7 \
 --batch_size 128