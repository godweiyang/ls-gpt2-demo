python3 -m torch.distributed.launch \
    --nproc_per_node=8 \
    train.py \
    --model_name_or_path uer/gpt2-chinese-cluecorpussmall \
    --train_file train.txt \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 8 \
    --num_train_epochs 150 \
    --learning_rate 1.5e-4 \
    --output_dir /tmp/test-97 \
    --overwrite_output_dir \
    --fp16 \
    --logging_steps 10 \
    --enable_quant false
