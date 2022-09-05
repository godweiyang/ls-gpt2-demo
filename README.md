# 极简代码，教你如何轻松加速GPT2训练和推理

```shell
python3 -m torch.distributed.launch \
    --nproc_per_node=8 \
    train/train.py \
    --model_name_or_path uer/gpt2-chinese-cluecorpussmall \
    --train_file data/train.txt \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 8 \
    --num_train_epochs 150 \
    --learning_rate 1.5e-4 \
    --output_dir /tmp/test-97 \
    --overwrite_output_dir \
    --fp16 \
    --logging_steps 10 \
    --enable_quant false
```

```shell
python3 -m torch.distributed.launch \
    --nproc_per_node=8 \
    train/train.py \
    --model_name_or_path uer/gpt2-chinese-cluecorpussmall \
    --train_file data/train.txt \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 8 \
    --num_train_epochs 200 \
    --learning_rate 5e-6 \
    --output_dir /tmp/quant/test-97 \
    --overwrite_output_dir \
    --resume_from_checkpoint /tmp/test-97 \
    --fp16 \
    --logging_steps 10 \
    --enable_quant true
```

```shell
python3 export/export.py -m /tmp/test-97/pytorch_model.bin
```

```shell
python3 export/export_int8.py -m /tmp/quant/test-97/pytorch_model.bin
```

```shell
python3 generate/generate.py -m /tmp/test-97/pytorch_model.hdf5
```

```shell
python3 generate/generate.py -m /tmp/quant/test-97/pytorch_model.hdf5 -q
```