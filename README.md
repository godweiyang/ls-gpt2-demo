# 极简代码，教你如何轻松加速GPT2训练和推理

## 训练
### 用fp16精度pretrain模型
```shell
python3 -m torch.distributed.launch \
    --nproc_per_node=8 \
    train.py \
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

### （可选）用int8精度finetune模型
```shell
python3 -m torch.distributed.launch \
    --nproc_per_node=8 \
    train.py \
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

## 导出
### 导出fp16模型
```shell
python3 export.py \
    -m /tmp/test-97/pytorch_model.bin \
    -l 500
```

### （可选）导出int8模型
```shell
python3 export.py \
    -m /tmp/quant/test-97/pytorch_model.bin \
    -l 500 \
    -q
```

## 生成
### 用fp16模型生成句子
```shell
python3 generate.py \
    -m /tmp/test-97/pytorch_model.hdf5 \
    -i "我好难受" \
    -p "uer/gpt2-chinese-cluecorpussmall"
```

### （可选）用int8模型生成句子
```shell
python3 generate.py \
    -m /tmp/quant/test-97/pytorch_model.hdf5 \
    -i "我好难受" \
    -p "uer/gpt2-chinese-cluecorpussmall" \
    -q
```
