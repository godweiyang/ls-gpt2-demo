import time
import argparse

import torch
from transformers import BertTokenizer
import lightseq.inference as lsi


def ls_gpt2(model, inputs):
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    results = model.sample(inputs)
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    return results, end_time - start_time


def ls_generate(model, tokenizer, inputs):
    ls_res_ids, ls_time = ls_gpt2(model, inputs)
    ls_res = tokenizer.batch_decode(ls_res_ids, skip_special_tokens=True)
    print(f"lightseq time: {ls_time}s")
    print("lightseq results:")
    for sent in ls_res:
        print("".join(sent.split()) + "\n===================================")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", type=str)
    parser.add_argument("-q", action="store_true")
    args = parser.parse_args()

    ls_tokenizer = BertTokenizer.from_pretrained(
        "uer/gpt2-chinese-cluecorpussmall"
    )
    ls_tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    if args.q:
        ls_model = lsi.QuantGpt(args.m, max_batch_size=16)
    else:
        ls_model = lsi.Gpt(args.m, max_batch_size=16)

    sentences = ["我要找个男朋友，", "我今天很开心，", "我要睡觉了，", "我不会再说话了，", "我要学英语了，", "我单身很久了，"]

    ls_inputs = ls_tokenizer(sentences, return_tensors="pt", padding=True)["input_ids"]
    ls_generate(ls_model, ls_tokenizer, ls_inputs)


if __name__ == "__main__":
    main()
