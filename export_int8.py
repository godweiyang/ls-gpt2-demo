import os
import json
import argparse
from collections import OrderedDict

import h5py
import numpy as np

import torch
from lightseq.training.ops.pytorch.export_quant import (
    export_ls_quant_encoder,
    fill_quant_hdf5_layer,
    quantize,
)

src_emb_mapping_dict = OrderedDict(
    {
        "norm_scale": "ln_f weight",
        "norm_bias": "ln_f bias",
        "output_ln_clip_max": "lm_head input_quant clip_value_max",
        "logits_clip_max": "lm_head output_quant _amax",
    }
)


def extract_gpt_weights(
    output_file,
    model_dir,
    topk=1,
    topp=0.75,
    eos_id=21127,
    pad_id=21128,
    max_step=500,
):
    with open(os.path.join(os.path.dirname(model_dir), "config.json")) as f:
        config = json.load(f)
    head_num = config["n_head"]
    state_dict = torch.load(model_dir, "cpu")
    var_name_list = list(state_dict.keys())

    hdf5_file = h5py.File(output_file, "w")

    emb_dim = state_dict["transformer.wte.weight"].shape[1]
    layer_nums = 0
    for name in var_name_list:
        if name.endswith("para"):
            layer_nums += 1

    export_ls_quant_encoder(hdf5_file, state_dict, emb_dim, emb_dim * 4, False, True)

    fill_quant_hdf5_layer(
        var_name_list,
        state_dict,
        hdf5_file,
        "src_embedding/",
        src_emb_mapping_dict,
    )

    token_embedding = state_dict["transformer.wte.weight"]
    token_embedding = quantize(
        token_embedding.numpy(),
        127,
        state_dict["transformer.wte.emb_quant.clip.clip_value_max"].numpy(),
    ).transpose()
    hdf5_file.create_dataset(
        "src_embedding/token_embedding", data=token_embedding, dtype="uint8"
    )
    hdf5_file.create_dataset(
        "src_embedding/emb_clip_max",
        data=state_dict["transformer.wte.emb_quant.clip.clip_value_max"],
    )

    position_emb = state_dict["transformer.wpe.weight"][:max_step, :].flatten().tolist()
    hdf5_file.create_dataset(
        "src_embedding/position_embedding", data=position_emb, dtype="f4"
    )

    hdf5_file.create_dataset("model_conf/n_encoder_stack", data=layer_nums, dtype="i4")
    hdf5_file.create_dataset("model_conf/head_num", data=head_num, dtype="i4")
    hdf5_file.create_dataset("model_conf/src_padding_id", data=pad_id, dtype="i4")
    hdf5_file.create_dataset(
        "model_conf/sampling_method",
        data=np.array([ord(c) for c in "topk"]).astype(np.int8),
        dtype="i1",
    )
    hdf5_file.create_dataset("model_conf/topp", data=topp, dtype="f4")
    hdf5_file.create_dataset("model_conf/topk", data=topk, dtype="i4")
    hdf5_file.create_dataset("model_conf/eos_id", data=eos_id, dtype="i4")

    hdf5_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", type=str)
    args = parser.parse_args()

    model_name = ".".join(args.m.split(".")[:-1])
    hdf5_path = f"{model_name}.hdf5"

    topk = 1
    topp = 0.75
    eos_id = 21127
    pad_id = 21128
    max_step = 500
    extract_gpt_weights(
        hdf5_path,
        args.m,
        topk=topk,
        topp=topp,
        eos_id=eos_id,
        pad_id=pad_id,
        max_step=max_step,
    )
