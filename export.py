import os
import json
import argparse
from collections import OrderedDict

import h5py
import numpy as np
import torch
from lightseq.training.ops.pytorch.export import (
    export_ls_encoder,
    fill_hdf5_layer,
)
from lightseq.training.ops.pytorch.export_quant import (
    export_ls_quant_encoder,
    fill_quant_hdf5_layer,
    quantize,
)

src_emb_mapping_dict = OrderedDict(
    {
        "norm_scale": "ln_f weight",
        "norm_bias": "ln_f bias",
    }
)


def extract_gpt_weights(
    output_file,
    model_dir,
    enable_quant,
    topk=4,
    topp=0.75,
    max_step=500,
):
    with open(os.path.join(os.path.dirname(model_dir), "config.json")) as f:
        config = json.load(f)
    eos_id = config["vocab_size"] - 1
    pad_id = config["vocab_size"]
    head_num = config["n_head"]
    state_dict = torch.load(model_dir, "cpu")
    var_name_list = list(state_dict.keys())

    hdf5_file = h5py.File(output_file, "w")

    emb_dim = state_dict["transformer.wte.weight"].shape[1]
    layer_nums = 0
    for name in var_name_list:
        if name.endswith("para"):
            layer_nums += 1

    if enable_quant:
        export_ls_quant_encoder(
            hdf5_file, state_dict, emb_dim, emb_dim * 4, False, True
        )
        src_emb_mapping_dict.update(
            {"output_ln_clip_max": "lm_head input_quant clip_value_max"}
        )
        src_emb_mapping_dict.update({"logits_clip_max": "lm_head output_quant _amax"})
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
    else:
        export_ls_encoder(hdf5_file, state_dict, emb_dim, emb_dim * 4, False)
        src_emb_mapping_dict.update({"token_embedding": "wte weight"})
        fill_hdf5_layer(
            var_name_list,
            state_dict,
            hdf5_file,
            "src_embedding/",
            src_emb_mapping_dict,
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
        data=np.array([ord(c) for c in "topp"]).astype(np.int8),
        dtype="i1",
    )
    hdf5_file.create_dataset("model_conf/topp", data=topp, dtype="f4")
    hdf5_file.create_dataset("model_conf/topk", data=topk, dtype="i4")
    hdf5_file.create_dataset("model_conf/eos_id", data=eos_id, dtype="i4")
    if not enable_quant:
        hdf5_file.create_dataset("model_conf/extra_decode_length", data=0, dtype="i4")

    hdf5_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", type=str)
    parser.add_argument("-l", type=int, default=500)
    parser.add_argument("-q", action="store_true")
    args = parser.parse_args()

    model_name = ".".join(args.m.split(".")[:-1])
    hdf5_path = f"{model_name}.hdf5"

    extract_gpt_weights(
        hdf5_path,
        args.m,
        args.q,
        max_step=args.l,
    )
