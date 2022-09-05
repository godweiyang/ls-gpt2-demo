from torch import nn

from lightseq.training.ops.pytorch.quantization import (
    qat_mode,
    enable_quant,
    QuantLinear,
    TensorQuantizer,
    emb_quant_config,
)
from lightseq.training.ops.pytorch.torch_transformer_layers import copy_para
from lightseq.training import ls_hf_gpt_enc_convert


def get_hf_gpt_enc_layer_params(layer, config):
    init_ws = []
    init_bs = []

    init_ws.extend(
        layer.attn.c_attn.weight.detach().clone().t().split(config.hidden_size, 0)
    )
    init_bs.extend(layer.attn.c_attn.bias.detach().clone().split(config.hidden_size, 0))

    init_ws.append(layer.attn.c_proj.weight.detach().clone().t().reshape(-1))
    init_bs.append(layer.attn.c_proj.bias.detach().clone())
    init_ws.append(layer.ln_1.weight.detach().clone())
    init_bs.append(layer.ln_1.bias.detach().clone())

    init_ws.append(layer.mlp.c_fc.weight.detach().clone().t().reshape(-1))
    init_bs.append(layer.mlp.c_fc.bias.detach().clone())
    init_ws.append(layer.mlp.c_proj.weight.detach().clone().t().reshape(-1))
    init_bs.append(layer.mlp.c_proj.bias.detach().clone())
    init_ws.append(layer.ln_2.weight.detach().clone())
    init_bs.append(layer.ln_2.bias.detach().clone())

    return init_ws, init_bs


def get_hf_gpt_emb_layer_params(layer):
    init_ws = []

    init_ws.append(layer.wte.weight.detach().clone())
    init_ws.append(layer.wpe.weight.detach().clone())

    return init_ws


class GptEmbedding(nn.Embedding):
    def __init__(self, training_args, initial_embeddings=None, *args, **kwargs):
        super(GptEmbedding, self).__init__(*args, **kwargs)
        self.emb_quant = TensorQuantizer(emb_quant_config)

        if initial_embeddings is not None:
            self.weight.data.copy_(copy_para(initial_embeddings, training_args.fp16))

    def forward(self, input_ids):
        x = super(GptEmbedding, self).forward(input_ids)
        x = self.emb_quant(x)
        return x


def inject_ls_layer(model, training_args, model_args, config):
    init_ws = get_hf_gpt_emb_layer_params(model.transformer)
    model.transformer.wte = GptEmbedding(
        training_args, init_ws[0], config.vocab_size, config.hidden_size
    )
    if model_args.enable_quant:
        model.transformer.wte.apply(qat_mode)

    ls_hf_gpt_enc_convert(model, training_args, config)
    for i in range(config.num_hidden_layers):
        if model_args.enable_quant:
            model.transformer.h[i].apply(enable_quant)

    q_lm_head = QuantLinear(config.n_embd, config.vocab_size, bias=False)
    q_lm_head.weight = model.transformer.wte.weight
    q_lm_head.weight_quant = model.transformer.wte.emb_quant
    model.lm_head = q_lm_head
    if model_args.enable_quant:
        model.lm_head.apply(qat_mode)
