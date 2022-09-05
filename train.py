import math
from dataclasses import dataclass
from itertools import chain

from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    BertTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from layer import inject_ls_layer


@dataclass
class ModelArguments:
    model_name_or_path: str
    enable_quant: bool = False


@dataclass
class DataTrainingArguments:
    train_file: str
    block_size: int = 500


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    set_seed(training_args.seed)

    data_files = {}
    dataset_args = {}
    data_files["train"] = data_args.train_file
    dataset_args["keep_linebreaks"] = True
    raw_datasets = load_dataset(
        "text",
        data_files=data_files,
        **dataset_args,
    )
    raw_datasets["validation"] = load_dataset(
        "text",
        data_files=data_files,
        split=f"train[:{5}%]",
        **dataset_args,
    )
    raw_datasets["train"] = load_dataset(
        "text",
        data_files=data_files,
        split=f"train[{5}%:]",
        **dataset_args,
    )

    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    tokenizer = BertTokenizer.from_pretrained(model_args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
    )

    # Replace with LightSeq encoder layers.
    inject_ls_layer(model, training_args, model_args, config)

    model.resize_token_embeddings(len(tokenizer))

    column_names = raw_datasets["train"].column_names

    def tokenize_function(examples):
        output = tokenizer(examples["text"])
        return output

    with training_args.main_process_first(desc="dataset map tokenization"):
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=column_names,
        )
    block_size = min(data_args.block_size, tokenizer.model_max_length)

    def group_texts(examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    with training_args.main_process_first(desc="grouping texts together"):
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
        )
    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()

    metrics = train_result.metrics
    max_train_samples = len(train_dataset)
    metrics["train_samples"] = min(max_train_samples, len(train_dataset))
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    metrics = trainer.evaluate()
    max_eval_samples = len(eval_dataset)
    metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
    metrics["perplexity"] = math.exp(metrics["eval_loss"])
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
