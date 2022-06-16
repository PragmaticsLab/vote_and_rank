from datasets import load_metric, load_from_disk
from transformers import TrainingArguments, Trainer
from transformers import DistilBertTokenizer, BertTokenizer, RobertaTokenizer, AlbertTokenizer,\
T5Tokenizer, DebertaTokenizer, GPT2Tokenizer
from transformers import DistilBertForSequenceClassification, BertForSequenceClassification,\
RobertaForSequenceClassification, AlbertForSequenceClassification, T5ForConditionalGeneration,\
DebertaForSequenceClassification, GPT2ForSequenceClassification
import numpy as np
import os
import argparse

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task_name",
        default="rte",
        type=str,
        required=False,
    )
    
    parser.add_argument(
        "--model_name",
        default="distilbert-base-uncased",
        type=str,
        required=False,
    )

    parser.add_argument(
        "--batch_size",
        default=16,
        type=int,
        required=False,
    )
    
    parser.add_argument(
        "--num_train_epochs",
        default=5,
        type=int,
        required=False,
    )
    
    parser.add_argument(
        "--random_seed",
        default=0,
        type=int,
        required=False,
    )

    parser_args = parser.parse_args()

    print(vars(parser_args))

    task = parser_args.task_name
    model_name = parser_args.model_name
    batch_size = parser_args.batch_size
    epochs = parser_args.num_train_epochs

    name_to_model = {
        "distilbert-base-uncased": {
            "model": DistilBertForSequenceClassification,
            "tokenizer": DistilBertTokenizer
        },
        "bert-base-uncased": {
            "model": BertForSequenceClassification,
            "tokenizer": BertTokenizer
        },
        "bert-large-uncased": {
            "model": BertForSequenceClassification,
            "tokenizer": BertTokenizer
        },
        "roberta-base": {
            "model": RobertaForSequenceClassification,
            "tokenizer": RobertaTokenizer
        },
        "roberta-large": {
            "model": RobertaForSequenceClassification,
            "tokenizer": RobertaTokenizer
        },
        "distilroberta-base": {
            "model": RobertaForSequenceClassification,
            "tokenizer": RobertaTokenizer
        },
        "albert-base-v2": {
            "model": AlbertForSequenceClassification,
            "tokenizer": AlbertTokenizer
        },
        "albert-xxlarge-v2": {
            "model": AlbertForSequenceClassification,
            "tokenizer": AlbertTokenizer
        },
        "t5-base": {
            "model": T5ForConditionalGeneration,
            "tokenizer": T5Tokenizer
        },
        "deberta-base": {
            "model": DebertaForSequenceClassification,
            "tokenizer": DebertaTokenizer
        },
        "gpt2": {
            "model": GPT2ForSequenceClassification,
            "tokenizer": GPT2Tokenizer
        },
        "distilgpt2": {
            "model": GPT2ForSequenceClassification,
            "tokenizer": GPT2Tokenizer
        },
    }

    is_t5 = (model_name[:2] == 't5')
    actual_task = "mnli" if task == "mnli-mm" else task
    dataset = load_from_disk("datasets/glue/{}".format(actual_task))
    metric = load_metric("metrics/glue", actual_task)
    tokenizer = name_to_model[model_name]["tokenizer"].from_pretrained("tokenizers/{}".format(model_name), model_max_length=512)

    task_to_keys = {
        "cola": ("sentence", None),
        "mnli": ("premise", "hypothesis"),
        "mnli-mm": ("premise", "hypothesis"),
        "mrpc": ("sentence1", "sentence2"),
        "qnli": ("question", "sentence"),
        "qqp": ("question1", "question2"),
        "rte": ("sentence1", "sentence2"),
        "sst2": ("sentence", None),
        "stsb": ("sentence1", "sentence2"),
        "wnli": ("sentence1", "sentence2"),
    }

    sentence1_key, sentence2_key = task_to_keys[task]

    def preprocess_function(examples):
        if is_t5:
            examples['label'] = [[i] for i in examples['label']]
        if sentence2_key is None:
            return tokenizer(examples[sentence1_key], padding=True, truncation=True)
        return tokenizer(examples[sentence1_key], examples[sentence2_key], padding=True, truncation=True)

    encoded_dataset = dataset.map(preprocess_function, batched=True)

    def model_init():
        num_labels = 3 if task.startswith("mnli") else 1 if task=="stsb" else 2
        return name_to_model[model_name]["model"].from_pretrained("models/{}/{}".format(model_name, num_labels), num_labels=num_labels)

    metric_name = "pearson" if task == "stsb" else "matthews_correlation" if task == "cola" else "accuracy"

    args = TrainingArguments(
        output_dir=f'saved_models/{model_name}/{parser_args.task_name}/{parser_args.random_seed}',
        evaluation_strategy = "epoch",
        save_strategy = "no",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        load_best_model_at_end=False,
        metric_for_best_model=metric_name,
        seed=parser_args.random_seed,
    )

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if is_t5:
            predictions = np.array(predictions[0]).reshape([predictions[0].shape[0], predictions[0].shape[2]])
            labels = labels.reshape([len(labels)])
        if task != "stsb":
            predictions = np.argmax(predictions, axis=1)
        elif model_name != 'deberta-base':
            predictions = predictions[:, 0]
        return metric.compute(predictions=predictions, references=labels)

    validation_key = "validation_mismatched" if task == "mnli-mm" else "validation_matched" if task == "mnli" else "validation"

    trainer = Trainer(
        model_init=model_init,
        args=args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset[validation_key],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

if __name__ == "__main__":
    main()
