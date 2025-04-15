import os
import torch
import yaml
import sys
import glob
import sys

# Get the absolute path of the current file
current_file_path = os.path.abspath(__file__)

# Get the directory of the current file (babyLM10M)
current_dir = os.path.dirname(current_file_path)

# Get the parent directory (examples)
parent_dir = os.path.dirname(current_dir)

# Get the grandparent directory (causalFlashT5), which contains src
grandparent_dir = os.path.dirname(parent_dir)

# Add the grandparent directory to the Python module search path
sys.path.append(grandparent_dir)

# Configure wandb
os.environ["WANDB_PROJECT"] = "Flash-T5-babyLM10M"

torch._dynamo.config.optimize_ddp=False

import datasets

from transformers import Trainer, TrainingArguments, AutoTokenizer, HfArgumentParser

from src.model.configuration_flash_t5 import FlashT5Config
from src.data.data_collator_ul2 import DataCollatorForUL2MLM
from src.model.modeling_flash_t5 import FlashT5ForConditionalGeneration
from optimization import create_optimizer, create_wsd_scheduler

import evaluate

# Initialize
with open(sys.argv[1], 'r') as config_file:
    config = yaml.safe_load(config_file)

tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"])

df_train = datasets.load_from_disk(config["train_dataset"], keep_in_memory=False).with_format("np")
df_valid = datasets.load_from_disk(config["valid_dataset"], keep_in_memory=False).with_format("np")

train_dataset = df_train
valid_dataset = df_valid

train_dataset = train_dataset.remove_columns(["special_tokens_mask"])
valid_dataset = valid_dataset.remove_columns(["special_tokens_mask"])

config_collator = config["collator_args"]
model_config = config["model_args"]
config_training_arguments = config["training_args"]

# denoiser list with lower case prefixes
# no prefix for causal learning because that is later on the default
# Todo enable causal denoising without prefix
custom_denoiser_list = [{"mu": 3.0, "r": 0.15, "max_spans": config_collator["max_token_length"], "prefix": "[r]"},
{"mu": 8.0, "r": 0.15, "max_spans": config_collator["max_token_length"], "prefix": "[r]"},
{"mu": 4.0, "r": 0.0, "max_spans": 1, "prefix": "[s]"},
{"mu": 3.0, "r": 0.5, "max_spans": config_collator["max_token_length"], "prefix": "[x]"},
{"mu": 8.0, "r": 0.15, "max_spans": config_collator["max_token_length"], "prefix": "[x]"},
{"mu": 64.0, "r": 0.15, "max_spans": config_collator["max_token_length"], "prefix": "[x]"},
{"mu": 64.0, "r": 0.5, "max_spans": config_collator["max_token_length"], "prefix": "[x]"}]
old_denoiser_proportions=[0.165, 0.165, 0.34, 0.0825, 0.0825, 0.0825, 0.0825]

#mostly causal denoising with variablespan size
#only 10% is masked language modeling
causal_denoiser = [
  {"mu": 4.0, "r": 0.0, "max_spans": 1, "prefix": "[S]"},
  {"mu": 6.0, "r": 0.0, "max_spans": 1, "prefix": "[S]"},
  {"mu": 8.0, "r": 0.0, "max_spans": 1, "prefix": "[S]"},
  {"mu": 10.0, "r": 0.0, "max_spans": 1, "prefix": "[S]"},
  {"mu": 12.0, "r": 0.0, "max_spans": 1, "prefix": "[S]"},
  {"mu": 14.0, "r": 0.0, "max_spans": 1, "prefix": "[S]"},
  {"mu": 16.0, "r": 0.0, "max_spans": 1, "prefix": "[S]"},
  {"mu": 18.0, "r": 0.0, "max_spans": 1, "prefix": "[S]"},
  {"mu": 20.0, "r": 0.0, "max_spans": 1, "prefix": "[S]"},
  {"mu": 32.0, "r": 0.3, "max_spans": config_collator["max_token_length"], "prefix": "[X]"}
]

causal_denoiser_proportions=[0.1]*10

data_collator = DataCollatorForUL2MLM(
    tokenizer=tokenizer,
    max_length=config_collator["max_token_length"],
    max_labels_length=config_collator["max_labels_length"],
    batch_size=config_collator["output_batch_size"],
    denoiser_list=causal_denoiser,
    denoiser_proportions=causal_denoiser_proportions
)

# Set a configuration for our T5 model
model_config["vocab_size"] = tokenizer.vocab_size
model_config["pad_token_id"] = tokenizer.pad_token_id
config_hf_model = FlashT5Config.from_dict(config["model_args"])

# Initialize the model from a configuration without pretrained weights
model = FlashT5ForConditionalGeneration(config=config_hf_model)

print('Num parameters: ', model.num_parameters())

masked_accuracy = evaluate.load("accuracy")

def preprocess_logits_for_metrics(logits, labels):
    pred_ids = torch.argmax(logits[0], dim=-1)
    return pred_ids

def compute_metrics(eval_pred):
    logits = eval_pred.predictions
    labels = eval_pred.label_ids

    labels = labels.flatten()
    predictions=logits.flatten()[labels > 0]
    labels = labels[labels > 0]

    return {'MaskedAccuracy': masked_accuracy.compute(predictions=predictions, references=labels)["accuracy"]}

parser = HfArgumentParser(TrainingArguments)
config_training_arguments["gradient_accumulation_steps"] = max(1, config_training_arguments["gradient_accumulation_steps"] // torch.cuda.device_count())
config_training_arguments["report_to"] = ["codecarbon"]
config_training_arguments["output_dir"] = config["model_name"] + "_v" + str(config["version"])
config_training_arguments["run_name"] = config["model_name"] + "_fr_" + \
    str(model_config["position_encoding_type"])
config_training_arguments["report_to"] = "wandb"
os.environ["CLEARML_TASK"]=config_training_arguments["run_name"]

training_args = parser.parse_dict(config_training_arguments)[0]

# Configure the optimizer and scheduler
optimizer = create_optimizer(model,
                            lr=training_args.learning_rate,
                            betas=(training_args.adam_beta1, training_args.adam_beta2),
                            eps=training_args.adam_epsilon,
                            weight_decay=training_args.weight_decay,
                            kahan_sum=False)
scheduler = create_wsd_scheduler(training_args.warmup_steps, training_args.warmup_ratio, training_args.max_steps, optimizer)

# Initialize the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=data_collator,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    compute_metrics=compute_metrics,
    optimizers=(optimizer, scheduler)
    )

is_already_trained = (len(set([os.path.dirname(p) for p in glob.glob(config_training_arguments["output_dir"] + "/*/*")])) != 0)
result = trainer.train(resume_from_checkpoint=(config["checkpoint_name"] and is_already_trained))
trainer.evaluate()
