import datasets
from transformers import AutoTokenizer

max_length = 1e16
tokenizer = AutoTokenizer.from_pretrained("tokenizer-fat5-babyLM10M")

# Load the dataset
dataset_name = "Bachstelze/BabyLM-10M-2025-shuffled"
train_dataset = datasets.load_dataset(dataset_name, split="train", num_proc=64)
valid_dataset = datasets.load_dataset(dataset_name, split="validation", num_proc=64)
test_dataset = datasets.load_dataset(dataset_name, split="test", num_proc=64)

def tokenize_with_length(x):
    tokens_dict = tokenizer(x['text'], padding='do_not_pad', truncation=False, max_length=max_length, return_special_tokens_mask=True, return_tensors='pt')
    tokens_dict["length"] = tokens_dict["attention_mask"].sum()
    return tokens_dict

tokenized_valid = valid_dataset.map(lambda x: tokenize_with_length(x), remove_columns=['text'], batched=False, num_proc=64)
tokenized_valid.save_to_disk("babyLM10M_tokenized_flasht5" + "/valid")

tokenized_train = train_dataset.map(lambda x: tokenize_with_length(x), remove_columns=['text'], batched=False, num_proc=64)
tokenized_train.save_to_disk("babyLM10M_tokenized_flasht5" + "/train")

tokenized_test = test_dataset.map(lambda x: tokenize_with_length(x), remove_columns=['text'], batched=False, num_proc=64)
tokenized_test.save_to_disk("babyLM10M_tokenized_flasht5" + "/test")
