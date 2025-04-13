import datasets

from tokenizers import Tokenizer, Regex
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Sequence, Split, WhitespaceSplit, Digits, Punctuation
from tokenizers.normalizers import Lowercase

from transformers import T5TokenizerFast

# Load the dataset
df = datasets.load_dataset("Bachstelze/BabyLM-10M-2025-shuffled")["train"].select_columns(['text'])

def batch_iterator(dataset, batch_size=1000):
    for batch in dataset.iter(batch_size=batch_size):
        yield batch["text"]

# include learning prefix
special_tokens_dict = ["<cls>", "<s>", "</s>", "<mask>", "<pad>", "<sep>", "<unk>", "<r>", "<x>"]

# Add extra masking tokens for the FAT5 model
for i in range(256):
    special_tokens_dict.append("<extra_id_" + str(i) + ">")

#vocab_size = 32768
# test small vocab size
vocab_size = 1024

# Train the tokenizer
tokenizer = Tokenizer(BPE(unk_token="<unk>"))
trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens_dict, max_token_length=20)
tokenizer.pre_tokenizer = Sequence([Punctuation(), WhitespaceSplit(), Digits(individual_digits=True)])
tokenizer.normalizer = Lowercase()

tokenizer.train_from_iterator(batch_iterator(df), trainer)
pretrained_tokenizer = T5TokenizerFast(tokenizer_object=tokenizer, clean_up_tokenization_spaces=False)
pretrained_tokenizer.save_pretrained("tokenizer-fat5-babyLM10M")
