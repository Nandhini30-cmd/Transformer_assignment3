import spacy
from datasets import load_dataset
from collections import Counter
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


def collate_batch(batch, pad_idx=1):
    """
    Custom collate function to pad sequences in a batch to the same length.
    
    Args:
        batch: List of (src, tgt) tuples from the dataset
        pad_idx: Index of padding token
    
    Returns:
        src_padded: Padded src tensor [batch_size, max_src_len]
        tgt_padded: Padded tgt tensor [batch_size, max_tgt_len]
    """
    src_batch, tgt_batch = zip(*batch)
    
    # Pad sequences to max length in batch
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=pad_idx)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=pad_idx)
    
    return src_padded, tgt_padded


class TokenizedDataset(Dataset):
    """PyTorch Dataset wrapper for tokenized data."""
    
    def __init__(self, data, pad_idx=1):
        """
        Args:
            data: List of (src_indices, tgt_indices) tuples
            pad_idx: Index of padding token
        """
        self.data = data
        self.pad_idx = pad_idx
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        src, tgt = self.data[idx]
        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)


class VocabWrapper:
    """Wrapper around vocab dict to support lookup_token() method."""
    
    def __init__(self, vocab_dict):
        """
        Args:
            vocab_dict: Dict mapping token string -> idx
        """
        self.vocab = vocab_dict
        # Create reverse mapping: idx -> token
        self.idx_to_token = {v: k for k, v in vocab_dict.items()}
    
    def lookup_token(self, idx):
        """Convert token index to string."""
        return self.idx_to_token.get(idx, "<unk>")

    def token_to_id(self, token):
        """Convert token string to index."""
        return self.vocab.get(token, self.vocab.get("<unk>"))

    def __getitem__(self, token):
        return self.vocab[token]

    def __len__(self):
        return len(self.vocab)


class Multi30kDataset:
    def __init__(self, split='train'):
        """
        Loads the Multi30k dataset and prepares tokenizers.
        """
        self.split = split
        # Load dataset from Hugging Face
        # https://huggingface.co/datasets/bentrevett/multi30k
        # TODO: Load dataset, load spacy tokenizers for de and en
        self.dataset = load_dataset("bentrevett/multi30k")[split]
        
        # Load spacy tokenizers
        self.spacy_de = spacy.load("de_core_news_sm")
        self.spacy_en = spacy.load("en_core_web_sm")

        # Special tokens
        self.special_tokens = ["<unk>", "<pad>", "<sos>", "<eos>"]

        # Initialize vocab placeholders
        self.src_vocab = None
        self.tgt_vocab = None
    
    def tokenize_de(self, text):
        """
        Tokenizes German text using spacy.

        Args:
            text (str): The German sentence to tokenize.

        Returns:
            list[str]: A list of token strings.
        """ 
        return [tok.text for tok in self.spacy_de(text)]
    
    def tokenize_en(self, text):
        return [tok.text.lower() for tok in self.spacy_en(text)]

    def build_vocab(self, min_freq=2):
        """
        Builds the vocabulary mapping for src (de) and tgt (en), including:
        <unk>, <pad>, <sos>, <eos>
        """
        # TODO: Create the vocabulary dictionaries or torchtext Vocab equivalent
        de_counter = Counter()
        en_counter = Counter()
        for sample in self.dataset:
            de_tokens = self.tokenize_de(sample["de"])
            en_tokens = self.tokenize_en(sample["en"])

            de_counter.update(de_tokens)
            en_counter.update(en_tokens)

        # Special tokens
        self.special_tokens = ["<unk>", "<pad>", "<sos>", "<eos>"]

        # Initialize vocab
        self.src_vocab = {tok: idx for idx, tok in enumerate(self.special_tokens)}
        self.tgt_vocab = {tok: idx for idx, tok in enumerate(self.special_tokens)}

        # Add German words
        for word, freq in de_counter.items():
            if freq >= min_freq and word not in self.src_vocab:
                self.src_vocab[word] = len(self.src_vocab)

        # Add English words
        for word, freq in en_counter.items():
            if freq >= min_freq and word not in self.tgt_vocab:
                self.tgt_vocab[word] = len(self.tgt_vocab)
                
                
    def process_data(self):
        """
        Convert English and German sentences into integer token lists using
        spacy and the defined vocabulary. 
        """
        # TODO: Tokenize and convert words to indices
        processed_data = []

        for sample in self.dataset:
            # Tokenize
            de_tokens = self.tokenize_de(sample["de"])
            en_tokens = self.tokenize_en(sample["en"])

            # Convert to indices
            src_indices = [self.src_vocab["<sos>"]] + \
                        [self.src_vocab.get(tok, self.src_vocab["<unk>"]) for tok in de_tokens] + \
                        [self.src_vocab["<eos>"]]

            tgt_indices = [self.tgt_vocab["<sos>"]] + \
                        [self.tgt_vocab.get(tok, self.tgt_vocab["<unk>"]) for tok in en_tokens] + \
                        [self.tgt_vocab["<eos>"]]

            processed_data.append((src_indices, tgt_indices))

        return processed_data


# ══════════════════════════════════════════════════════════════════════
# INITIALIZE DATASETS AND VOCABULARIES
# ══════════════════════════════════════════════════════════════════════

# Create dataset instances for each split
_train_dataset_obj = Multi30kDataset(split='train')
_val_dataset_obj = Multi30kDataset(split='validation')
_test_dataset_obj = Multi30kDataset(split='test')

# Build vocabularies on training data
_train_dataset_obj.build_vocab(min_freq=2)

# Use the same vocabularies for all splits
_val_dataset_obj.src_vocab = _train_dataset_obj.src_vocab
_val_dataset_obj.tgt_vocab = _train_dataset_obj.tgt_vocab
_test_dataset_obj.src_vocab = _train_dataset_obj.src_vocab
_test_dataset_obj.tgt_vocab = _train_dataset_obj.tgt_vocab

# Process data for all splits
train_data = _train_dataset_obj.process_data()
val_data = _val_dataset_obj.process_data()
test_data = _test_dataset_obj.process_data()

# Create PyTorch datasets
train_dataset = TokenizedDataset(train_data, pad_idx=1)
val_dataset = TokenizedDataset(val_data, pad_idx=1)
test_dataset = TokenizedDataset(test_data, pad_idx=1)

# Export vocabularies with wrapper class for compatibility with evaluate_bleu
src_vocab = VocabWrapper(_train_dataset_obj.src_vocab)
tgt_vocab = VocabWrapper(_train_dataset_obj.tgt_vocab)