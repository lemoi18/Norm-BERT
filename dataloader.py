# coding: utf-8

import os
import os.path as osp
import numpy as np
import pickle
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

# Set random seeds for reproducibility
np.random.seed(1)
torch.manual_seed(1)


class FilePathDataset(Dataset):
    def __init__(self, dataset_file, max_text_length=512, tokenizer="NbAiLab/nb-bert-large"):
        with open(dataset_file, 'rb') as handle:
            self.data = pickle.load(handle)
        
        self.max_text_length = max_text_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        unnormalized_input_ids = torch.tensor(item['unnormalized_tokens'], dtype=torch.long)
        normalized_input_ids = torch.tensor(item['normalized_tokens'], dtype=torch.long)
        
        # Pad sequences to max length
        unnormalized_input_ids = torch.nn.functional.pad(unnormalized_input_ids, (0, self.max_text_length - len(unnormalized_input_ids)), value=self.tokenizer.pad_token_id)
        normalized_input_ids = torch.nn.functional.pad(normalized_input_ids, (0, self.max_text_length - len(normalized_input_ids)), value=self.tokenizer.pad_token_id)

        # Get alignment map
        alignment_map = item['alignment']
        
        # Convert alignment map into tensors of matching indices
        alignment_tensor = torch.zeros((self.max_text_length, 2), dtype=torch.long)
        for unnorm_idx, norm_indices in alignment_map:
            if unnorm_idx < self.max_text_length:
                alignment_tensor[unnorm_idx, :len(norm_indices)] = torch.tensor(norm_indices[:2])  # Limiting to 2 alignments for simplicity

        return unnormalized_input_ids, normalized_input_ids, alignment_tensor



class Collater:
    def __init__(self, tokenizer="NbAiLab/nb-bert-large"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    def __call__(self, batch):
        unnormalized_tokens_list, normalized_tokens_list, alignment_list = zip(*batch)

        unnormalized_tokens_padded = torch.nn.utils.rnn.pad_sequence(
            unnormalized_tokens_list, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        normalized_tokens_padded = torch.nn.utils.rnn.pad_sequence(
            normalized_tokens_list, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        alignment_padded = torch.nn.utils.rnn.pad_sequence(
            alignment_list, batch_first=True, padding_value=-1
        )

        return unnormalized_tokens_padded, normalized_tokens_padded, alignment_padded


def build_dataloader(df,
                     validation=False,
                     batch_size=4,
                     num_workers=1,
                     collate_config={},
                     dataset_config={}):
    
    dataset = FilePathDataset(df, **dataset_config)
    collate_fn = Collater(**collate_config)
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=(not validation),
                             num_workers=num_workers,
                             drop_last=(not validation),
                             collate_fn=collate_fn,
                             pin_memory=True)
    
    return data_loader


# Sample debugging code for testing data loader functionality
# Sample debugging code for testing data loader functionality
if __name__ == "__main__":
    data_loader = build_dataloader('processed_text_with_alignment.pkl')
    tokenizer = AutoTokenizer.from_pretrained("NbAiLab/nb-bert-large")
    count = 0
    # Test block to examine a batch
    for batch in data_loader:
        unnormalized_tokens, normalized_tokens, alignment = batch

        # Display tensor shapes for verification
        print("Unnormalized tokens shape:", unnormalized_tokens.shape)
        print("Normalized tokens shape:", normalized_tokens.shape)
        print("Alignment tensor shape:", alignment.shape)

        # Decode and display for manual verification
        print("\nDecoded Sample:")
        print("Unnormalized tokens:", tokenizer.decode(unnormalized_tokens[0], skip_special_tokens=True))
        print("Normalized tokens:", tokenizer.decode(normalized_tokens[0], skip_special_tokens=True))
        print("Alignment (first 10 indices):", alignment[0][:10])
        count +=1
        
        # Break after displaying one batch
        if count == 10:
            break
