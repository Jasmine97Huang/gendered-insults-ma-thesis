import torch # pip install torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertConfig 
from transformers import AdamW, BertForMaskedLM
import pandas as pd 
import numpy as np
import json
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from tqdm import tqdm, trange
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LyricsMLMDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings['input_ids'])

        
if __name__ == "__main__":
    with open(sys.argv[1], 'rb') as f:
        input_ids = np.load(f)
        labels = np.load(f)
        attention_mask = np.load(f)
    inputs = {}
    inputs['input_ids'] = torch.from_numpy(input_ids)
    inputs['labels'] = torch.from_numpy(labels)
    inputs['attention_mask'] = torch.from_numpy(attention_mask)

    dataset = LyricsMLMDataset(inputs)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    model = BertForMaskedLM.from_pretrained("bert-base-uncased")
    model.to(device)

    model.train()
    optim = AdamW(model.parameters(), lr=1e-7)

    epochs = int(sys.argv[2])
    print(f"Finetuning for {epochs} epochs")

    for epoch in range(epochs):
        loop = tqdm(dataloader, leave = True)
        for batch in loop:
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

            loss=outputs.loss
            loss.backward()
            optim.step()

            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())


    model.save_pretrained('output/mlm_bert')
