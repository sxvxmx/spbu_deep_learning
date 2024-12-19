from typing import Tuple
import torch
from torch import nn
from torch.utils.data import Dataset

class Tokenizer:
    def __init__(self, cut_text, max_len: int = 128):
        self.text = cut_text
        self.max_len = max_len
        self.specials = ['<pad>', '<bos>', '<eos>']
        self.int2char = dict(enumerate(tuple(set(cut_text))))
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}
        self._add_special("<pad>")
        self._add_special('<bos>')
        self._add_special('<eos>')
    
    def _add_special(self, symbol) -> None:
        sym_num = len(self.char2int)
        self.char2int[symbol] = sym_num
        self.int2char[sym_num] = symbol

    @property
    def vocab_size(self):
        return len(self.int2char) 
        
    def decode_symbol(self, el):
        return self.int2char[el]
        
    def encode_symbol(self, el):
        return self.char2int[el]
        
    def str_to_idx(self, chars):
        return [self.char2int[sym] for sym in chars] 

    def idx_to_str(self, idx):
        return [self.int2char[toc] for toc in idx]

    def encode(self, chars, eos=True):
        if eos:
            chars = ['<bos>'] + list(chars) + ['<eos>']
        else:
            chars = ['<bos>'] + list(chars)
        return self.str_to_idx(chars)

    def decode(self, idx):
        chars = self.idx_to_str(idx)
        return "".join(chars) 
    

class JokesDataset(Dataset):
    def __init__(self, tokenizer, cut_text, max_len: int = 512):
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.cut_text = cut_text
        self.pad_index = self.tokenizer.encode_symbol("<pad>")

    def __len__(self):
        return len(self.cut_text)
        
    def __getitem__(self, item):
        encoded = self.tokenizer.encode(self.cut_text[item])[:self.max_len]
        padded = torch.full((self.max_len, ), self.pad_index, dtype=torch.long)
        padded[:len(encoded)] = torch.tensor(encoded)
        return padded, len(encoded)
    
def training_step(
    model,
    train_batch: Tuple[torch.Tensor, torch.Tensor],
    vocab_size: int,
    criterion: nn.Module,
    optimizer,
    device="cpu"
) -> torch.Tensor:
    inputs, lengths = train_batch
    optimizer.zero_grad()
    batch_size, seq_len = inputs.shape
    targets = inputs[:, 1:]
    outputs, _ = model(inputs[:, :-1], lengths-1)
    tar = []
    for i, item in enumerate(lengths):
        tar.append(targets[i][:max(lengths)-1])
    targets = torch.stack(tar)
    targets = targets.reshape(len(inputs), -1)
    logits = outputs.view(-1, vocab_size) 
    targets = targets.view(-1)
    loss = criterion(logits, targets)
    loss.backward()
    optimizer.step()
    return loss.item()