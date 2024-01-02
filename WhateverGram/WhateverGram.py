from typing import List

import torch
import torch.nn.functional as F

from DataLoader import DataLoader

class WhateverGram():
    def __init__(self, dir_path:str, ngrams:int, add_title:bool) -> None:
        loader = DataLoader(dir_path=dir_path)
        self.txts = loader.read_texts_from_folder(add_title=add_title)
        all_txt_min_len = self._min_characters()

        if all_txt_min_len <= ngrams:
            raise ValueError("Please select lower ngram value.")

        all_characters = sorted(list(set("".join(self.txts))))
        self.total_character_len = len(all_characters)
        
        # lookup tables
        self.stoi = {s:i+1 for i, s in enumerate(all_characters)}
        self.stoi["<S>"] = 0
        self.stoi["<E>"] = self.total_character_len+1
        self.itos = {i:s for s, i in self.stoi.items()}

        # empty tensors
        self.N = self._populate_empty_array()

        # Probability tensors
        self.P = self._calculate_probabilities()

    def _min_characters(self) -> int:
        min_val = 9999999

        for txt in self.txts:
            txt_len = len(txt) 
            if txt_len < min_val:
                min_val = txt_len

        return min_val
     
    def _populate_empty_array(self) -> List[float]:
        N = torch.zeros((len(self.itos), len(self.itos)))
        
        for txt_file in self.txts:
            chs = ["<S>"] + list(txt_file) + ["<E>"]

            for ch1, ch2 in zip(chs, chs[1:]):
                ix1 = self.stoi[ch1]
                ix2 = self.stoi[ch2]
                N[ix1, ix2] += 1
        
        return N

    def _calculate_probabilities(self) -> List[float]:
        P = self.N
        P = P/P.sum(1, keepdim=True)
        return P
    
    def generate_text(self) -> str:
        ix = 0
        all_text = ""
        while (True):
            p = self.P[ix]

            ix = torch.multinomial(p, num_samples=1, replacement=True).item()

            current_character = self.itos[ix]
            all_text += current_character
            
            if ix == (self.total_character_len + 1):
                break
        
        return all_text[:-3]