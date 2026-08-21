"""
Headline Generator
"""
import os
import sys
import numpy as np 
import pandas as pd
import re
import random
import pickle
import torch
import torch.optim as optim
#from torchtext.data.metrics import bleu_score
#from rouge import Rouge
from transformers import T5ForConditionalGeneration, T5Tokenizer
"""
Headline Generator
"""
import os
import sys
import numpy as np 
import pandas as pd
import re
import random
import pickle
import torch
import torch.optim as optim
from transformers import T5ForConditionalGeneration, T5Tokenizer

class headline_gen():
    def __init__(self, device='cpu', path=None):
        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'
        self.device = device
        
        self.model = T5ForConditionalGeneration.from_pretrained('t5-base').to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained('t5-base', legacy=False)
        if path is not None:
            if os.path.exists(path):
                self.model.load_state_dict(torch.load(path, map_location=torch.device(self.device)))
            else:
                print(f"Warning: Checkpoint at {path} not found. Using pre-trained t5-base weights.")
            
    def generate_batch(self, data):
        output = random.sample(data, 4)

        inp, label = [], []
        for dat in output:
            inp.append(dat[0])
            label.append(dat[1])

        return inp, label
    
    def fit(self, art, head):
        assert len(art) == len(head)
        data = []
        for i, j in zip(art, head):
            data.append([i, j])
    
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.00001)

        scalar = 0
        for i in range(3000):
            self.model.train()
            inp, label = self.generate_batch(data)
            input = self.tokenizer(text=inp, text_target=label, padding=True, return_tensors='pt', max_length=600, truncation=True)
            outputs = self.model(input_ids=input['input_ids'].to(self.device), labels=input['labels'].to(self.device))
            loss = outputs[0]
            
            scalar += loss.item()
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            del outputs
            del input
            del inp
            del label
            if (i + 1) % 50 == 0:
                print('iteration={}, training loss={}'.format(i + 1, scalar / (4 * 50)))
                scalar = 0

            loss.backward()
            self.optimizer.step()
                
    # def predict(self, articles):
    #     with torch.no_grad():
    #         preds = []
    #         for line in articles:
    #             inp = [line]
    #             input = self.tokenizer(inp, padding=True, return_tensors='pt', max_length=512, truncation=True)

    #             output = self.model.generate(input_ids=input['input_ids'].to(self.device),
    #                                        num_beams=5, early_stopping=True, max_length=35)
    #             out = self.tokenizer.batch_decode(output, skip_special_tokens=True)
    #             if self.device == 'cuda':
    #                 torch.cuda.empty_cache()
    #             out[0] = re.sub(r"<\S+", "", out[0])
    #             temp = out[0].split()
    #             if len(temp) > 20:
    #                 temp = temp[:20]
    #             final = ' '.join(x for x in temp)
    #             final = final.lstrip().rstrip()
                
    #             preds.append(final)
             
    #         return preds
    
    def predict(self, articles):
        with torch.no_grad():
            inputs = [
            "summarize: " + re.sub(r"\s+", " ", str(article)).strip()
            for article in articles
        ]

        encoded = self.tokenizer(
            inputs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        outputs = self.model.generate(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            num_beams=4,
            max_new_tokens=30,
            min_new_tokens=6,
            no_repeat_ngram_size=3,
            early_stopping=True
        )

        predictions = self.tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True
        )

        results = []

        for text in predictions:
            text = re.sub(r"\s+", " ", text).strip()
            text = re.sub(r"<[^>]+>", "", text)

            words = text.split()
            if len(words) > 20:
                text = " ".join(words[:20])

            results.append(text)

        return results

