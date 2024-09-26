from model import HangmanTransformer
import torch

class Optimus:
    def __init__(self, model_dict_pth, vocab, vocab_size, embed_size, block_size, n_head, n_layer, output_size, dropout, device):
        self.model = model = HangmanTransformer(vocab_size, embed_size, block_size, n_head, n_layer, output_size, dropout, device)
        self.model.load_state_dict(torch.load(model_dict_pth, map_location=device))
        self.model.eval()
        self.model = self.model.to(device)
        self.device = device

        tokens=vocab
        self.encode_dict = dict([(x, idx) for idx, x in enumerate(tokens)])
        self.max_len = block_size

        self.order = ['e', 'i', 'a', 'n', 'o', 'r', 's', 't', 'l', 'c', 'u', 'd', 'p', 'm', 'h', 'g', 'y', 'b', 'f', 'v', 'k', 'w', 'z', 'x', 'q', 'j']

        self.tried = []

    def reset(self):
        self.tried = []

    def encode(self, word):
        word = word.strip()
        encoding = [self.encode_dict[x] for x in word]

        if len(encoding) > self.max_len:
            encoding = encoding[:self.max_len]
        else:
            while len(encoding) < self.max_len:
                encoding.append(0)

        return encoding
    
    def fill_word(self, word):
        inp = self.encode(word)
        inp = torch.tensor(inp).unsqueeze(0).to(self.device)
        out = self.model(inp)[0].tolist()
        mapped = [(a,b) for a,b in zip("abcdefghijklmnopqrstuvwxyz", out)]
        return sorted(mapped, key=lambda tup: -tup[1])
    
    def guess_probs(self, word):
        inp = self.encode(word)
        inp = torch.tensor(inp).unsqueeze(0).to(self.device)
        out = self.model(inp)[0].tolist()
        
        return out
          
    def guess(self, word):
        order = self.fill_word(word)

        for g, p in order:
            if (g not in self.tried):
                print(g)
                self.tried.append(g)
                return g
        
        return 'e'