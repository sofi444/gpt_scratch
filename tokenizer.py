import torch



class CharacterTokenizer:
    def __init__(self, data):
        self.vocab = sorted(list(set(data)))
        self.vocab_size = len(self.vocab)
        self.char_to_int = {
            char:i for i,char in enumerate(self.vocab)
        }
        self.int_to_char = {
            i:char for i,char in enumerate(self.vocab)
        }

    def encode(self, text) -> list[int]:
        encoded = [self.char_to_int[char] for char in text]
        return torch.tensor(encoded, dtype=torch.long)

    def decode(self, encoded) -> str:
        if isinstance(encoded, torch.Tensor):
            encoded = encoded.tolist()
        return ''.join([self.int_to_char[i] for i in encoded])
