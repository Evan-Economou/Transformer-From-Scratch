from dataclasses import dataclass
import torch
import torch.nn as nn
from jaxtyping import Float, Int
import unicodedata
from collections import Counter

class Tokenizer():
    def __init__(self, raw_data: str):
        """! @brief Initializes the Tokenizer with raw text data, processing and tokenizing it, and builds the vocabulary
        @param raw_data The raw text data to be tokenized and used for building the vocabulary"""
        self.raw_data = raw_data 
        self.tokenized_text = self.tokenize(raw_data, process=True)

        vocab_counts: Counter[str] = Counter(self.tokenized_text).most_common()

        self.vocab_size: int = len(vocab_counts)

        #vocab inverse takes integers and turns them back into words
        self.vocab_inverse: list[str] = [token for token, _ in vocab_counts]

        #vocab takes words and turns them into integers
        self.vocab: dict[str: int] = dict(zip(self.vocab_inverse,range(0,self.vocab_size)))

        print(self.vocab)
        print(self.vocab_inverse)
        
    def process_raw_data(self, text: str, 
                        allowed_punctuation: str = "-.,;:!?()\"" + "".join(str(x) for x in range(10)),
                        punctuation_convert: dict[str,str] = {'—': '-'}
                        ) -> str:
        """! @brief Processes raw text data by removing unrecognized punctuation, adding spaces, and removing excess whitespace
        @param text The raw text data to be processed
        @param allowed_punctuation A string of punctuation characters that should be preserved from the raw text
        @param punctuation_convert A dictionary mapping punctuation characters to their replacements
        @return The processed text with punctuation removed, spaces added, and excess whitespace removed 
        """
        for char, replacement in punctuation_convert.items():
            text = text.replace(char, replacement)
              
        text = '\n'.join(
                    line 
                    for line in text.split('\n')
                    if '.jpg' not in line
                )
        
        text = unicodedata.normalize('NFKD', text)

        # Encode to ASCII bytes, then decode back to string, ignoring errors
        text = text.encode('ascii', 'ignore').decode('ascii')

        # remove newlines and tabs
        text = text.replace('\n', ' ').replace('\t', ' ')

        for char in allowed_punctuation:
            text = text.replace(char, f' {char} ')
              
        text = text.strip()

        # remove multiple connected spaces
        while '  ' in text:
            text = text.replace('  ', ' ')

        text = ''.join((char if (char.isalnum() or char in allowed_punctuation or char == ' ') else ' ') for char in text)
        
        text = text.lower()

        text = text.strip()

        return text
    
    def tokenize(self, 
        text: str,
        process: bool = False,
    ) -> str:
        if process:
            text = self.process_raw_data(text)
        tokenized_text = text.split(' ')

        return tokenized_text
    
    def add_data(self, new_data: str):
        """! @brief Adds new raw text data to the Tokenizer, processing and tokenizing it, and updates the vocabulary
        @param new_data The new raw text data to be added to the Tokenizer"""
        self.raw_data += " " + new_data
        tokenized_text = self.process_raw_data(self.raw_data).split(' ')

        vocab_counts: Counter[str] = Counter(tokenized_text).most_common()

        #update vocab size, vocab inverse, and vocab
        self.vocab_size: int = len(vocab_counts)
        self.vocab_inverse: list[str] = [token for token, _ in vocab_counts]
        self.vocab: dict[str: int] = dict(zip(self.vocab_inverse,range(0,self.vocab_size)))

    #should turn string of words into numbers
    def encode(self, data: str) -> Int[torch.Tensor, "n_context"]:
        data = self.process_raw_data(data).split(' ')
        return torch.tensor([self.vocab[word] for word in data], dtype=torch.long)
    
    #should turn numbers into a string of words
    def decode(self, tokens: Int[torch.Tensor, "n_context"]) -> str:
        return " ".join([self.vocab_inverse[token] for token in tokens])

# %%
@dataclass
class Config():
    d_model: int
    d_vocab: int
    d_hidden: int
    tokenizer: Tokenizer

# %%
class MLP(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.linear1 = nn.Linear(config.d_model, config.d_hidden)
        self.linear2 = nn.Linear(config.d_hidden, config.d_model)
        

    def forward(self, x: Float[torch.Tensor, "n_context d_model"]) -> Float[torch.Tensor, "n_context d_model"]:
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x

# %%
class AttentionHead(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.W_qk = nn.Linear(config.d_model, config.d_model)
        self.W_vo = nn.Linear(config.d_model, config.d_model)
        self.softmax = nn.Softmax(dim=-1)
        

    def create_mask(self, n_c: int) -> torch.Tensor:
        mask: Float[torch.Tensor, "n_context n_context"] = torch.triu(-1 * torch.inf * torch.ones(n_c, n_c), diagonal=1)
        return mask

    def forward(self, x: Float[torch.Tensor, "n_context d_model"]) -> Float[torch.Tensor, "n_context d_model"]:
        #create mask, with size n_c x n_c
        mask = self.create_mask(x.shape[0])

        #compute attention scores
        # A = softmax((X @ W_qk @ X^T) + M) @ X @ W_vo
        A = self.softmax((self.W_qk(x)) @ x.transpose(0, -1) + mask) @ self.W_vo(x)
        return A

# %%
class TransformerBlock(torch.nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.attention_head = AttentionHead(config)
        self.mlp = MLP(config)

    def forward(self, x: Float[torch.Tensor, "n_context d_model"]) -> Float[torch.Tensor, "n_context d_model"]:
        return x + self.attention_head(x) + self.mlp(x)

# %%
class Transformer(torch.nn.Module):
    def __init__(self, num_blocks: int, config: Config, positional_embedding: bool = True):
        super().__init__()
        self.config = config
        self.embedding = nn.Linear(config.d_vocab, config.d_model)
        if(positional_embedding):
            self.positional_embedding = nn.Linear(config.d_vocab, config.d_model)
        else:
            self.positional_embedding = None
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(num_blocks)])
        self.softmax = nn.Softmax(dim=-1)
        

    def forward(self, x: Int[torch.Tensor, "n_context"]) -> Float[torch.Tensor, "n_context d_vocab"]:
        x_onehot = torch.zeros(x.shape[0], self.config.d_vocab)
        for i, token in enumerate(x):
            x_onehot[i, token] = 1.0
        x = self.embedding(x_onehot)
        
        if self.positional_embedding is not None:
            pos_indices = torch.arange(x.shape[0])
            pos_onehot = torch.zeros(x.shape[0], self.config.d_vocab)
            for i, pos in enumerate(pos_indices):
                pos_onehot[i, pos] = 1.0
            x += self.positional_embedding(pos_onehot)
            
        # print(x.shape)
        for block in self.blocks:
            x = block.forward(x)
        x = (x @ self.embedding.weight)
        return x
    
    def generate_output(self, x:str, n_tokens: int = 100) -> str:
        output_str = ""
        for _ in range(n_tokens):
            token_probs = self.softmax(self.forward(self.config.tokenizer.encode(x))[-1,:])
            idx = torch.multinomial(token_probs[0:100], num_samples=1)
            x += " " + self.config.tokenizer.decode(idx.unsqueeze(0))
            output_str += " " + self.config.tokenizer.decode(idx.unsqueeze(0))
            if x.__contains__("."):
                break
            
        return output_str