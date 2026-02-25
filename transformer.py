# %%
from dataclasses import dataclass
from pathlib import Path
import torch
import torch.nn as nn
from jaxtyping import Float, Int
import requests
import unicodedata
from collections import Counter


# %% [markdown]
# # Training Data

# %%
def get_gutenberg_book(
	id: int|None = 84,
	data_temp: Path|str = "../../../../data/gutenberg_data",
	remove_gutenberg_meta: bool = True,
) -> str:
	
	data_temp = Path(data_temp)
	data_temp.mkdir(parents=True, exist_ok=True)
	
	url: str = f"https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"
	data_path: Path = Path(data_temp) / f"{id}.txt"
	data: str
	# read from cache if it exists
	if data_path.exists():
		with open(data_path, 'r', encoding='utf-8') as file:
			data = file.read()
	else:
		# download if it doesn't exist
		response = requests.get(url)
		response.raise_for_status()  # Ensure that the download was successful
		data = response.text

		# save to cache
		with open(data_path, 'w', encoding='utf-8') as file:
			file.write(data)

	# remove header/footer
	if remove_gutenberg_meta:
		data = '***'.join(data.split('***')[2:])
		data = '***'.join(data.split('***')[:-1])
	
	return data

def get_many_books(
		ids: list[int],
		data_temp: Path|str = "../data/gutenberg_data",
	) -> list[list[str]]:
	
	data: list[str] = []
	for id in ids:
		print(f"Getting book {id}...")
		item: str = get_gutenberg_book(id, data_temp)
		print(f"\t{len(item)} characters read")
		data.append(item)
	
	return data

# %% [markdown]
# # Model Definition

# %%
class Tokenizer():
    def __init__(self, raw_data: str):
        """! @brief Initializes the Tokenizer with raw text data, processing and tokenizing it, and builds the vocabulary
        @param raw_data The raw text data to be tokenized and used for building the vocabulary"""
        self.raw_data = raw_data 
        tokenized_text = self.process_raw_data(raw_data).split(' ')

        vocab_counts: Counter[str] = Counter(tokenized_text).most_common()

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
        """! @brief Processes raw text data by removing punctuation, adding spaces, and removing excess whitespace
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

        # remove multiple spaces
        while '  ' in text:
            text = text.replace('  ', ' ')

        text = ''.join((char if (char.isalnum() or char in allowed_punctuation or char == ' ') else ' ') for char in text)
        
        text = text.lower()

        text = text.strip()

        return text
    
    #I dont think we need this function
    # def tokenize(self, 
    #     text: str,
    #     process: bool = False,
    # ) -> str:
    #     if process:
    #         text = self.process_raw_data(text)
    #     tokenized_text = text.split(' ')

    #     return tokenized_text
    
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
    def __init__(self, num_blocks: int, config: Config):
        super().__init__()
        self.config = config
        self.embedding = nn.Linear(config.d_vocab, config.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(num_blocks)])
        self.softmax = nn.Softmax(dim=-1)
        

    def forward(self, x: Int[torch.Tensor, "n_context"]) -> Float[torch.Tensor, "n_context d_vocab"]:
        x_onehot = torch.zeros(x.shape[0], self.config.d_vocab)
        for i, token in enumerate(x):
            x_onehot[i, token] = 1.0
        x = self.embedding(x_onehot)
        # print(x.shape)
        for block in self.blocks:
            x = block.forward(x)
        x = (x @ self.embedding.weight)
        return x
    
    def generate_output(self, x:str, n_tokens: int = 10) -> str:
        output_str = ""
        for _ in range(n_tokens):
            last_logit = self.forward(self.config.tokenizer.encode(x))[-1,:]
            idx = torch.argmax(last_logit)
            x += " " + self.config.tokenizer.decode(idx.unsqueeze(0))
            output_str += " " + self.config.tokenizer.decode(idx.unsqueeze(0))
            
        return output_str
    

# %% [markdown]
# # Tests

# %%
# Attention head test
x: Float[torch.Tensor, "n_context d_model"] = torch.ones(5, 16)
tokenizer = Tokenizer(raw_data="Aaaah Im Tokenizing It")

print(tokenizer.encode("it im tokenizing"))
print(tokenizer.decode(torch.tensor([1,1,1],dtype=torch.int)))

config = Config(d_model=16, d_vocab=1000, d_hidden=64,tokenizer=tokenizer)
attention_head: AttentionHead = AttentionHead(config)
output: Float[torch.Tensor, "n_context d_model"] = attention_head.forward(x)
print(output.shape)

# %%
# Test the whole thing
config = Config(d_model=16, d_vocab=1000, d_hidden=64,tokenizer=tokenizer)
transformer = Transformer(num_blocks=2, config=config)
x = torch.tensor([0, 2, 1, 3], dtype=torch.int)
y: Float[torch.Tensor, "vocab n_context"] = transformer(x)
print(y.shape)
print(y)
print(x)

# %% [markdown]
# # Training Loop

# %%
def train_model(
    model: Transformer,
    raw_data: str,
    loss_fn: torch.nn.CrossEntropyLoss = nn.CrossEntropyLoss(),
    lr: Float = 1e-3,
    epochs: Int = 1
    ):
    optimizer: torch.optim.SGD = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    training_data = model.config.tokenizer.encode(raw_data)
    # CrossEntropyLoss expects integer class indices, not one-hot vectors
    n = training_data.shape[0]
    true_values = training_data[1:n]  # Next token predictions
    
    for epoch in range(epochs):
        outputs = model(training_data)
        # outputs shape: (n_context, d_vocab)
        # true_values shape: (n_context-1,) with integer class indices
        loss = loss_fn(outputs[0:n-1, :], true_values)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'Loss for epoch {epoch}: {loss.item():.4f}')


def main():
    decision = input("Do you want to train a new model? (y/n): ")
    if decision.lower() == 'n': #assume a model file exists and try to load it
        while True:
            try:
                model_path = input("Enter the path to the saved model: ")
                model = torch.load(model_path)
                print("Model loaded successfully.")
                break
            except Exception as e:
                print(f"Error loading model: {e}. Please try again.")

        while True:
            try:
                input_str = input("Enter a string to generate output (or 'exit' to quit): ")
                if input_str.lower() == 'exit':
                    print("Exiting the program.")
                    break
                output = model.generate_output(input_str)
                print(f"Generated output: {output}")
            except Exception as e:
                print(f"Error generating output: {e}. Please try again.")
    elif (decision.lower() == 'y'): #user wants to load data and train a new model
        raw_data = get_gutenberg_book()[0:10000]
        tokenizer = Tokenizer(raw_data)
        config = Config(d_model=16, d_vocab=tokenizer.vocab_size, d_hidden=64,tokenizer=tokenizer)
        model = Transformer(num_blocks=2, config=config)
        train_model(model, raw_data, epochs = 50)
        torch.save(model, "transformer_model.pt")
    else: #this is the case where I am just messing around and making sure the model works
        prompt = input("Running input through an untrained model, enter prompt: ")
        tokenizer = Tokenizer(raw_data="Aaaah Im Tokenizing It")
        config = Config(d_model=16, d_vocab=tokenizer.vocab_size, d_hidden=64,tokenizer=tokenizer)
        model = Transformer(num_blocks=2, config=config)
        output = model.generate_output(prompt)
        print(f"Generated output: {output}")

if __name__ == "__main__":
    main()