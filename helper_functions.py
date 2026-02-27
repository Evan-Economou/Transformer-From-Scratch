import transformer
import matplotlib as plt
import configparser
import requests
from pathlib import Path
from jaxtyping import Float, Int
import torch
import torch.nn as nn

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

def load_config(config_path: Path|str) -> tuple[int, int, transformer.Config]:
    config_parser = configparser.ConfigParser()
    config_parser.read(config_path)

    #load model information
    d_model = config_parser.getint('MODEL', 'd_model')
    d_hidden = config_parser.getint('MODEL', 'd_hidden')

    #load data information and create tokenizer parsing it
    tokenizer_data_path = config_parser.get('DATA', 'tokenizer_data_path')
    with open(tokenizer_data_path, 'r', encoding='utf-8') as file:
        tokenizer_data = file.read()
    tokenizer = transformer.Tokenizer(raw_data=tokenizer_data)

    #load training information
    batch_size = config_parser.getint('TRAINING', 'batch_size')
    positional_embedding = config_parser.getboolean('TRAINING', 'positional_embedding')

    return positional_embedding, batch_size, tokenizer.Config(d_model=d_model, d_vocab=tokenizer.vocab_size, d_hidden=d_hidden, tokenizer=tokenizer)

def train_model(
    model: transformer.Transformer,
    raw_data: str,
    loss_fn: torch.nn.CrossEntropyLoss = nn.CrossEntropyLoss(),
    lr: Float = 1e-3,
    batch_size: Int = None
    ):
    optimizer: torch.optim.SGD = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    training_data = model.config.tokenizer.encode(raw_data)
    n = training_data.shape[0]
    loss_history = []
    
    if batch_size is None: # if no batch size is provided, just use all the data
        batch_size = model.config.d_vocab
    
    # otherwise break the data into batches with size batch_size
    n_batch = 0
    
    for i in range(0, n - batch_size, batch_size):
        batch_tokens = training_data[i:i+batch_size]
        targets = training_data[i+1:i+batch_size+1]
        
        outputs = model(batch_tokens)
        loss = loss_fn(outputs[:-1, :], targets[:-1])
        print(f"Batch {n_batch}: Loss = {loss.item():.4f}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        n_batch += 1
        
    return loss_history

def plot_loss(loss_history: list[float], save_path: str = "loss_plot.png"):
    """Plot training loss as a function of batches, display it and save it"""
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(loss_history)), loss_history, 'b-', linewidth=2)
    plt.xlabel('Batch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss over Batches', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Loss plot saved to {save_path}")
    plt.show()

def conversation_loop(model : transformer.Transformer):
    while True:
        try:
            input_str = input("Enter a string to generate output (or 'exit' to quit): ")
            if input_str.lower() == 'exit':
                break
            output = model.generate_output(input_str)
            print(f"Generated output: {output}")
        except Exception as e:
            print(f"Error generating output: {e}. Please try again.")