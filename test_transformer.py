from transformer import Transformer, Config, Tokenizer, AttentionHead
import pytest
import torch


@pytest.fixture
def small_model():
    torch.manual_seed(0)
    tokenizer = Tokenizer("hello world world")
    config = Config(d_model=16,d_vocab=tokenizer.vocab_size,d_hidden=32,tokenizer=tokenizer)
    model = Transformer(num_blocks=2, config=config, positional_embedding=True)
    return model

def test_forward_shape(small_model):
    x = small_model.config.tokenizer.encode("hello world")
    out = small_model(x)
    assert out.shape == (len(x), small_model.config.d_vocab)
    

def test_causal_mask():
    tokenizer = Tokenizer("a b c")
    config = Config(d_model=8, d_vocab=3, d_hidden=16, tokenizer=tokenizer)
    head = AttentionHead(config)
    mask = head.create_mask(4)
    
    assert torch.isinf(mask[0,1])
    assert mask[0,1] < 0
    
    assert mask[0,0] == 0
    assert mask[3,0] == 0
    
    
def test_attention_rows_sum_to_one():
    tokenizer = Tokenizer("a b c d")
    config = Config(d_model=8, d_vocab=4, d_hidden=16, tokenizer=tokenizer)
    head = AttentionHead(config)
    
    x = torch.randn(5,8)
    mask = head.create_mask(5)
    
    scores = (head.W_qk(x)) @ x.transpose(0,1) + mask
    probs = head.softmax(scores)
    
    row_sums = probs.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)
    

def test_no_future_leakage(small_model):
    tokenizer = small_model.config.tokenizer
    x1 = tokenizer.encode("hello world hello")
    x2 = x1.clone()
    
    x2[-1] = 0
    
    out1 = small_model(x1)
    out2 = small_model(x2)
    
    assert torch.allclose(out1[:-1], out2[:-1], atol=1e-5)
    
def test_gradients_exist(small_model):
    x = small_model.config.tokenizer.encode("hello world")
    out = small_model(x)
    loss = out.sum()
    loss.backward()
    
    grads = [p.grad for p in small_model.parameters() if p.requires_grad]
    
    assert all(g is not None for g in grads)
    assert all(not torch.isnan(g).any() for g in grads)
    
    
def test_positional_embedding_toggle():
    tokenizer = Tokenizer("a b c")  
    config = Config(d_model=8, d_vocab=3, d_hidden=16, tokenizer=tokenizer)
    model_with = Transformer(1, config, positional_embedding=True)
    model_without = Transformer(1, config, positional_embedding=False)
    x = tokenizer.encode("a b c")
    out1 = model_with(x)
    out2 = model_without(x)
    assert not torch.allclose(out1, out2)
    
def test_deterministic_forward():
    torch.manual_seed(42)
    tokenizer = Tokenizer("a b c")
    config = Config(d_model=8, d_vocab=3, d_hidden=16, tokenizer=tokenizer)
    model = Transformer(1, config)
    x = tokenizer.encode("a b c")
    out1 = model(x)
    out2 = model(x)
    assert torch.allclose(out1, out2)
    
    
def test_overfit_tiny_batch():
    torch.manual_seed(0)
    
    text = "hello hello hello hello"
    tokenizer = Tokenizer(text)
    config = Config(d_model=16, d_vocab=tokenizer.vocab_size, d_hidden=32, tokenizer=tokenizer)
    model = Transformer(1, config)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    tokens = tokenizer.encode(text)
    initial_loss = None
    
    for _ in range(50):
        outputs = model(tokens[:-1])
        targets = tokens[1:]
        loss = loss_fn(outputs, targets)
        if initial_loss in None:
            initial_loss = loss.item()
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    assert loss.item() < initial_loss
    