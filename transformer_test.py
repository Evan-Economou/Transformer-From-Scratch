import transformer
import pytest
import torch


def test_process_raw_data():
    tokenizer = transformer.Tokenizer("hello i am, being : uh, tested; yeah!")
    assert tokenizer.process_raw_data(tokenizer.raw_data) == "hello i am , being : uh , tested ; yeah !"

def test_encode():
    tokenizer = transformer.Tokenizer("aah im testing my shit : ''''")
    assert tokenizer.encode("aah im testing my shit : ''''").shape == torch.Size([5]) 


    
    