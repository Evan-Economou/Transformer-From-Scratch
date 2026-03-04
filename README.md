# Transformer-From-Scratch

Implementation of Transformer Neural Network architecture; implementation of training loop; Trained and tested on presidential speech data pulled from the University of Virginia.

### Authors:
- **Evan Economou**
- **Joe Huston**
- **Nathan Hoehndorf**

## How to run the code:
1. Clone the repo.
2. From the following Google Drive link, download the pres_data and place it in the ./data/ directory, then download any pretrained models that you would like to use. https://drive.google.com/drive/folders/1aZxdEda2bGkUnMhWXT0T9aDp7Uz-LRFx?usp=sharing
3. Run `uv sync` in the top level repo directory to create a virtual environment with all the correct dependencies installed, then activate the virtual environment. The project expects Python 3.11 or 3.12.
4. Running `interface.py` with `python interface.py` (or similar, depending on OS/environment) will print a menu to the terminal with a few options, allowing you to load in a pretrained model, train a new model, or talk to an untrained model.

### Testing guidelines
`config.ini` is the default when using the interface, and contains the parameters and data path we used for our main models we analyzed  
`small_data_config.ini` gives a much smaller model, about 60 thousand parameters, that trains on a small portion of a single book. This allows the quick generation of a loss curve and a general idea of how our code works. Just enter `./small_data_config.ini` when the interface asks you for a config directory.

### Config file notes
The config contains all the hyperparameters that can be adjusted to affect the model.  
[MODEL]  
`d_model = 32`  
`d_hidden = 128`  
`num_blocks = 4` The number of transformer blocks/layers in the model  
`max_seq_len = 1024` The maximum amount of tokens that the model can take as input  
`positional_embedding=True` Allows you to decide if positional embedding will be used or not, for comparison    
`pos_embedding_type="Sinusoidal"` Takes values "Sinusoidal" and "Learned" to decide   

[DATA]  
`tokenizer_data_path` = data/pres_data.txt # The path to where the data is stored, we put this in here because it was annoying to prompt for it every time  

## Repo Layout
`interface.py`: Contains a main function that just handles the user I/O and invokes our other files. <br />
`transformer.py`: Contains the transformer class and all related classes, including our implementation of attention heads, transformer blocks, tokenzier, and an MLP. <br />
`helper_functions.py`: Most importantly contains our train_model function, but also includes various other pieces of code that were useful to abstract for readability or reusability. <br />
`json_to_txt.py`: Translates the many .json files used to store the president speech data into a single .txt file that can easily be used as training data for the model.<br />
`config.ini`: Contains all of the information needed to configure the model setup and the training process.<br />
`pyproject.toml`: Dependency information enabling `uv sync`<br />
`data/`: Contains all of the .txt data files used to train the transformer.<br />
`results.md`: Contains loss plots, example I/O, and analysis of the results.  

## Individual Contributions
Everyone pair programmed most of the attention head, transformer block, transformer class, and the tokenizer. Then, **Joe** worked on the JSON parser that converted our presidential speech data into a text file, the sinusoidal positional embedding, the analysis of the results, and README instructions. **Nathan** worked on the encode & decode functions in the Tokenizer class, pair programming the training loop with Evan, implementing the PyTests, and writing up contributions, challenges, & future. Essentially everything else was done by **Evan**, and most things listed were done with the help of Evan.

## Discussion of Design Decisions
Starting with about 20MB of presidential speeches, our dataset was 21619261 tokens in length after running word-level tokenization. This seemed to be enough data that the training process was able to get close to plateauing with the size of model we are using, but incremental improvement consistently occurred until the dataset ended so more data would definitely be useful. Our models all had about 1.1 million parameters, `using d_model = 32`, `d_hidden = 128`, and `num_blocks = 4`. This worked fairly well, producing text that seems more coherent than the gibberish created by the untrained model while not taking an unreasonable amount of time to train with the limited computing power we had.  

In terms of training length, it depended quite a bit on hardware. The exact times are given in `results.md` for the three models we trained. Generally we aimed for less than 30 minutes per training run through the data.

There are a few interesting design decisions made in our generate() function. We had a lot of issues with repeated words in outputs, especially with the untrained model. To fix this we initially just tried a `torch.multinomial` sample instead of picking greedily, but that didn't quite produce the results we wanted. From there we tried adding a temperature parameter, flattening the logit distribution in an attempt to smooth out peaks and stop the randomly generated weights from getting stuck repeating the most likely word. We still were getting outputs that only consisted of a few words repeated in varying patterns, so we added a more strict generation requirement that the next token couldn't match the previous token. This increases generation time, but the outputs generally seem to more closely match what a user would expect as output, eliminating things like double commas or sequences of repeated generation of "the". To accomodate the increased generation time, we wanted to cut off the generation after a certain amount of time, not necessarily going to the max each time. To do this we just counted the number of periods generated by the model, and once it had created three sentences ended by a period we just end the generation.

## Challenges during Development
We faced quite a few challenges implementing our Tokenizer. From a software engineering perspective, we simply weren't sure what functionality should belong to which classes, and ended up with a slight mess of overlapping functionality and difficult to parse output. Eventually, after some whiteboard matrix drawings, we put together that Tokenizers really do just handle everything from getting the raw data up to the data being put into the model, and we were able to assemble everything in a manner that made sense

When positional embedding was first being implemented, there was a fundamental misunderstanding that led to large loss. The misunderstanding was largely, and simply, ignorance. After research and revewing of notes, our understanding was made concrete and we were able to implement positional embedding.

In the last week before the project was due, we learned that none of us knew what a one-hot vector was or that we needed to implement it. Fortunately, we were able to quickly remind ourselves and implement it within the same class period.

When we begun implementing training we were struggling due to very long training times, upwards of an hour and a half. Our first thought was that the excessive time was due to the length of the training data, the whole King James Bible. Reducing the amount of input decreased the training time margnially, but we still were above 30 minutes. Switching our optimizer from Stochastic Gradient Descent to Adam resulting in the times dropping to 10-15 minutes, depending on the computer. This allowed us to test our training loop multiple times in a single programming session.

## Future Directions
There are a few features we could implement if we had more time: multi-head attention, batching, and byte-pair tokenization. This would make the training process more efficient and produce a better output. Further, we could try using a larger dataset and using GPU training. If we had more time to experiment with the hyperparameters and the model configuration as a whole, we would be able to find more optimal configurations for training efficiency and model and accuracy. 

### References
Miller Center of Public Affairs, University of Virginia. "Presidential Speeches: Downloadable Data." Accessed March 1, 2026. data.millercenter.org. 
Vaswani, Ashish, et al.   
"Attention is all you need." Advances in neural information processing systems 30 (2017).
