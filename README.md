# Transformer-From-Scratch

Implementation of Transformer Neural Network architecture; implementation of training loop; Trained and tested on book data pulled from Gutenberg

### Authors:
- **Evan Economou**
- **Joe Huston**
- **Nathan Hoehndorf**

## How to run the code:
1. Clone the repo.
2. Run `uv sync` in the main repo directory to create a virtual environment with all the correct dependencies installed, then activate the virtual environment.
3. Running `interface.py` with `python interface.py` (or similar, depending on OS/environment) will print a menu to the terminal with a few options, allowing you to load in a pretrained model, train a new model, or talk to an untrained model.

## Code Structure
`interface.py`: Contains a main function that just handles the user I/O and invokes our other files. <br />
`transformer.py`: Contains the transformer class and all related classes, including our implementation of attention heads, transformer blocks and an MLP. <br />
`helper_functions.py`: Most importantly contains our train_model function, but also includes various other pieces of code that were useful to abstract for readability or reusability. <br />
`json_to_txt.py`: Translates the many .json files used to store the president speech data into a single .txt file that can easily be used as training data for the model.<br />
`config.ini`: Contains all of the information needed to configure the model setup and the training process.<br />
`pyproject.toml`: Dependency information enabling `uv sync `<br />
`data/`: Contains all of the .txt data files used to train the transformer.<br />

### References (for data acquisition)
Miller Center of Public Affairs, University of Virginia. "Presidential Speeches: Downloadable Data." Accessed March 17, 2022. data.millercenter.org. 
