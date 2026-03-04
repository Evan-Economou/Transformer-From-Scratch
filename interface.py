from transformer import Transformer, Config, Tokenizer, TransformerBlock, AttentionHead, MLP
import torch
import helper_functions as hf
import time


def main():
    decision = input("""
1. Train New Model
2. Load Existing Model
3. Test Untrained Model
4. Exit

Enter choice (1-4): """)
    match decision:
        case '1': #user wants to load data and train a new model
            config_path = input("Enter the path to the config file (press enter for ./config.ini): ") or "./config.ini"
            model_path = input("Enter a name for this model (press enter for transformer_model): ") or "transformer_model"
            model_path = "./" + model_path + ".pt"
            config = hf.load_config(config_path)
            #train on the same data used to build the tokenizer vocabulary
            raw_data = config.tokenizer.raw_data
            model = Transformer(config)
            start_time = time.time()
            loss_history = hf.train_model(model, raw_data)
            end_time = time.time()
            print(f"Training completed in {end_time - start_time:.2f} seconds.")
            print(f"Initial loss: {loss_history[0]:.4f}, Final loss: {loss_history[-1]:.4f}")
            torch.save(model, model_path)
            print(f"Model saved to {model_path}")
            hf.plot_loss(loss_history)

            hf.conversation_loop(model)
        case '2': #assume a model file exists and try to load it
            while True:
                try:
                    model_path = input("Enter the path to the saved model: ") or "./transformer_model.pt"
                    model = torch.load(model_path, weights_only=False)
                    print("Model loaded successfully.")
                    break
                except Exception as e:
                    print(f"Error loading model: {e}. Please try again.")
            hf.conversation_loop(model)
        case '3':
            config_path = input("Enter the path to the config file (press enter for ./config.ini): ") or "./config.ini"
            config = hf.load_config(config_path)
            #train on the same data used to build the tokenizer vocabulary
            raw_data = config.tokenizer.raw_data
            model = Transformer(config)
            #increase temperature to flatten proability distribution so that the untrained model doesn't just generate one word repeatedly
            hf.conversation_loop(model, n_tokens = 40, temperature=5.0) 
        case _:
            print("Goodbye")


if __name__ == "__main__":
    main()