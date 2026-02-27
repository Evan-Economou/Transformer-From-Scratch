from transformer import Transformer, Config, Tokenizer, TransformerBlock, AttentionHead, MLP
import torch
import helper_functions as hf


def main():
    decision = input("""
╔═══════════════════════════╗
║  1. Train New Model       ║
║  2. Load Existing Model   ║
║  3. Test Untrained Model  ║
║  4. Exit                  ║
╚═══════════════════════════╝
Enter choice (1-4): """)
    match decision:
        case '1': #user wants to load data and train a new model
            config_path = input("Enter the path to the config file (press enter for ./config.ini): ") or "./config.ini"
            num_blocks, positional_embedding, batch_size, config = hf.load_config(config_path)
            # Train on the same data used to build the tokenizer vocabulary
            raw_data = config.tokenizer.raw_data
            model = Transformer(num_blocks, config, positional_embedding)
            loss_history = hf.train_model(model, raw_data, batch_size=batch_size)
            torch.save(model, "transformer_model.pt")
            print("Model saved to transformer_model.pt")
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
            prompt = input("Running input through an untrained model, enter prompt: ")
            tokenizer = Tokenizer(raw_data="Aaaah Im Tokenizing It")
            config = Config(d_model=16, d_vocab=tokenizer.vocab_size, d_hidden=64,tokenizer=tokenizer)
            model = Transformer(num_blocks=2, config=config)
            output = model.generate_output(prompt)
            print(f"Generated output: {output}")
        case _:
            print("Goodbye")


if __name__ == "__main__":
    main()