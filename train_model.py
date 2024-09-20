
from src.train_setup import TrainSetup


model_config = {
    "model-name": "bootz_model_unbalanced",           # give your model a descriptive name
    "dataset-name": "bootz_groups_unbalanced",       # name of dataset you want to use (must contain "dataset.pickle" and "nationalities.json")
    "test-size": 0.1,                   # for test and validation set seperatly
    "optimizer": "Adam",                # changing it here doesn't make a difference, please change in train_setup.py directly
    "loss-function": "NLLLoss",         # changing it here doesn't make a difference, please change in train_setup.py directly
    "epochs": 12,                       # amount of epochs to train for
    "batch-size": 1024,                 # amount of samples to process in parallel
    "cnn-parameters": [3, 256],         # idx 0: kernel size, idx 1: list of feature map dimension)
    "hidden-size": 200,                 # default = 200
    "rnn-layers": 2,                    # default = 2
    "lr-schedule": [0.001, 0.99, 200],  # (idx 0: current lr, idx 1: decay rate, idx 2: decay intervall in iterations), change idx 0 when resuming training
    "dropout-chance": 0.2,              # dropout chance after the LSTM layer
    "embedding-size": 200,              # dimension of the input embeddings
    "augmentation": 0.0,                # name part switching will slow down the training process when set high
    "resume": False,                    # set to True when resuming a cancelled training run and update the idx 0 of lr-schedule accordingly
    "wandb-project": "n2e",   # name of the Weights&Biases project you want to track your experiments in (if you don't use it leave as is)
    "wandb-entity": "theodorp"    # name of the Weights&Biases username (if you don't use it leave as is)
}

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a model with the specified configuration.")
    parser.add_argument("--model_name", type=str, default=model_config["model-name"], help="Name of the model/experiment (default: spanish_german_else)")
    parser.add_argument("--dataset_name", type=str, default=model_config["dataset-name"], help="Name of the dataset folder (default: spanish_german_else)")
    args = parser.parse_args()

    # Update model configuration with CLI arguments
    model_config = model_config.copy()
    model_config["model-name"] = args.model_name
    model_config["dataset-name"] = args.dataset_name

    # Train and test
    train_setup = TrainSetup(model_config)
    # train_setup.train()
    train_setup.test(print_amount=None, plot_confusion_matrix=True, plot_scores=False)
