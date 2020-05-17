from Project_A3.data.get_data import get_data
from Project_A3.model.model import TransformerModel
from Project_A3.train import train_model
from Project_A3.evaluate import evaluate_model

import torch
import torch.nn as nn

args_parameters = {
    "writer_dir": None,
    "train_batch_size": 64,
    "eval_batch_size": 16,
    "n_tokens": None,
    "embedding_dim": 200,
    "fc_hidden_size": 200,
    "n_layers": 4,
    "n_heads": 4,
    "dropout_p": 0.2,
    "learning_rate": 7.,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "epochs": 100,
    "save_model_path": "../Project_A3/model/saved_model.pth",
}

input_dataset, TEXT = get_data(args_parameters["train_batch_size"],
                               args_parameters["eval_batch_size"],
                               args_parameters["device"])

args_parameters["n_tokens"] = len(TEXT.vocab.stoi)
args_parameters["writer_dir"] = "experiments/run_04"

loaded_model = torch.load("model/saved_model.pth")
criterion = nn.CrossEntropyLoss()
test_loss = evaluate_model(model=loaded_model, val_data=input_dataset["test_data"],
                           epoch=None, criterion=criterion,
                           n_tokens=args_parameters["n_tokens"],
                           bqtt_len=35, writer_dir=None)

print(test_loss)
