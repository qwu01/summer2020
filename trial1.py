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


model = TransformerModel(n_tokens=args_parameters["n_tokens"],
                         embedding_dim=args_parameters["embedding_dim"],
                         n_heads=args_parameters["n_heads"],
                         fc_hidden_size=args_parameters["fc_hidden_size"],
                         n_layers=args_parameters["n_layers"],
                         dropout_p=args_parameters["dropout_p"]).to(device=args_parameters["device"])

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=args_parameters["learning_rate"])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

best_model = train_model(model=model,
                         train_data=input_dataset["train_data"],
                         val_data=input_dataset["val_data"],
                         epochs=args_parameters["epochs"],
                         criterion=criterion,
                         optimizer=optimizer,
                         scheduler=scheduler,
                         n_tokens=args_parameters["n_tokens"],
                         bqtt_len=35,
                         writer_dir=args_parameters["writer_dir"])

test_loss = evaluate_model(model=best_model, val_data=input_dataset["test_data"],
                           epoch=None, criterion=criterion,
                           n_tokens=args_parameters["n_tokens"],
                           bqtt_len=35, writer_dir=None)
print('=' * 100)
print(f'| End of training | test loss {test_loss:5.2f} |')
print('=' * 100)
torch.save(best_model, args_parameters["save_model_path"])
# loaded_model = torch.load(args_parameters["save_model_path"])
# print(loaded_model)
