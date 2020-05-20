from Project_A4.data.get_data import get_data
from Project_A4.model.model import TransformerModel
from Project_A4.train import train_model
from Project_A4.evaluate import evaluate_model

import torch
import torch.nn as nn
from argparse import Namespace

args = Namespace(
    writer_dir=None,
    train_batch_size=128,
    eval_batch_size=32,
    n_tokens=None,
    embedding_dim=200,
    fc_hidden_size=200,
    n_layers=4,
    n_heads=4,
    dropout_p=0.2,
    learning_rate=6.,
    gamma_=0.95,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    epochs=42,
    save_model_path="../Project_A4/model/saved_model.pth",
)


input_dataset, TEXT = get_data(args.train_batch_size,
                               args.eval_batch_size,
                               args.device)

args.n_tokens = len(TEXT.vocab.stoi)
args.writer_dir = "experiments/run_04"

#####
loaded_model = torch.load("../Project_A4/model/saved_model.pth")
## torch.save(loaded_model.state_dict(), "../Project_A3/model/saved_model_A_state.pth")

## loaded_model = TransformerModel(n_tokens=args.n_tokens,
##                                 embedding_dim=args.embedding_dim,
##                                 n_heads=args.n_heads,
##                                 fc_hidden_size=args.fc_hidden_size,
##                                 n_layers=args.n_layers,
##                                 dropout_p=args.dropout_p).to(device=args.device)
##
## loaded_model.load_state_dict(torch.load("../Project_A3/model/saved_model_A_state.pth"))


criterion = nn.CrossEntropyLoss()
test_loss = evaluate_model(model=loaded_model, val_data=input_dataset["test_data"],
                           epoch=None, criterion=criterion,
                           n_tokens=args.n_tokens,
                           bqtt_len=35, writer_dir=None)

print(test_loss) # 0.27508195076778436
