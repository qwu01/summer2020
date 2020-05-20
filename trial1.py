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
args.writer_dir = "experiments/run_11"

model = TransformerModel(args.n_tokens,
                         args.embedding_dim,
                         args.n_heads,
                         args.fc_hidden_size,
                         args.n_layers,
                         args.dropout_p).to(args.device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.gamma_)

best_model = train_model(model=model,
                         train_data=input_dataset["train_data"],
                         val_data=input_dataset["val_data"],
                         epochs=args.epochs,
                         criterion=criterion,
                         optimizer=optimizer,
                         scheduler=scheduler,
                         n_tokens=args.n_tokens,
                         bqtt_len=35,
                         writer_dir=args.writer_dir)

test_loss = evaluate_model(model=best_model, val_data=input_dataset["test_data"],
                           epoch=None, criterion=criterion,
                           n_tokens=args.n_tokens,
                           bqtt_len=35, writer_dir=None)
print('=' * 100)
print(f'| End of training | test loss {test_loss:5.2f} |')
print('=' * 100)
torch.save(best_model, args.save_model_path)
# loaded_model = torch.load(args.save_model_path)
# print(loaded_model)
