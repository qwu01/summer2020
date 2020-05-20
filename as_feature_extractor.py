from Project_A4.data.get_data import get_data
from Project_A4.data.get_batch import get_batch
import torch
import pandas as pd
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
args.writer_dir = "experiments/run_50"

loaded_model = torch.load("model/saved_model.pth")

for param in loaded_model.parameters():
    param.requires_grad = False

loaded_model.eval()


with torch.no_grad():
    print("*" * 100)
    ## input_data, target = get_batch(input_dataset["test_data"], 0, input_dataset["test_data"].size(0))
    ## print(f"input_data shape: {input_data.shape}, target shape: {target.shape}")
    # for batch, i in enumerate(range(0, input_dataset["test_data"].size(0) - 1, 35)):
    #     if batch % 100 == 0:
    #         print(f"Batch {batch}, i {i}")
    #         input_data, _ = get_batch(input_dataset["test_data"], i, 35)
    #         print(input_data.shape)
    #         print(input_data[0, :])  # .unsqueeze(1))
    #         fc_as_decoder_output, transformer_encoder_output, \
    #         positional_encoding_output, embedding_output = loaded_model(input_data)
    #
    #         print(fc_as_decoder_output.shape)
    #         print(transformer_encoder_output.shape)
    #         print(positional_encoding_output.shape)
    #         print(embedding_output.shape)
    #
    #         print(transformer_encoder_output[0, :, :].shape)

    #### bqtt_len = 35
    for batch, i in enumerate(range(0, input_dataset["train_data"].size(0) - 1, 35)):
        if batch % 100 == 0:
            input_data, targets = get_batch(input_dataset["train_data"], i, 35)
            print(f"input_data shape: {input_data.shape}; targets shape: {targets.shape}")

            fc_as_decoder_output, transformer_encoder_output, \
            positional_encoding_output, embedding_output = loaded_model(input_data)

            print(f"output shape {fc_as_decoder_output.shape}")

            print("-"*150, "\n", "-"*150)
            print(fc_as_decoder_output.shape)
            print(torch.argmax(fc_as_decoder_output,dim=2)[:,0])
            print(targets[:,0])
            print("-"*150, "\n", "-"*150)
            print(transformer_encoder_output.shape)

            


