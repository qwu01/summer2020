from Project_A3.data.get_data import get_data
from Project_A3.data.get_batch import get_batch
import torch
import torch.nn as nn
import pandas as pd

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

for param in loaded_model.parameters():
    param.requires_grad = False

class Identity(nn.Module):
    """
    https://discuss.pytorch.org/t/how-to-delete-layer-in-pretrained-model/17648
    """
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x


loaded_model.fc_layer_as_decoder = Identity()

loaded_model.eval()

input_data_to_be_saved = {}
output_data_to_be_saved = {}
with torch.no_grad():
    for batch, i in enumerate(range(0, input_dataset["test_data"].size(0) - 1, 35)):
        input_data, _ = get_batch(input_dataset["test_data"], i, 35)
        input_data_to_be_saved[str(batch)] = input_data
        # print(f"Input dat shape: {input_data.shape}")
        output_data = loaded_model(input_data)
        output_data_to_be_saved[str(batch)] = output_data
        # print(f"Output data shape: {output_data.shape}")

print("-"*100)

inputs = input_data_to_be_saved["0"]
inputs = inputs.permute(1, 0)
print(inputs.shape)
inputs_0 = inputs[0, :]
for i in range(1, inputs.shape[0]):
    temp = inputs[i, :]
    inputs_0 = torch.cat((inputs_0, temp), dim=0)
inputs_0 = inputs_0.to("cpu").numpy()

outputs = output_data_to_be_saved["0"]
outputs = outputs.permute(1, 0, 2)
print(outputs.shape)
outputs_0 = outputs[0, :, :].squeeze()
for j in range(1, outputs.shape[0]):
    temp = outputs[j, :, :].squeeze()
    outputs_0 = torch.cat((outputs_0, temp), dim=0)
outputs_0 = outputs_0.to("cpu").numpy()

print(inputs_0.shape, outputs_0.shape)
df_inputs_0 = pd.DataFrame(inputs_0)
df_inputs_0.to_csv("../Project_A3/res/inputs_0.csv", index=True)

df_outputs_0 = pd.DataFrame(outputs_0)
df_outputs_0.to_csv("../Project_A3/res/outputs_0.csv", index=True)

# https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
# https://github.com/pytorch/examples/tree/master/word_language_model
# https://medium.com/huggingface/encoder-decoders-in-transformers-a-hybrid-pre-trained-architecture-for-seq2seq-af4d7bf14bb8
# https://huggingface.co/transformers/model_doc/bert.html#bertmodel
# https://github.com/huggingface/transformers/issues/1950
# https://www.tensorflow.org/tutorials/text/transformer
# https://discuss.pytorch.org/t/how-to-delete-layer-in-pretrained-model/17648
