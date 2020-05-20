from Project_A4.data.get_batch import get_batch
import torch
from torch.utils.tensorboard import SummaryWriter


def evaluate_model(model, val_data, epoch, criterion, n_tokens, bqtt_len, writer_dir):
    if writer_dir:
        writer = SummaryWriter(writer_dir)

    best_val_loss = 0.
    best_model = None

    # val
    model.eval()
    total_loss = 0.
    val_loss = 0.

    with torch.no_grad():
        for batch, i in enumerate(range(0, val_data.size(0) - 1, bqtt_len)):
            input_data, targets = get_batch(val_data, i, bqtt_len)
            output_data, _, _, _ = model(input_data)
            total_loss += len(input_data) * criterion(output_data.view(-1, n_tokens),
                                                      targets.view(-1)).item()

    val_loss = total_loss / (val_data.size(0) - 1)

    if writer_dir:
        writer.add_scalar("Validation Loss", val_loss, epoch)
        # Update best model
        if best_val_loss < val_loss:
            best_val_loss = val_loss
            best_model = model

        return best_model
    else:
        return val_loss
