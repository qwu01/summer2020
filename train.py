from Project_A4.data.get_batch import get_batch
from Project_A4.evaluate import evaluate_model
import torch.nn as nn
import time
from torch.utils.tensorboard import SummaryWriter


def train_model(model, train_data, val_data, epochs, criterion, optimizer, scheduler, n_tokens, bqtt_len, writer_dir):

    writer = SummaryWriter(writer_dir)

    best_val_loss = 0.
    best_model = None

    for epoch in range(epochs):

        epoch_start_time = time.time()

        # train
        model.train()
        batch_loss = 0.

        for batch, i in enumerate(range(0, train_data.size(0) - 1, bqtt_len)):

            input_data, targets = get_batch(train_data, i, bqtt_len)
            optimizer.zero_grad()
            output_data, _, _, _ = model(input_data)

            loss = criterion(output_data.view(-1, n_tokens), targets.view(-1))
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            batch_loss += loss.item()

            if batch % 100 == 0 and batch > 0:
                current_loss = batch_loss / 100
                print("-" * 60)
                print(f"epoch {epoch}, batch {batch} / {len(train_data) // bqtt_len}")
                print(f"loss {current_loss:5.2f}")
                writer.add_scalar("Training Loss (Batch)", current_loss, epoch * (len(train_data) // bqtt_len) + batch)
                batch_loss = 0
        print(f"epoch {epoch}, {(time.time() - epoch_start_time) // 60} minutes.")
        print("-" * 60, f"End of epoch {epoch}", "-" * 60)

        scheduler.step()

        best_model = evaluate_model(model, val_data, epoch, criterion, n_tokens, bqtt_len, writer_dir)
        writer.close()

    return best_model