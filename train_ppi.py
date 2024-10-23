import torch
import argparse
import os
import json
import time
import numpy as np

from utils import load_ppi_partition, PPIDataLoader
from model import GAT
from typing import Tuple
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score

def create_checkpoint(model_dir : str, epoch : int, model : GAT, optimizer : torch.optim.Optimizer):
    d = model.to_dict()
    d["epoch"] = epoch

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    
    with open(os.path.join(model_dir, ".metadata.json"), "w+", encoding = "utf-8") as f:
        json.dump(d, f, indent = 4)

    torch.save(model.state_dict(), os.path.join(model_dir, "checkpoint.pth"))
    torch.save(optimizer.state_dict(), os.path.join(model_dir, "optimizer.pth"))

def from_checkpoint(model_dir : str, input_features : int) -> Tuple[GAT, torch.optim.Optimizer, int]:
    with open(os.path.join(model_dir, ".metadata.json"), "r", encoding = "utf-8") as f:
        metadata = json.load(f)
    
    model = GAT(input_features,
                heads_per_layer = metadata["heads_per_layer"],
                features_per_layer = metadata["features_per_layer"],
                dropout_p = metadata["dropout_p"],
                residual = metadata["residual"])
    
    epoch = metadata["epoch"]

    model_state_dict = torch.load(os.path.join(model_dir, "checkpoint.pth"))
    model.load_state_dict(model_state_dict)

    optim_state_dict = torch.load(os.path.join(model_dir, "optimizer.pth"))
    optimizer = torch.optim.Adam(model.parameters)
    optimizer.load_state_dict(optim_state_dict)

    return model, optimizer, epoch

def parse_args():
    # Not all hyperparameters were discussed in the GAT paper but they can be found in their original GitHub repo
    # https://github.com/PetarV-/GAT/blob/master/execute_cora.py

    args = argparse.ArgumentParser()
    args.add_argument("data_dir", type = str, help = "Directory that contains the CORA dataset.")
    args.add_argument("model_dir", type = str, help = "Directory where the model will be saved.")
    args.add_argument("epochs", type = int, help = "Number of training epochs")

    args.add_argument("--attention_heads_per_layer", type = int, nargs = "+", help = "List of numbers of attention heads per GAT layer.", required = True)
    args.add_argument("--num_features_per_layer", type = int, nargs = "+", help = "List of number of features per GAT layer. Output number of features for a particular layer is num_output_features * num_attention_heads", required = True)
    args.add_argument("--residual", type = bool, nargs = "?", const = True, default = False, help = "Specify if the network should use residual connections. Defaults to False")
    args.add_argument("--dropout_p", type = float, default = 0.6, help = "Specify dropout probability. Dropout is applied to the input of GAT layers as well as on per-edge attention scores.")
    args.add_argument("--from_checkpoint", type = bool, nargs = "?", const = True, default = False, help = "Specify if you wish to continue training from a checkpoint in model_dir.")
    args.add_argument("--learning_rate", type = float, default = 5e-3, help = "Specify learning rate.")
    args.add_argument("--weight_decay", type = float, default = 5e-4, help = "Specify amount of applied weight decay.")
    args.add_argument("--random_seed", type = int, default = 41, help = "Specify reproducibility seed.")
    args.add_argument("--log_every", type = int, default = 100, help = "Specify logging frequency.")
    args.add_argument("--checkpoint_period", type = int, default = 100, help = "Specify checkpointing period.")
    args.add_argument("--batch_size", type = int, default = 1, help = "Specify the batch size for training. Batch size for inductive GNN training determines number of graphs that are simultaneously forwarded through the network.")

    return args.parse_args()

def forward_loader(model : GAT, loader : PPIDataLoader, device : str):
    predictions = []
    real_labels = []

    losses = []
    loss_fn = torch.nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for batch in loader:
            node_features = batch[0].to(device)
            node_labels = batch[1].to(device)
            edge_index = batch[2].to(device)

            logits = model.forward((node_features, edge_index))
            l = loss_fn(logits, node_labels)
            losses.append(l.item())

            # By sigmoid definition, when logit is > 0 implied probability is > 0.5, which is the prediction threshold
            preds = (logits > 0).cpu().numpy()
            predictions.append(preds)
            real_labels.append(node_labels.cpu().numpy())

    predictions = np.concatenate(predictions)
    real_labels = np.concatenate(real_labels)
    loss = sum(losses) / len(losses)
    return loss, f1_score(predictions, real_labels, average = "micro")

if __name__ == "__main__":
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.random_seed)

    assert args.num_features_per_layer[-1] == 121, f"Expected 121 dimensional output from the network, as PPI is a 121 multi-label classification problem!"

    train_dl, num_features, num_classes = load_ppi_partition(args.data_dir, "train", batch_size = args.batch_size)
    val_dl, _, _ = load_ppi_partition(args.data_dir, "valid")
    test_dl, _, _ = load_ppi_partition(args.data_dir, "test")
    prev_epoch = 0

    writer = SummaryWriter()

    if args.from_checkpoint:
        model, optimizer, prev_epoch = from_checkpoint(args.model_dir, num_features)

    else:
        model = GAT(num_features, 
                    heads_per_layer = args.attention_heads_per_layer,
                    features_per_layer = args.num_features_per_layer,
                    dropout_p = args.dropout_p,
                    residual = args.residual).to(device)
        
        # Even though AdamW should be used to correctly apply weight decay, authors use ordinary Adam optimizer, so we will stick to that as well
        # to reproduce their results as closely as possible
        optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate, weight_decay = args.weight_decay)

    loss = torch.nn.BCEWithLogitsLoss()
    start = time.time()

    for epoch in range(prev_epoch, args.epochs):
        model.train()

        for batch in train_dl:
            node_features = batch[0].to(device)
            node_labels = batch[1].to(device)
            edge_index = batch[2].to(device)

            logits = model.forward((node_features, edge_index))
            l = loss(logits, node_labels)
            l.backward()
            optimizer.step()
            optimizer.zero_grad()

        model.eval()

        train_loss, train_f1 = forward_loader(model, train_dl, device)
        val_loss, val_f1 = forward_loader(model, val_dl, device)

        writer.add_scalar("train_loss", train_loss, epoch)
        writer.add_scalar("train_f1", train_f1, epoch)
        writer.add_scalar("val_loss", val_loss, epoch)
        writer.add_scalar("val_f1", val_f1, epoch)

        if (epoch + 1) % args.log_every == 0 or epoch == args.epochs - 1:
            print(f"Epoch: {epoch + 1} --- Training and inference time up to this point: {(time.time() - start):.4f}s. --- Train loss: {train_loss:.4f} --- Validation loss: {val_loss:.4f} --- Train micro-averaged F1: {train_f1:.4f} --- Validation micro-averaged F1: {val_f1:.4f}")
            start = time.time()

        if epoch % args.checkpoint_period == 0 or epoch == args.epochs - 1:
            create_checkpoint(args.model_dir, epoch + 1, model, optimizer)
    
    test_loss, test_f1 = forward_loader(model, test_dl, device)
    print(f"Test micro-averaged F1: {test_f1:.4f}")