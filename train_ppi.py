import torch
import argparse
import os
import json
import time

from utils import load_ppi_partition
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

if __name__ == "__main__":
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.random_seed)

    train_dl, num_features, num_classes = load_ppi_partition(args.data_dir, "train", batch_size = args.batch_size)
    val_dl, _, _ = load_ppi_partition(args.data_dir, "valid")
    test_dl, _, _ = load_ppi_partition(args.data_dir, "train")
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

    loss = torch.nn.CrossEntropyLoss()

    for epoch in range(prev_epoch, args.epochs):
        start = time.time()

        model.train()
        logits = model.forward(data)
        train_logits = logits.index_select(0, train_indices)
        l = loss(train_logits, train_labels)

        l.backward()
        optimizer.step()
        optimizer.zero_grad()

        model.eval()
        with torch.no_grad():
            logits = model.forward(data)

            train_logits = logits.index_select(0, train_indices)
            train_loss = loss(train_logits, train_labels)
            train_acc = accuracy(train_logits, train_labels)

            val_logits = logits.index_select(0, val_indices)
            val_acc = accuracy(val_logits, val_labels)
            val_loss = loss(val_logits, val_labels)

            writer.add_scalar("train_loss", train_loss, epoch)
            writer.add_scalar("train_accuracy", train_acc, epoch)
            writer.add_scalar("validation_loss", val_loss, epoch)
            writer.add_scalar("validation_accuracy", val_acc, epoch)
            
            if (epoch + 1) % args.log_every == 0 or epoch == args.epochs - 1:
                print(f"Epoch: {epoch + 1} --- Train step and inference time: {(time.time() - start):.4f}s. --- Train loss: {train_loss.item():.4f} --- Validation loss: {val_loss.item():.4f} --- Train accuracy: {train_acc:.4f} --- Val accuracy: {val_acc:.4f}")
        
        if epoch % args.checkpoint_period == 0 or epoch == args.epochs - 1:
            create_checkpoint(args.model_dir, epoch + 1, model, optimizer)
    
    model.eval()
    logits = model.forward(data)
    test_logits = logits.index_select(0, test_indices)
    acc = accuracy(test_logits, test_labels)
    print(f"Test accuracy: {acc:.4f}")