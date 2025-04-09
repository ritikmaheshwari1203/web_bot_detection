import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from transformers import get_scheduler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

from .model import BotDetectionModel
from .data import WebTrafficDataset  # Updated to use the refactored dataset class

# Training
def train(model, dataloader, optimizer, scheduler, device, epoch, writer):
    model.train()
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    for batch in tqdm(dataloader, desc=f"Training Epoch {epoch}"):
        optimizer.zero_grad()
        inputs, labels = batch
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        labels = labels.to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    writer.add_scalar("Loss/train", avg_loss, epoch)
    return avg_loss

# Evaluation
def evaluate(model, dataloader, device, epoch, writer):
    model.eval()
    all_labels, all_predictions, all_probs = [], [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating on {epoch}"):
            inputs, labels = batch
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probs, dim=1)
            all_labels.extend(labels.numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    acc = accuracy_score(all_labels, all_predictions)
    writer.add_scalar("Accuracy/test", acc, epoch)
    return all_labels, all_predictions, all_probs, acc

def collate_fn(batch):
    inputs = {key: torch.stack([item[0][key] for item in batch]) for key in batch[0][0]}
    labels = torch.stack([item[1] for item in batch])
    return inputs, labels

# Save & Resume
def save_checkpoint(model, optimizer, scheduler, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict()
    }, path)

def load_checkpoint(path, model, optimizer=None, scheduler=None):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state'])
    if optimizer and scheduler:
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
    return checkpoint['epoch'] + 1

def main():
    import argparse
    from transformers import AutoTokenizer
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the CSV data file")
    parser.add_argument("--model_name", default="bert-base-uncased")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=7)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default="runs")
    parser.add_argument("--save_path", type=str, default="weights")
    args = parser.parse_args()

    writer = SummaryWriter(log_dir=args.log_dir)

    dir_path = Path(f"{args.save_path}")
    dir_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv_path)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Define a class to make the Dataset work with our training code
    class TrainingDataset(WebTrafficDataset):
        def __init__(self, tokenizer, max_length=128, df=None):
            super().__init__(tokenizer, max_length, df)
            self.label_column = 'ROBOT'
            
        def __getitem__(self, idx):
            encoding = super().__getitem__(idx)
            label = torch.tensor(int(self.data.iloc[idx][self.label_column]))
            return encoding, label
    
    train_dataset = TrainingDataset(tokenizer=tokenizer, df=train_df)
    test_dataset = TrainingDataset(tokenizer=tokenizer, df=test_df)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BotDetectionModel(model_name=args.model_name).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0,
                             num_training_steps=len(train_loader) * args.epochs)

    start_epoch = 0
    best_acc = 0.0
    if args.resume:
        start_epoch = load_checkpoint(args.resume, model, optimizer, scheduler)
        print(f"Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        train_loss = train(model, train_loader, optimizer, scheduler, device, epoch, writer)
        save_checkpoint(model, optimizer, scheduler, epoch, f"./{args.save_path}/{args.model_name}_epoch_{epoch}.pt")
        labels, preds, probs, acc = evaluate(model, test_loader, device, epoch, writer)
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {train_loss:.4f}, Accuracy: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            save_checkpoint(model, optimizer, scheduler, epoch, f"./{args.save_path}/best_model.pt")

    writer.close()

if __name__ == "__main__":
    main()