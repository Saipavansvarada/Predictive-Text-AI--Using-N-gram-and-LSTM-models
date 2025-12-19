# models/lstm_predictor.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import re
from collections import Counter
import math
import random
import os

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

class WordDataset(Dataset):
    def __init__(self, lines, seq_len=4, min_freq=1):
        tokens = []
        for l in lines:
            tokens += tokenize(l)
        counts = Counter(tokens)
        self.vocab = {"<pad>":0, "<unk>":1, "<s>":2, "</s>":3}
        for w,c in counts.items():
            if c >= min_freq:
                self.vocab[w] = len(self.vocab)
        self.inv_vocab = {i:w for w,i in self.vocab.items()}

        self.seq_len = seq_len
        # build sequences
        self.data = []
        for l in lines:
            toks = tokenize(l)
            toks = ["<s>"] + toks + ["</s>"]
            ids = [self.vocab.get(t, self.vocab["<unk>"]) for t in toks]
            for i in range(0, max(1, len(ids)-seq_len)):
                seq = ids[i:i+seq_len]
                target = ids[i+seq_len] if i+seq_len < len(ids) else self.vocab["</s>"]
                if len(seq) < seq_len:
                    seq = seq + [self.vocab["<pad>"]] * (seq_len - len(seq))
                self.data.append((torch.tensor(seq, dtype=torch.long), torch.tensor(target, dtype=torch.long)))

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, emb_size=64, hidden=128, num_layers=1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=0)
        self.lstm = nn.LSTM(emb_size, hidden, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden, vocab_size)

    def forward(self, x):
        # x: (batch, seq_len)
        emb = self.emb(x)
        out, _ = self.lstm(emb)
        # take last output (at seq end)
        last = out[:, -1, :]
        logits = self.fc(last)
        return logits

def train_model(lines, seq_len=4, epochs=10, batch_size=32, lr=0.01, device='cpu', save_path="models/lstm.pth"):
    ds = WordDataset(lines, seq_len=seq_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    model = LSTMModel(len(ds.vocab)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for ep in range(1, epochs+1):
        total_loss = 0.0
        model.train()
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        avg = total_loss / len(ds)
        print(f"Epoch {ep}/{epochs} loss={avg:.4f}")
    # save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "vocab": ds.vocab,
        "seq_len": seq_len
    }, save_path)
    print("Saved LSTM model to", save_path)
    return save_path

def load_model(path, device='cpu'):
    checkpoint = torch.load(path, map_location=device)
    vocab = checkpoint["vocab"]
    seq_len = checkpoint["seq_len"]
    model = LSTMModel(len(vocab))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    inv_vocab = {i:w for w,i in vocab.items()}
    return model, vocab, inv_vocab, seq_len

def predict_next_words(model, vocab, inv_vocab, seq_len, context_words, top_k=5, device='cpu'):
    # Use last seq_len tokens as input
    tokens = [w.lower() for w in context_words if w]
    toks = (["<s>"] + tokens)[-seq_len:]
    ids = [vocab.get(t, vocab.get("<unk>")) for t in toks]
    import torch.nn.functional as F
    x = torch.tensor([ids], dtype=torch.long).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
    # top k
    topk_idx = probs.argsort()[-top_k:][::-1]
    return [(inv_vocab[i], float(probs[i])) for i in topk_idx]

if __name__ == "__main__":
    # tiny demo train
    with open("data/sample_corpus.txt", "r", encoding="utf8") as f:
        lines = [l.strip() for l in f if l.strip()]
    train_model(lines, epochs=50, batch_size=8, lr=0.01, save_path="models/lstm_demo.pth")
    model, vocab, inv_vocab, seq_len = load_model("models/lstm_demo.pth")
    print(predict_next_words(model, vocab, inv_vocab, seq_len, ["i", "am"], top_k=5))
