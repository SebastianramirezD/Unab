"""
train_dl_models.py
------------------
Entrena tres arquitecturas de Deep Learning para clasificación de spam:
    1. CNN 1D       — Convoluciones sobre embeddings
    2. BiLSTM       — LSTM bidireccional
    3. BERT-tiny    — Transformer preentrenado (fine-tuning)

Salidas:
    models/cnn1d_model.pt
    models/bilstm_model.pt
    models/bert_tiny_model/   (directorio HuggingFace)
    reports/dl_results.csv
"""

import os
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import joblib

from sklearn.metrics import f1_score, roc_auc_score, classification_report

# ─── Reproducibilidad total ───────────────────────────────────────────────────
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR   = os.path.join(os.path.dirname(__file__), "../..")
PROC_DIR   = os.path.join(BASE_DIR, "datasets/processed")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# ─── Hiperparámetros globales ─────────────────────────────────────────────────
MAX_LEN    = 128      # longitud máxima de secuencia (tokens)
VOCAB_SIZE = 20_000   # tamaño del vocabulario
EMBED_DIM  = 128      # dimensión de embeddings
BATCH_SIZE = 64
EPOCHS     = 10
LR         = 1e-3


# ─── Dataset ──────────────────────────────────────────────────────────────────

class SpamDataset(Dataset):
    """Dataset que tokeniza texto a nivel de caracteres simples (toy tokenizer)."""

    def __init__(self, texts, labels, vocab=None, max_len=MAX_LEN):
        self.labels  = torch.tensor(labels, dtype=torch.long)
        self.max_len = max_len

        if vocab is None:
            # Construir vocabulario a partir del corpus
            from collections import Counter
            all_words = " ".join(texts).split()
            counts    = Counter(all_words)
            self.vocab = {"<PAD>": 0, "<UNK>": 1}
            for w, _ in counts.most_common(VOCAB_SIZE - 2):
                self.vocab[w] = len(self.vocab)
        else:
            self.vocab = vocab

        self.sequences = [self._encode(t) for t in texts]

    def _encode(self, text):
        ids = [self.vocab.get(w, 1) for w in text.split()]
        if len(ids) < self.max_len:
            ids += [0] * (self.max_len - len(ids))
        return torch.tensor(ids[:self.max_len], dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


# ─── Modelos ──────────────────────────────────────────────────────────────────

class CNN1D(nn.Module):
    """
    CNN 1D para clasificación de texto.
    Aplica convoluciones con múltiples tamaños de kernel (3, 4, 5)
    para capturar n-gramas de distintas longitudes.
    """
    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM,
                 num_filters=128, kernel_sizes=(3, 4, 5), num_classes=2, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, k) for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x):
        # x: (batch, seq_len)
        emb = self.embedding(x).permute(0, 2, 1)          # (batch, embed, seq)
        pooled = []
        for conv in self.convs:
            c = torch.relu(conv(emb))                      # (batch, filters, seq-k+1)
            p = torch.max(c, dim=2).values                 # max-pooling global
            pooled.append(p)
        cat = torch.cat(pooled, dim=1)                     # (batch, filters*3)
        out = self.fc(self.dropout(cat))
        return out


class BiLSTM(nn.Module):
    """
    BiLSTM para clasificación de texto.
    Usa el estado oculto del último paso de ambas direcciones.
    """
    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM,
                 hidden_dim=128, num_layers=2, num_classes=2, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        emb = self.dropout(self.embedding(x))              # (batch, seq, embed)
        _, (h_n, _) = self.lstm(emb)
        # Concatenar últimas capas de ambas direcciones
        h_fwd = h_n[-2]    # última capa, dirección forward
        h_bwd = h_n[-1]    # última capa, dirección backward
        out = self.fc(self.dropout(torch.cat([h_fwd, h_bwd], dim=1)))
        return out


# ─── Entrenamiento ────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
        correct    += (logits.argmax(1) == y_batch).sum().item()
        total      += len(y_batch)
    return total_loss / total, correct / total


def evaluate(model, loader):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            logits  = model(X_batch)
            probs   = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds   = logits.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(y_batch.numpy())
    f1  = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    return f1, auc, all_preds, all_labels


def train_torch_model(model, train_loader, val_loader, name, save_path):
    """Loop de entrenamiento con early stopping (paciencia=3)."""
    model.to(DEVICE)
    optimizer   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler   = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
    # Pesos de clase para manejar desbalance
    criterion   = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 6.5]).to(DEVICE))

    best_f1, patience_count = 0.0, 0
    history = []

    t0 = time.time()
    print(f"\nEntrenando {name} en {DEVICE}...")

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        val_f1, val_auc, _, _ = evaluate(model, val_loader)
        scheduler.step(1 - val_f1)

        history.append({"epoch": epoch, "train_loss": train_loss,
                         "train_acc": train_acc, "val_f1": val_f1, "val_auc": val_auc})
        print(f"  Época {epoch:2d}/{EPOCHS}  loss={train_loss:.4f}  "
              f"acc={train_acc:.4f}  val_F1={val_f1:.4f}  val_AUC={val_auc:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), save_path)
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= 3:
                print(f"  Early stopping en época {epoch}.")
                break

    elapsed = time.time() - t0
    print(f"\nMejor F1 validación: {best_f1:.4f} — Tiempo total: {elapsed:.1f}s")

    # Cargar mejor modelo y evaluar en test
    model.load_state_dict(torch.load(save_path, map_location=DEVICE))
    return model, elapsed, pd.DataFrame(history)


def get_test_metrics(model, test_loader, name, elapsed):
    f1, auc, preds, labels = evaluate(model, test_loader)
    print(f"\nResultados TEST — {name}")
    print(classification_report(labels, preds, target_names=["ham", "spam"]))
    from sklearn.metrics import precision_score, recall_score, accuracy_score
    return {
        "Modelo":    name,
        "Accuracy":  round(accuracy_score(labels, preds), 4),
        "Precision": round(precision_score(labels, preds), 4),
        "Recall":    round(recall_score(labels, preds),    4),
        "F1_Score":  round(f1, 4),
        "AUC_ROC":   round(auc, 4),
        "Tiempo_s":  round(elapsed, 2),
    }


# ─── Fine-tuning BERT-tiny ────────────────────────────────────────────────────

def train_bert_tiny(X_train_raw, y_train, X_val_raw, y_val, X_test_raw, y_test):
    """
    Fine-tuning de prajjwal1/bert-tiny (HuggingFace) para clasificación de spam.
    Requiere: pip install transformers
    """
    try:
        from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                                  Trainer, TrainingArguments)
        from torch.utils.data import Dataset as TorchDataset
    except ImportError:
        print("ADVERTENCIA: transformers no instalado. Saltando BERT-tiny.")
        return None

    MODEL_NAME = "prajjwal1/bert-tiny"
    BERT_DIR   = os.path.join(MODEL_DIR, "bert_tiny_model")
    os.makedirs(BERT_DIR, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    class BertSpamDataset(TorchDataset):
        def __init__(self, texts, labels):
            enc = tokenizer(list(texts), truncation=True, padding=True,
                            max_length=MAX_LEN, return_tensors="pt")
            self.input_ids      = enc["input_ids"]
            self.attention_mask = enc["attention_mask"]
            self.labels         = torch.tensor(labels, dtype=torch.long)

        def __len__(self): return len(self.labels)
        def __getitem__(self, idx):
            return {"input_ids":      self.input_ids[idx],
                    "attention_mask": self.attention_mask[idx],
                    "labels":         self.labels[idx]}

    train_ds = BertSpamDataset(X_train_raw, y_train)
    val_ds   = BertSpamDataset(X_val_raw,   y_val)
    test_ds  = BertSpamDataset(X_test_raw,  y_test)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"f1": f1_score(labels, preds)}

    args = TrainingArguments(
        output_dir=BERT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        seed=RANDOM_SEED,
        logging_steps=50,
    )

    trainer = Trainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    t0 = time.time()
    print("\nFine-tuning BERT-tiny...")
    trainer.train()
    elapsed = time.time() - t0

    # Evaluación en test
    preds_raw = trainer.predict(test_ds)
    preds     = np.argmax(preds_raw.predictions, axis=-1)
    f1  = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, preds_raw.predictions[:, 1])

    print(f"\nResultados TEST — BERT-tiny")
    print(classification_report(y_test, preds, target_names=["ham", "spam"]))

    # Guardar modelo y tokenizer
    trainer.save_model(BERT_DIR)
    tokenizer.save_pretrained(BERT_DIR)
    print(f"Modelo guardado en: {BERT_DIR}")

    from sklearn.metrics import precision_score, recall_score, accuracy_score
    return {
        "Modelo":    "BERT-tiny (fine-tuning)",
        "Accuracy":  round(accuracy_score(y_test, preds), 4),
        "Precision": round(precision_score(y_test, preds), 4),
        "Recall":    round(recall_score(y_test, preds),    4),
        "F1_Score":  round(f1, 4),
        "AUC_ROC":   round(auc, 4),
        "Tiempo_s":  round(elapsed, 2),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("ENTRENAMIENTO DE MODELOS DEEP LEARNING")
    print("=" * 60)

    # Cargar datos crudos (texto limpio)
    X_train_raw = joblib.load(os.path.join(PROC_DIR, "X_train_raw.joblib"))
    X_val_raw   = joblib.load(os.path.join(PROC_DIR, "X_val_raw.joblib"))
    X_test_raw  = joblib.load(os.path.join(PROC_DIR, "X_test_raw.joblib"))
    y_train     = joblib.load(os.path.join(PROC_DIR, "y_train.joblib"))
    y_val       = joblib.load(os.path.join(PROC_DIR, "y_val.joblib"))
    y_test      = joblib.load(os.path.join(PROC_DIR, "y_test.joblib"))

    # Crear datasets con vocabulario compartido
    train_ds = SpamDataset(X_train_raw, y_train)
    vocab    = train_ds.vocab
    val_ds   = SpamDataset(X_val_raw,   y_val,   vocab=vocab)
    test_ds  = SpamDataset(X_test_raw,  y_test,  vocab=vocab)
    joblib.dump(vocab, os.path.join(MODEL_DIR, "vocab.joblib"))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

    results = []

    # ── CNN 1D ────────────────────────────────────────────────────────────────
    cnn = CNN1D()
    cnn, elapsed_cnn, _ = train_torch_model(
        cnn, train_loader, val_loader,
        "CNN 1D", os.path.join(MODEL_DIR, "cnn1d_model.pt")
    )
    results.append(get_test_metrics(cnn, test_loader, "CNN 1D", elapsed_cnn))

    # ── BiLSTM ────────────────────────────────────────────────────────────────
    bilstm = BiLSTM()
    bilstm, elapsed_bi, _ = train_torch_model(
        bilstm, train_loader, val_loader,
        "BiLSTM", os.path.join(MODEL_DIR, "bilstm_model.pt")
    )
    results.append(get_test_metrics(bilstm, test_loader, "BiLSTM", elapsed_bi))

    # ── BERT-tiny ─────────────────────────────────────────────────────────────
    bert_result = train_bert_tiny(
        X_train_raw, y_train, X_val_raw, y_val, X_test_raw, y_test
    )
    if bert_result:
        results.append(bert_result)

    # Guardar reporte
    df = pd.DataFrame(results)
    path = os.path.join(REPORT_DIR, "dl_results.csv")
    df.to_csv(path, index=False)
    print(f"\nReporte guardado en: {path}")
    print("\nResumen final:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
