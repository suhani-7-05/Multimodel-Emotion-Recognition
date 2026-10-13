import os
import torch
import librosa
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from tqdm.auto import tqdm
from sklearn.metrics import f1_score

TARGET = ['neutral','joy','anger','sadness','surprise']

# ---------- AUDIO ----------
def get_audio(path):
    try:
        y, sr = librosa.load(path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)
        rms = librosa.feature.rms(y=y).mean()
        feat = np.concatenate([mfcc,[0.0,rms]])
        return (feat - feat.mean())/(feat.std()+1e-6)
    except:
        return np.zeros(15)

# ---------- DATA ----------
def load_df(base, csv):
    df = pd.read_csv(os.path.join(base,csv))
    df = df[df['Emotion'].isin(TARGET)].reset_index(drop=True)
    df['shift']=0
    return df

def load_feats(df, split, base):
    feats={}
    path=os.path.join(base,f"{split}_wav")
    for _,r in tqdm(df.iterrows(), total=len(df)):
        key=f"dia{r['Dialogue_ID']}_utt{r['Utterance_ID']}"
        feats[key]=get_audio(os.path.join(path,f"{key}.wav"))
    return feats

# ---------- DATASET ----------
class DS(Dataset):
    def __init__(self,df,feats):
        self.df=df
        self.f=feats
        self.tok=BertTokenizer.from_pretrained('bert-base-uncased')
        self.map={e:i for i,e in enumerate(TARGET)}

    def __len__(self): return len(self.df)

    def __getitem__(self,i):
        r=self.df.iloc[i]
        key=f"dia{r['Dialogue_ID']}_utt{r['Utterance_ID']}"
        enc=self.tok(str(r['Utterance']),return_tensors='pt',
                     padding='max_length',truncation=True,max_length=128)

        return {
            'ids':enc['input_ids'].squeeze(),
            'mask':enc['attention_mask'].squeeze(),
            'aud':torch.tensor(self.f[key],dtype=torch.float32),
            'y':torch.tensor(self.map[r['Emotion']])
        }

# ---------- MODEL ----------
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert=BertModel.from_pretrained('bert-base-uncased')

        # freeze except last 2 layers
        for name,p in self.bert.named_parameters():
            if "encoder.layer.10" in name or "encoder.layer.11" in name:
                p.requires_grad=True
            else:
                p.requires_grad=False

        self.t_proj=nn.Linear(768,128)
        self.a_proj=nn.Linear(15,128)

        self.w_t=nn.Linear(128,1)
        self.w_a=nn.Linear(128,1)

        self.fc=nn.Sequential(
            nn.Linear(128,256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256,5)
        )

    def forward(self,b):
        t=self.bert(input_ids=b['ids'],attention_mask=b['mask']).pooler_output
        t=self.t_proj(t)
        a=self.a_proj(b['aud'])

        s=torch.cat([self.w_t(t),self.w_a(a)],dim=1)
        w=torch.softmax(s,dim=1)

        f=w[:,0:1]*t + w[:,1:2]*a
        return self.fc(f)

# ---------- LOSS ----------
class FocalLoss(nn.Module):
    def __init__(self,gamma=2):
        super().__init__()
        self.g=gamma

    def forward(self,logits,y):
        ce=F.cross_entropy(logits,y,reduction='none')
        pt=torch.exp(-ce)
        return ((1-pt)**self.g * ce).mean()

import argparse

# ---------- TRAIN ----------
def train(args=None):
    if args is None:
        base = os.environ.get("DATA_DIR", "/workspace/data")
        batch_size = 64
        epochs = 40
        lr = 2e-5
        save_path = "best3.pt"
    else:
        base = args.data_dir
        batch_size = args.batch_size
        epochs = args.epochs
        lr = args.lr
        save_path = args.save_path

    print(f"Loading data from: {base}")
    if not os.path.exists(base):
        raise FileNotFoundError(
            f"Dataset directory '{base}' does not exist! Please ensure MELD dataset files "
            f"(train_sent_emo.csv, dev_sent_emo.csv, train_wav/, dev_wav/) are placed in '{base}' "
            f"or specify with --data_dir <path>."
        )

    train_df = load_df(base, "train_sent_emo.csv")
    dev_df = load_df(base, "dev_sent_emo.csv")

    print(f"Loaded {len(train_df)} training and {len(dev_df)} validation utterances.")
    print("Extracting audio features...")
    train_f = load_feats(train_df, "train", base)
    dev_f = load_feats(dev_df, "dev", base)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    if use_cuda:
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    num_workers = 2 if use_cuda else 0
    tr = DataLoader(DS(train_df, train_f), batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=use_cuda)
    dv = DataLoader(DS(dev_df, dev_f), batch_size=batch_size, num_workers=num_workers, pin_memory=use_cuda)

    m = Model().to(device)

    opt = optim.AdamW(filter(lambda p: p.requires_grad, m.parameters()), lr=lr)
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=3)
    loss_fn = FocalLoss()

    best = 0

    for e in range(epochs):
        m.train()
        for b in tqdm(tr, desc=f"Epoch {e+1}/{epochs}"):
            for k in b: b[k] = b[k].to(device)
            opt.zero_grad()
            out = m(b)
            loss = loss_fn(out, b['y'])
            loss.backward()
            opt.step()

        # val
        m.eval()
        p = []; y = []
        with torch.no_grad():
            for b in dv:
                for k in b: b[k] = b[k].to(device)
                out = m(b)
                p += torch.argmax(out, 1).cpu().tolist()
                y += b['y'].cpu().tolist()

        f1 = f1_score(y, p, average='weighted')
        print(f"\nEpoch {e+1} F1: {f1:.4f}")
        sch.step(f1)

        if f1 > best:
            best = f1
            torch.save(m.state_dict(), save_path)
            print(f"✓ Saved new best checkpoint to {save_path}")

    print(f"Training Complete. Best Validation F1: {best:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Multimodal Emotion Recognition Model on DGX / GPU")
    parser.add_argument("--data_dir", type=str, default=os.environ.get("DATA_DIR", "/workspace/data"), help="Path to dataset directory")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--epochs", type=int, default=40, help="Number of epochs (default: 40)")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate (default: 2e-5)")
    parser.add_argument("--save_path", type=str, default="best3.pt", help="Path to save best checkpoint")
    args = parser.parse_args()
    train(args)