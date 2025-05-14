from pathlib import Path
import ssl
import torch
import matplotlib.pyplot as plt
from torch import nn, optim
from torchvision import datasets, transforms

# paths & globals
out_dir = Path('./LAB 5/plots')
out_dir.mkdir(parents=True, exist_ok=True)
print('[INFO] Saving plots to', out_dir)

# baseline hyper‑params
a = 1e-3  # learning‑rate
b = 32    # batch‑size
c = 1     # hidden layers
d = 256   # hidden width
e = 'adam'  # optimiser
EPOCHS = 20  # default epochs
print(
    f"[INFO] Baseline → lr={a}, bs={b}, layers={c}, width={d}, opt={e}, epochs={EPOCHS}")

# data
ssl._create_default_https_context = ssl._create_unverified_context
print('[INFO] Downloading / loading KMNIST…')
train_ds = datasets.KMNIST(root='data', train=True, download=True,
                           transform=transforms.ToTensor())
X = train_ds.data.view(-1, 784).float() / 255.
y = train_ds.targets.clone()
print('[DEBUG] Dataset size:', len(X))
perm = torch.randperm(len(X))
val_len = int(0.2 * len(X))
val_idx, tr_idx = perm[:val_len], perm[val_len:]
Xtr, Ytr = X[tr_idx], y[tr_idx]
Xv, Yv = X[val_idx], y[val_idx]
print(f'[DEBUG] Train split → {len(Xtr)}, Val split → {len(Xv)}')

# model builder
def mlp(layers: int, width: int) -> nn.Sequential:
    seq = []
    p = 784
    for i in range(layers):
        print(
            f'[TRACE] Adding Linear({p},{width}) + ReLU layer {i + 1}/{layers}')
        seq += [nn.Linear(p, width), nn.ReLU()]
        p = width
    seq.append(nn.Linear(p, 10))
    return nn.Sequential(*seq)

# training helpers
def run(cfg: dict, *, epochs: int | None = None, log: bool = False):
    n_epochs = epochs if epochs is not None else cfg.get('epochs', EPOCHS)
    print('[INFO] Starting run with cfg:', cfg, 'epochs=', n_epochs)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = mlp(cfg['layers'], cfg['width']).to(dev)

    if cfg['opt'] == 'sgd':
        opt = optim.SGD(net.parameters(), lr=cfg['lr'])
    elif cfg['opt'] == 'momentum':
        opt = optim.SGD(net.parameters(), lr=cfg['lr'], momentum=0.9)
    else:
        opt = optim.Adam(net.parameters(), lr=cfg['lr'])
    lossfn = nn.CrossEntropyLoss()

    Xt, Yt = Xtr.to(dev), Ytr.to(dev)
    Xv_, Yv_ = Xv.to(dev), Yv.to(dev)
    step_loss, val_acc = [], []
    for ep in range(n_epochs):
        net.train()
        epoch_loss = 0.0
        if log:
            print(f'[EPOCH {ep + 1:02d}/{n_epochs}] Training…')
        for s in range(0, len(Xt), cfg['bs']):
            xb, yb = Xt[s:s + cfg['bs']], Yt[s:s + cfg['bs']] #random batches
            loss = lossfn(net(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step_loss.append(loss.item())
            epoch_loss += loss.item()
        net.eval()
        with torch.no_grad():
            acc = (net(Xv_).argmax(1) == Yv_).float().mean().item() * 100
            val_acc.append(acc)
        if log:
            avg_loss = epoch_loss / ((len(Xt) // cfg['bs']) + 1)
            print(
                f'[EPOCH {ep + 1:02d}] avg_train_loss={avg_loss:.4f} val_acc={acc:.2f}%')
    best = max(val_acc) if val_acc else float('nan')
    print('[INFO] Run finished – best val_acc={:.2f}%'.format(best))
    return step_loss, val_acc


def run_tv(cfg: dict, *, epochs: int | None = None):
    n_epochs = epochs if epochs is not None else cfg.get('epochs', EPOCHS)
    print('[INFO] Starting run_tv with cfg:', cfg, 'epochs=', n_epochs)

    tr, va = [], []
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = mlp(cfg['layers'], cfg['width']).to(dev)
    opt = optim.Adam(net.parameters(), lr=cfg['lr'])
    lossfn = nn.CrossEntropyLoss()

    Xt, Yt = Xtr.to(dev), Ytr.to(dev)
    Xv_, Yv_ = Xv.to(dev), Yv.to(dev)
    for ep in range(n_epochs):
        net.train()
        epoch_loss = 0.0
        for s in range(0, len(Xt), cfg['bs']):
            loss = lossfn(net(Xt[s:s + cfg['bs']]), Yt[s:s + cfg['bs']])
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        net.eval()
        with torch.no_grad():
            tr_acc = (net(Xt).argmax(1) == Yt).float().mean().item() * 100
            va_acc = (net(Xv_) .argmax(1) == Yv_).float().mean().item() * 100
            tr.append(tr_acc)
            va.append(va_acc)
        print(f'[EPOCH {ep + 1:02d}/{n_epochs}] loss={epoch_loss/((len(Xt)//cfg["bs"])+1):.4f} train_acc={tr_acc:.2f}% val_acc={va_acc:.2f}%')
    print('[INFO] run_tv finished')
    return tr, va


# baseline run
print('[INFO] Baseline run…')
base = {'lr': a, 'bs': b, 'layers': c, 'width': d, 'opt': e}
steps, acc = run(base, log=True)
plt.plot(steps)
plt.xlabel('step')
plt.ylabel('loss')
plt.tight_layout()
plt.savefig(out_dir / 'loss_per_step.png')
plt.close()
print('[INFO] Saved loss_per_step.png')

tr, va = run_tv(base)
plt.plot(tr, label='train')
plt.plot(va, label='val')
plt.xlabel('epoch')
plt.ylabel('acc%')
plt.legend()
plt.tight_layout()
plt.savefig(out_dir / 'acc_train_val.png')
plt.close()
print('[INFO] Saved acc_train_val.png')

# compare helper


def compare(name: str, values: list, cfg_func, *, epochs: int):
    """Run a sweep over *values* changing *name*; plot val‑acc curves."""
    print(
        '[INFO] Comparing',
        name,
        'over values',
        values,
        f'for {epochs} epoch(s)')
    for v in values:
        cfg = cfg_func(v)
        _, va = run(cfg, epochs=epochs)
        print(f'[RESULT] {name}={v} final acc {va[-1]:.2f}%')
        plt.plot(va, label=str(v))
    plt.xlabel('epoch')
    plt.ylabel('val acc%')
    plt.legend()
    plt.tight_layout()
    png = out_dir / f'{name}_compare.png'
    plt.savefig(png)
    plt.close()
    print('[INFO] Saved', png)


# sweeps with custom epochs
compare('lr',
        [1e-4,
         1e-3,
         1e-2],
        lambda v: {'lr': v,
                   'bs': b,
                   'layers': c,
                   'width': d,
                   'opt': e},
        epochs=30)
compare('batch',
        [16,
         32,
         128],
        lambda v: {'lr': a,
                   'bs': v,
                   'layers': c,
                   'width': d,
                   'opt': e},
        epochs=1)   # <‑‑ bs=1 only 1 epoch
compare('layers', [0, 1, 10], lambda v: {
        'lr': a, 'bs': b, 'layers': v, 'width': d, 'opt': e}, epochs=8)
compare('opt', ['sgd', 'momentum', 'adam'], lambda v: {
        'lr': a, 'bs': b, 'layers': c, 'width': d, 'opt': v}, epochs=10)

print('[DONE] All experiments complete', out_dir)
