"""SleepNet: a CNN reads each 30 s EEG epoch, a bidirectional GRU adds the surrounding context.

Sleep stages come in runs, and the epochs on either side of an ambiguous one (N1, REM)
are the best clue to what it is, so the GRU runs over a window of consecutive epochs.
"""
import torch
from torch import nn

from .preprocessing import EPOCH_SECONDS, SAMPLE_RATE, STAGES

SEQ_LEN = 20  # epochs per window (10 minutes): what the network is trained on


def _conv_bn_relu(c_in, c_out, kernel=7):
    return [
        nn.Conv1d(c_in, c_out, kernel, padding=kernel // 2, bias=False),
        nn.BatchNorm1d(c_out),
        nn.ReLU(inplace=True),
    ]


class EpochEncoder(nn.Module):
    """1-D CNN: one 30 s epoch (3000 samples) -> a feature vector."""

    def __init__(self, feat_dim=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            # a wide first kernel (0.5 s) with stride picks up slow waves and spindles
            nn.Conv1d(1, 32, 50, stride=6, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(8),
            nn.Dropout(dropout),
            *_conv_bn_relu(32, 64), *_conv_bn_relu(64, 64), nn.MaxPool1d(4),
            *_conv_bn_relu(64, feat_dim), *_conv_bn_relu(feat_dim, feat_dim),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.net(x)


class SleepNet(nn.Module):
    """(batch, n_epochs, 3000) normalised EEG -> (batch, n_epochs, 5) stage logits."""

    def __init__(self, n_classes=len(STAGES), feat_dim=128, hidden=64, dropout=0.3):
        super().__init__()
        self.args = {"n_classes": n_classes, "feat_dim": feat_dim,
                     "hidden": hidden, "dropout": dropout}
        self.encoder = EpochEncoder(feat_dim, dropout)
        self.gru = nn.GRU(feat_dim, hidden, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(2 * hidden, n_classes)

    def forward(self, x):
        batch, length, samples = x.shape
        feats = self.encoder(x.reshape(batch * length, 1, samples)).reshape(batch, length, -1)
        context, _ = self.gru(feats)
        return self.head(self.dropout(context))


@torch.inference_mode()
def predict_proba(model, epochs, seq_len=SEQ_LEN, batch_windows=16):
    """Stage probabilities for normalised epochs of shape (n, 3000), as a numpy (n, 5) array.

    Recordings longer than one window are cut into 50%-overlapping windows and the
    overlapping predictions are averaged. This single function scores the test set in
    train.py *and* answers /predict, so the reported metrics describe what the API does.
    """
    model.eval()
    device = next(model.parameters()).device
    epochs = torch.as_tensor(epochs, dtype=torch.float32)
    n = epochs.shape[0]

    if n <= seq_len:
        starts, length = [0], n
    else:
        starts, length = list(range(0, n - seq_len + 1, seq_len // 2)), seq_len
        if starts[-1] != n - seq_len:
            starts.append(n - seq_len)  # make sure the tail is covered

    total, counts = None, torch.zeros(n, 1)
    for i in range(0, len(starts), batch_windows):
        chunk = starts[i:i + batch_windows]
        windows = torch.stack([epochs[s:s + length] for s in chunk]).to(device)
        probs = torch.softmax(model(windows), dim=-1).cpu()
        if total is None:
            total = torch.zeros(n, probs.shape[-1])
        for start, window_probs in zip(chunk, probs):
            total[start:start + length] += window_probs
            counts[start:start + length] += 1
    return (total / counts).numpy()


def save_checkpoint(path, model, signal_scale, metrics=None, extra=None):
    """Write a checkpoint made only of tensors and plain Python values.

    That is what lets load_checkpoint use weights_only=True (no arbitrary unpickling).
    """
    torch.save({
        "model_args": model.args,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "stages": list(STAGES),
        "signal_scale": float(signal_scale),
        "seq_len": SEQ_LEN,
        "sample_rate": SAMPLE_RATE,
        "epoch_seconds": EPOCH_SECONDS,
        "parameters": sum(p.numel() for p in model.parameters()),
        "metrics": metrics or {},
        **(extra or {}),
    }, path)


def load_checkpoint(path):
    """Return (model in eval mode, metadata dict) for a checkpoint from save_checkpoint."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = SleepNet(**ckpt["model_args"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    meta = {key: value for key, value in ckpt.items() if key != "state_dict"}
    return model, meta
