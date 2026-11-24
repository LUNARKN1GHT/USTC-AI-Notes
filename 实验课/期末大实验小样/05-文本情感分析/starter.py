"""Read IMDb, fit a word-count baseline, and inspect two Transformer training batches."""

import html
import re
import tarfile
from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "data" / "aclImdb_v1.tar.gz"
SEED = 42
VOCAB_LIMIT = 4000
MAX_LENGTH = 64
PAD, UNKNOWN, MASK = 0, 1, 2


def tokenize(text):
    text = html.unescape(text).replace("<br />", " ")
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())


def read_small_sample():
    groups = {
        name: []
        for name in ("train/neg", "train/pos", "test/neg", "test/pos", "train/unsup")
    }
    with tarfile.open(ARCHIVE, "r|gz") as archive:
        for member in archive:
            parts = member.name.split("/")
            if member.isfile() and len(parts) == 4 and member.name.endswith(".txt"):
                key = "/".join(parts[1:3])
                if key in groups:
                    groups[key].append(member.name)

    rng = np.random.default_rng(SEED)
    selections = {"train": [], "validation": [], "test": [], "unlabeled": []}
    for label, name in enumerate(("neg", "pos")):
        train_names = rng.permutation(sorted(groups[f"train/{name}"]))
        selections["train"].extend((str(n), label) for n in train_names[:500])
        selections["validation"].extend((str(n), label) for n in train_names[500:600])
        test_names = rng.permutation(sorted(groups[f"test/{name}"]))[:200]
        selections["test"].extend((str(n), label) for n in test_names)
    unlabeled = rng.permutation(sorted(groups["train/unsup"]))[:1000]
    selections["unlabeled"] = [(str(n), -1) for n in unlabeled]
    wanted = {name for rows in selections.values() for name, _ in rows}
    assert len(wanted) == sum(map(len, selections.values()))

    # Read selected files in archive order to avoid repeatedly seeking in gzip.
    texts = {}
    with tarfile.open(ARCHIVE, "r|gz") as archive:
        for member in archive:
            if member.name in wanted:
                texts[member.name] = archive.extractfile(member).read().decode("utf-8")
    assert set(texts) == wanted
    return {
        split: (
            [tokenize(texts[name]) for name, _ in rows],
            np.array([label for _, label in rows], dtype=np.int64),
        )
        for split, rows in selections.items()
    }


def make_vocabulary(samples):
    frequencies = Counter()
    for split in ("train", "unlabeled"):
        for words in samples[split][0]:
            frequencies.update(words)
    return {
        word: i + 3 for i, (word, _) in enumerate(frequencies.most_common(VOCAB_LIMIT))
    }


def bag_of_words(documents, vocabulary):
    features = np.zeros((len(documents), len(vocabulary) + 3), dtype=np.float64)
    for i, words in enumerate(documents):
        ids = [vocabulary.get(word, UNKNOWN) for word in words]
        features[i] = np.bincount(ids, minlength=features.shape[1])
    return features


def word_count_baseline(samples, vocabulary):
    features = bag_of_words(samples["train"][0], vocabulary)
    labels = samples["train"][1]
    # Multinomial naive Bayes with additive smoothing.
    counts = np.stack([features[labels == label].sum(axis=0) + 1 for label in (0, 1)])
    word_log_prob = np.log(counts / counts.sum(axis=1, keepdims=True))
    class_log_prob = np.log(np.bincount(labels, minlength=2) / len(labels))
    majority = int(np.bincount(labels).argmax())
    for split in ("validation", "test"):
        x = bag_of_words(samples[split][0], vocabulary)
        y = samples[split][1]
        scores = np.einsum("nv,kv->nk", x, word_log_prob) + class_log_prob
        assert np.isfinite(scores).all()
        prediction = scores.argmax(axis=1)
        print(
            f"{split}: majority accuracy={(y == majority).mean():.3f}, "
            f"word-count NB accuracy={(prediction == y).mean():.3f}"
        )


def encode_batch(documents, vocabulary):
    ids = torch.full((len(documents), MAX_LENGTH), PAD, dtype=torch.long)
    for i, words in enumerate(documents):
        encoded = [vocabulary.get(word, UNKNOWN) for word in words[:MAX_LENGTH]]
        ids[i, : len(encoded)] = torch.tensor(encoded, dtype=torch.long)
    return ids


class TinyEncoder(nn.Module):
    def __init__(self, vocabulary_size):
        super().__init__()
        self.tokens = nn.Embedding(vocabulary_size, 64, padding_idx=PAD)
        self.positions = nn.Embedding(MAX_LENGTH, 64)
        layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=128, dropout=0.0, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=2, enable_nested_tensor=False
        )

    def forward(self, ids):
        positions = torch.arange(ids.shape[1], device=ids.device)
        embedding = self.tokens(ids) + self.positions(positions)[None]
        return self.encoder(embedding, src_key_padding_mask=(ids == PAD))


def check_transformer_batches(samples, vocabulary):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    encoder = TinyEncoder(len(vocabulary) + 3)
    initial_state = deepcopy(encoder.state_dict())
    classifier = nn.Linear(64, 2)
    optimizer = torch.optim.Adam(
        [*encoder.parameters(), *classifier.parameters()], lr=0.001
    )
    indices = rng.choice(len(samples["train"][1]), size=16, replace=False)
    ids = encode_batch([samples["train"][0][i] for i in indices], vocabulary)
    labels = torch.tensor(samples["train"][1][indices])
    hidden = encoder(ids)
    valid = (ids != PAD).unsqueeze(-1)
    pooled = (hidden * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
    logits = classifier(pooled)
    loss = nn.functional.cross_entropy(logits, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(
        f"Classification batch: ids={tuple(ids.shape)}, hidden={tuple(hidden.shape)}, "
        f"logits={tuple(logits.shape)}, initial loss={loss.item():.3f}"
    )

    # Start the MLM batch from the original encoder initialization.
    encoder.load_state_dict(initial_state)
    language_head = nn.Linear(64, len(vocabulary) + 3)
    optimizer = torch.optim.Adam(
        [*encoder.parameters(), *language_head.parameters()], lr=0.001
    )
    ids = encode_batch(samples["unlabeled"][0][:16], vocabulary)
    chosen = (torch.rand(ids.shape) < 0.15) & (ids >= 3)
    assert chosen.any()
    targets = ids.clone()
    targets[~chosen] = -100
    masked = ids.clone()
    masked[chosen] = MASK
    logits = language_head(encoder(masked))
    loss = nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(
        f"MLM batch: ids={tuple(masked.shape)}, logits={tuple(logits.shape)}, "
        f"predicted tokens={int(chosen.sum())}, initial loss={loss.item():.3f}"
    )
    print(
        "The two batches check tensor shapes and gradients. "
        "Extend them into training loops to measure learned model performance."
    )


def main():
    torch.set_num_threads(2)
    samples = read_small_sample()
    print("Sample sizes:", {split: len(rows[0]) for split, rows in samples.items()})
    print("First training tokens:", samples["train"][0][0][:20])
    vocabulary = make_vocabulary(samples)
    print(f"Vocabulary size: {len(vocabulary) + 3} including PAD/UNKNOWN/MASK")
    word_count_baseline(samples, vocabulary)
    check_transformer_batches(samples, vocabulary)


if __name__ == "__main__":
    main()
