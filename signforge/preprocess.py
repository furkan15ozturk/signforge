import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from signforge.dataset import SignTranslationDataset


def create_token_set(data):
    text_token_set = set()
    for sample in data:
        for token in sample["text"].split():
            text_token_set.add(token)
    return text_token_set


def create_vocab(text_token_set):
    vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
    for i, token in enumerate(sorted(text_token_set)):
        vocab[token] = i + 4
    return vocab


def get_min_max(data):
    frame_counts = [sample["sign"].shape[0] for sample in data]
    return min(frame_counts), max(frame_counts)


def compute_normalization_stats(dataset):
    all_features = torch.cat([sample["sign"] for sample in dataset], dim=0)
    mean = all_features.mean(dim=0)
    std = all_features.std(dim=0).clamp(min=1e-8)
    return mean, std


def make_collate_fn(vocab, mean_sign, std_sign):
    def collate_fn(batch):
        signs = [item["sign"] for item in batch]
        padded_sign = pad_sequence(signs, padding_value=0, batch_first=True)
        mask_sign = (padded_sign != 0).any(dim=-1).float()

        normalized = (padded_sign - mean_sign) / std_sign
        normalized = normalized * mask_sign.unsqueeze(-1)

        padded_text = pad_sequence(
            [
                torch.tensor(
                    [vocab["<bos>"]]
                    + [vocab.get(t, vocab["<unk>"]) for t in item["text"].split()]
                    + [vocab["<eos>"]],
                    dtype=torch.long,
                )
                for item in batch
            ],
            padding_value=0,
            batch_first=True,
        )

        mask_text = (padded_text != 0).float()

        return {
            "sign": normalized,
            "text": padded_text,
            "mask_sign": mask_sign,
            "mask_text": mask_text,
            "idx": torch.tensor([item["idx"] for item in batch], dtype=torch.long),
        }

    return collate_fn


def build_dataloaders(
    train_ds: SignTranslationDataset,
    dev_ds: SignTranslationDataset,
    collate_fn,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = False,
):
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, dev_loader
