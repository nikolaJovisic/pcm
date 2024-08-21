import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split

from dataset.csaw import CSAWDataset


def get_dataloaders(healthy_path, diseased_path, batch_size, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1"

    healthy_dataset = CSAWDataset(healthy_path, diseased=False)
    diseased_dataset = CSAWDataset(diseased_path, diseased=True)

    combined_dataset = torch.utils.data.ConcatDataset([healthy_dataset, diseased_dataset])

    len_healthy = len(healthy_dataset)
    len_diseased = len(diseased_dataset)

    weights = [1.0 / len_healthy] * len_healthy + [1.0 / len_diseased] * len_diseased

    total_len = len(combined_dataset)
    train_len = int(train_ratio * total_len)
    val_len = int(val_ratio * total_len)
    test_len = total_len - train_len - val_len

    train_dataset, val_dataset, test_dataset = random_split(
        combined_dataset, [train_len, val_len, test_len]
    )

    train_sampler = WeightedRandomSampler([weights[i] for i in train_dataset.indices], num_samples=train_len)
    val_sampler = WeightedRandomSampler([weights[i] for i in val_dataset.indices], num_samples=val_len)
    test_sampler = WeightedRandomSampler([weights[i] for i in test_dataset.indices], num_samples=test_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler)

    return train_loader, val_loader, test_loader
