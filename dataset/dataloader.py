import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset.csaw import CSAWDataset


def get_dataloader(healthy_path, diseased_path, batch_size):
    healthy_dataset = CSAWDataset(healthy_path, diseased=False)
    diseased_dataset = CSAWDataset(diseased_path, diseased=True)

    combined_dataset = torch.utils.data.ConcatDataset(
        [healthy_dataset, diseased_dataset]
    )

    len_healthy = len(healthy_dataset)
    len_diseased = len(diseased_dataset)

    weights = [1.0 / len_healthy] * len_healthy + [1.0 / len_diseased] * len_diseased

    sampler = WeightedRandomSampler(
        weights, num_samples=len(combined_dataset), replacement=True
    )

    return DataLoader(combined_dataset, batch_size=batch_size, sampler=sampler)
