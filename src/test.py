from torch.utils.data import DataLoader
from dataset import FundusSegmentationDataset

dataset = FundusSegmentationDataset()

for i in range(100):
    _, mask = dataset[i]
    print(i, sorted(mask.unique().tolist()))