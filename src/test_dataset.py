from torch.utils.data import DataLoader
from dataset import FundusSegmentationDataset

dataset = FundusSegmentationDataset()

print("dataset size:", len(dataset))

image, mask = dataset[0]

print("image shape:", image.shape)
print("mask shape:", mask.shape)

print("mask unique values:", mask.unique())

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True
)

batch_images, batch_masks = next(iter(loader))

print("batch image shape:", batch_images.shape)
print("batch mask shape:", batch_masks.shape)