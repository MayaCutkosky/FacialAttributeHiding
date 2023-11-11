# Dataset Loaders

## Datasets:

### Celeb-a:
- Availible at:https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
- Download dataset
- Use dataset as dataset dir

## Data loaders:

### tensorflow:
- CelebA(attributes, identities, directory)
- Get training dataset with
```python
from Dataset_loaders.datasets import CalebA
dset = CelebA(True,False)
dset.get_dset('train')
```
### torch:
- Still under construction
- torchvision has a dataloader for the Celeba dataset
