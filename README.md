This repository contains a ConvNeXt V2 implementation designed to work with 1D signals.
It can be loaded using Pytorch Hub.
We also share pretrained weights.
We trained the model on ECG signals from the Icentia11k dataset using a masking pretext task.

To load the pretrained model, use:
```python
model = torch.hub.load("cfauchereau/ecg-convnext", "ecg_convnext", pretrained=True)
```
