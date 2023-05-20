import torch
import torch.nn as nn
from lenet import LeNet
import matplotlib.pyplot as plt

# Define model and load weights
model = LeNet()
model.load_state_dict(torch.load('lenet.pth'))
model.eval()

# plot all filters in all conv layers
for name, module in model.named_modules():
    if isinstance(module, nn.Conv2d):
        filters = module.weight.data
        in_ch_num = module.in_channels
        out_ch_num = module.out_channels
        fig = plt.figure(figsize=(in_ch_num, out_ch_num))
        fig.suptitle(f'{name} Filters', fontsize=20)
        for i, filters1 in enumerate(filters):
            for j, filter in enumerate(filters1):
                ax = fig.add_subplot(out_ch_num, in_ch_num, i*in_ch_num+j+1)
                ax.imshow(filter.detach(), cmap='gray')
                ax.axis('off')

        plt.savefig(f'{name}_filters.png')