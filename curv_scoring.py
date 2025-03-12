import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from resnet_gn import ResNetGN18

device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

criterion = nn.CrossEntropyLoss()

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

train = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_test)
trainloader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=False)

def get_regularized_curvature_for_batch(net, batch_data, batch_labels, h=1e-3, niter=10, temp=1):
    num_samples = batch_data.shape[0]
    net.eval()
    net.zero_grad()
    regr = torch.zeros(num_samples)
    eigs = torch.zeros(num_samples)
    for _ in range(niter):
        v = torch.randint_like(batch_data, high=2).cuda()
        # Generate Rademacher random variables
        for v_i in v:
            v_i[v_i == 0] = -1

        v = h * (v + 1e-7)

        batch_data.requires_grad_()
        outputs_pos = net(batch_data + v)
        outputs_orig = net(batch_data)
        loss_pos = criterion(outputs_pos / temp, batch_labels)
        loss_orig = criterion(outputs_orig / temp, batch_labels)
        grad_diff = torch.autograd.grad((loss_pos-loss_orig), batch_data )[0]

        regr += grad_diff.reshape(grad_diff.size(0), -1).norm(dim=1).cpu().detach()

        net.zero_grad()
        if batch_data.grad is not None:
            batch_data.grad.zero_()

    curv_estimate = regr / niter
    return curv_estimate

def save_curv_scores_for_net(net, path, epoch=0):
    scores = torch.zeros(len(train))
    labels = torch.zeros_like(scores, dtype=torch.long)
    total = 0

    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, targets = data
        inputs, targets = inputs.to(device), targets.to(device)

        start_idx = total
        stop_idx = total + len(targets)
        idxs = [j for j in range(start_idx, stop_idx)]
        total = stop_idx

        inputs.requires_grad = True
        
        curv_estimate = get_regularized_curvature_for_batch(net, inputs, targets)
        scores[idxs] = curv_estimate.detach().clone().cpu()
        labels[idxs] = targets.cpu().detach()

    torch.save(scores, f'{path}/epoch_{epoch}_scores.pt')

def get_scores_for_model(model_dir, out_dir):
    print(f'Scoring for model at {model_dir} and saving at {out_dir}')
    for i in range(10):
        print(f'Saving scores for epoch {(i+1)*4}.')
        model_path = f'{model_dir}/epoch_{(i+1)*4}.pth'
        net = ResNetGN18(10)
        net.load_state_dict(torch.load(model_path, weights_only=True), strict=False)
        net.to(device)
        score_path = f'{out_dir}'

        save_curv_scores_for_net(net, score_path, (i+1)*4)

# get_scores_for_model('./standard', './curv_scores/standard')
get_scores_for_model('./eps10', './curv_scores/eps10')
