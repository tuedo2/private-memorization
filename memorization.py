import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from resnet_gn import ResNetGN18

def subset_train(seed, subset_ratio):
#   jrng = random.PRNGKey(seed)

  step_size = 0.1
  num_epochs = 10
  batch_size = 128
  momentum_mass = 0.9

#   num_train_total = mnist_data['train_images'].shape[0]
  num_train = int(num_train_total * subset_ratio)
  num_batches = int(np.ceil(num_train / batch_size))

#   rng = npr.RandomState(seed)
#   subset_idx = rng.choice(num_train_total, size=num_train, replace=False)
#   train_images = mnist_data['train_images'][subset_idx]
#   train_labels = mnist_data['train_labels'][subset_idx]

  def data_stream(shuffle=True):
    while True:
    #   perm = rng.permutation(num_train)
      for i in range(num_batches):
        batch_idx = perm[i * batch_size:(i + 1) * batch_size]
        yield train_images[batch_idx], train_labels[batch_idx]

  batches = data_stream()

#   opt_init, opt_update, get_params = optimizers.momentum(step_size, mass=momentum_mass)

  @jit
  def update(i, opt_state, batch):
    params = get_params(opt_state)
    # return opt_update(i, grad(loss)(params, batch), opt_state)

#   _, init_params = init_random_params(jrng, (-1, 28 * 28))
#   opt_state = opt_init(init_params)
#   itercount = itertools.count()

  for epoch in range(num_epochs):
    for _ in range(num_batches):
      opt_state = update(next(itercount), opt_state, next(batches))

#   params = get_params(opt_state)
#   trainset_correctness = batch_correctness(
#       params, (mnist_data['train_images'], mnist_data['train_labels']))
#   testset_correctness = batch_correctness(
#       params, (mnist_data['test_images'], mnist_data['test_labels']))

  trainset_mask = np.zeros(num_train_total, dtype=np.bool)
  trainset_mask[subset_idx] = True
  return trainset_mask, np.asarray(trainset_correctness), np.asarray(testset_correctness)


def estimate_infl_mem():
  n_runs = 2000
  subset_ratio = 0.7

  results = []
  for i_run in tqdm(range(n_runs), desc=f'SS Ratio={subset_ratio:.2f}'):
    results.append(subset_train(i_run, subset_ratio))

  trainset_mask = np.vstack([ret[0] for ret in results])
  inv_mask = np.logical_not(trainset_mask)
  trainset_correctness = np.vstack([ret[1] for ret in results])
  testset_correctness = np.vstack([ret[2] for ret in results])

  print(f'Avg test acc = {np.mean(testset_correctness):.4f}')

  def _masked_avg(x, mask, axis=0, esp=1e-10):
    return (np.sum(x * mask, axis=axis) / np.maximum(np.sum(mask, axis=axis), esp)).astype(np.float32)

  def _masked_dot(x, mask, esp=1e-10):
    x = x.T.astype(np.float32)
    return (np.matmul(x, mask) / np.maximum(np.sum(mask, axis=0, keepdims=True), esp)).astype(np.float32)

  mem_est = _masked_avg(trainset_correctness, trainset_mask) - _masked_avg(trainset_correctness, inv_mask)
  infl_est = _masked_dot(testset_correctness, trainset_mask) - _masked_dot(testset_correctness, inv_mask)

  return dict(memorization=mem_est, influence=infl_est)
