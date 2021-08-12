import torch
import torch.optim as optim
import numpy as np
import time
from tqdm import tqdm
import torch.nn as nn
from . import fom


def weights_init(model):
    if isinstance(model, nn.Linear):
        nn.init.kaiming_normal_(model.weight)
        nn.init.normal_(model.bias)


def bce_training(df, X, net, device='cpu', epochs=200, batch_size=2**10, lr=1, verbose=False, progress_bar=True, **kwargs):
    current_device = next(net.parameters()).device
    print(f'training on {device}')
    net.to(device)

    loss_fn = nn.BCELoss(reduction='none').to(device)

    optimizer = optim.SGD(net.parameters(), lr=lr, **kwargs)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    train_data = torch.from_numpy(X).to(device)
    train_labels = torch.from_numpy(df.labels.values).to(device)
    train_weights = torch.from_numpy(df.weights.values).to(device)
    idx_arr = np.arange(train_data.shape[0])

    loss_list = []

    start = time.time()

    if progress_bar:
        iterator = tqdm(range(epochs), desc='training network', total=epochs, unit='epochs', leave=True)
        print_fun = iterator.set_description
    else:
        iterator = range(epochs)
        print_fun = print

    for epoch in iterator:  # loop over the dataset multiple times
        running_loss = 0.0

        # randomly iterate through data
        np.random.shuffle(idx_arr)
        idx_iterate = np.split(idx_arr, np.arange(batch_size, len(df), batch_size))

        for i, idx in enumerate(idx_iterate, 0):
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(train_data[idx, :])
            labels = train_labels[idx]
            weights = train_weights[idx]

            loss = torch.sum(loss_fn(outputs[:, 0], labels) * weights) / weights.sum()

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

        average_loss = running_loss / (i + 1)

        if verbose or progress_bar:
            print_fun('[%d, %5d] loss: %.5f' % (epoch + 1, i + 1, average_loss))
        else:
            if (epoch % int(epochs / 20) == 0) or (epoch == epochs - 1):
                print_fun('[%d, %5d] loss: %.5f' % (epoch + 1, i + 1, average_loss))

        loss_list.append(average_loss)
        scheduler.step(average_loss)
        running_loss = 0.0

    training_time = time.time() - start

    if not progress_bar:
        print(f'Finished Training in {training_time/60:{.3}f} minutes')

    # move network back to previous device
    net.to(current_device)

    return net, loss_list


def punzi_training(df, X, net, n_masses, n_gen_signal, target_lumi, device='cpu', epochs=500, lr=0.01, scaling=1, verbose=False, progress_bar=True, **kwargs):
    current_device = next(net.parameters()).device
    print(f'training on {device}')
    net.to(device)

    optimizer = optim.SGD(net.parameters(), lr=lr, **kwargs)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)

    inputs = torch.from_numpy(X).to(device)
    train_weights, labels = torch.from_numpy(df[['weights', 'labels']].values.astype('float32')).to(device).t()

    loss_list = []
    start = time.time()

    sig_sparse, bkg_sparse = fom.gen_sparse_matrices(df.gen_mass, df.range_idx_low, df.range_idx_high, df.sig_m_range, n_masses)

    if progress_bar:
        iterator = tqdm(range(epochs), desc='training network', total=epochs, unit='epochs', leave=True)
        print_fun = iterator.set_description
    else:
        iterator = range(epochs)
        print_fun = print

    for epoch in iterator:  # loop over the dataset multiple times
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)

        if torch.isnan(outputs).any():
            break

        loss = torch.mean(fom.punziloss(sig_sparse, bkg_sparse, outputs[:, 0], train_weights, n_gen_signal, target_lumi, scaling=scaling))

        loss.backward()
        optimizer.step()

        if verbose or progress_bar:
            print_fun('[%d] loss: %.5f' % (epoch + 1, loss.item()))
        else:
            if (epoch % int(epochs / 20) == 0) or (epoch == epochs - 1):
                print_fun('[%d] loss: %.5f' % (epoch + 1, loss.item()))

        loss_list.append(loss.item())
        scheduler.step(loss.item())

    training_time = time.time() - start

    if not progress_bar:
        print(f'Finished Training in {training_time/60:{.3}f} minutes')

    # move network back to previous device
    net.to(current_device)

    return net, loss_list
