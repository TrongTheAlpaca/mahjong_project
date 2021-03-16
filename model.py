# PyTorch 1.7.1
# Python 3.8.5 Windows 10
import time
from pathlib import Path

import numpy as np
import scipy.sparse
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from tqdm import trange, tqdm


##############################################
# CLASSES
##############################################


class DiscardDataset(torch.utils.data.Dataset):

    def __init__(self, data_path, n_rows: int = None, reverse=False, phase: int = None):

        # If n_rows = None -> get all

        if n_rows:
            print(f"Loading Dataset with {n_rows} rows", end=' ')
        else:
            print(f"Loading Dataset with all rows", end=' ')

        if phase is not None:
            print(f"(Phase {phase})", end='')
        print()

        paths = list(Path(data_path).iterdir())
        if reverse:
            paths.reverse()

        loaded_rows = 0
        temp_matrices = []
        paths_load_bar = tqdm(total=n_rows, unit='rows')
        for idx, path in enumerate(paths):

            arr = scipy.sparse.load_npz(path)

            if phase is not None and phase in [0, 1, 2]:
                phased_matrices = self.generate_phase_column(arr.toarray())
                arr = phased_matrices[phase]
                # arr = scipy.sparse.csr_matrix(phased_matrices[phase])

            loaded_rows += arr.shape[0]

            temp_matrices.append(arr)

            paths_load_bar.set_postfix(files_loaded=(idx + 1))  # Update Bar

            if n_rows and n_rows <= loaded_rows:
                break

            paths_load_bar.update(arr.shape[0])

        if 1 < len(temp_matrices):
            arr = scipy.sparse.vstack(temp_matrices, format='csr', dtype=np.int8)
        else:
            arr = temp_matrices[0]

        # Due to the option to filter phases, the array is sometimes not of type np.ndarray
        if type(arr) is not np.ndarray:
            arr = arr.toarray()

        if n_rows:
            arr = arr[:n_rows]

        # Finalize tqdm bar
        paths_load_bar.n = arr.shape[0]
        paths_load_bar.last_print_n = arr.shape[0]
        paths_load_bar.refresh()
        paths_load_bar.close()

        self.x_data = torch.FloatTensor(arr[:, :-1])  # Must be Float it seems
        self.y_data = torch.LongTensor(arr[:, -1])  # Must be Long it seems

    @staticmethod
    def generate_phase_column(array: np.array) -> np.array:
        # Begin with merging all pools together

        merged_discards = array[:, 238:]  # Discards
        merged_discards = np.sum(merged_discards, axis=1)

        phases = np.zeros([array.shape[0]])  # Early Game
        phases[(24 < merged_discards) & (merged_discards <= 48)] = 1  # Mid Game
        phases[(48 < merged_discards)] = 2  # End Game

        return array[(phases == 0)], array[(phases == 1)], array[(phases == 2)]

    def __len__(self):
        return self.x_data.shape[0]

    def __getitem__(self, idx):
        return {
            'X': self.x_data[idx],
            'y': self.y_data[idx]
        }


class Net(torch.nn.Module):
    """ Simple Feed-Forward Net """

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(11 * 34, 1028)
        self.fc2 = torch.nn.Linear(1028, 512)
        self.fc3 = torch.nn.Linear(512, 256)
        self.fc4 = torch.nn.Linear(256, 128)
        self.fc5 = torch.nn.Linear(128, 34)

    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        x = F.leaky_relu(x)
        x = self.fc3(x)
        x = F.leaky_relu(x)
        x = self.fc4(x)
        x = F.leaky_relu(x)
        x = self.fc5(x)
        x = F.leaky_relu(x)

        return x


##############################################
# PARAMETERS
##############################################

torch.cuda.current_device()
torch.cuda.device(0)
torch.cuda.device_count()
torch.cuda.get_device_name(0)
torch.cuda.is_available()

torch.manual_seed(0)
np.random.seed(0)

DEVICE = torch.device("cuda")
EPOCHS = 15
VALIDATION_SPLIT = .2
SHUFFLE_DATASET = True
BATCH_SIZE = 32
PHASE = None

if DEVICE.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')

##############################################
# DATASET
##############################################
print("\nLOADING DATASETS:\n")

# Single-Phase Datasets
train_dataset = DiscardDataset("E:/mahjong/discard_datasets/2019", 100_000, phase=PHASE)

# Creating data indices for training and validation splits:  (Inspired from https://stackoverflow.com/a/50544887)
dataset_size = len(train_dataset)
indices = list(range(dataset_size))
split = int(np.floor(VALIDATION_SPLIT * dataset_size))

if SHUFFLE_DATASET:
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating data samplers and loaders:
train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=valid_sampler)

# Multi-phases
# train_dataset = torch.utils.data.ConcatDataset([
#     DiscardDataset('data/2019_discard_dataset/', 1_000_000, phase=0),
#     DiscardDataset('data/2019_discard_dataset/', 1_000_000, phase=1),
#     DiscardDataset('data/2019_discard_dataset/', 1_000_000, phase=2)
# ])
#
# test_dataset = torch.utils.data.ConcatDataset([
#     DiscardDataset('data/2019_discard_dataset/', 1_000, phase=0, reverse=True),
#     DiscardDataset('data/2019_discard_dataset/', 1_000, phase=1, reverse=True),
#     DiscardDataset('data/2019_discard_dataset/', 1_000, phase=2, reverse=True)
# ])

# Test Dataset
test_dataset = DiscardDataset("E:/mahjong/discard_datasets/2019", 10_000, phase=PHASE, reverse=True)
test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=SHUFFLE_DATASET)

##############################################
# MODEL
##############################################
model = Net().to(DEVICE)
print("\nMODEL ARCHITECTURE:\n", model)

criterion = torch.nn.CrossEntropyLoss().to(DEVICE)  # Loss function
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

predicted = []

all_loss = []

##############################################
# TRAINING & VALIDATION
##############################################
print("\nRUNNING MODEL:")
for epoch in range(EPOCHS):

    time.sleep(0.3)
    print("EPOCH: ", epoch)

    ##############################################
    # TRAINING
    ##############################################
    sum_epoch_loss = 0.0
    sum_epoch_acc = 0.0

    model.train()
    tl = tqdm(train_loader, desc=f"  Training", position=0, disable=False)
    for batch_idx, batch in enumerate(tl):
        X = batch['X'].to(DEVICE)
        y = batch['y'].to(DEVICE)

        optimizer.zero_grad()

        outputs = model(X)  # Outputs float values for each class (do softmax on `outputs` to get distribution)
        loss = criterion(outputs, y)  # avg loss in batch -> No need for softmax if criterion = Cross Entropy Loss

        # sum_epoch_loss += loss.item()
        # all_loss.append(loss.item())

        predictions = torch.argmax(outputs, dim=1)
        num_correct_predictions = torch.sum(torch.eq(predictions, y)).item()
        batch_acc = num_correct_predictions / y.shape[0]  # y.shape[0] = batch_size

        # all_loss.append(loss.item())
        sum_epoch_loss += loss.item()
        sum_epoch_acc += batch_acc

        # tl.set_postfix(Loss="{:>5.4}".format(loss.item()), Accuracy=batch_acc)
        # # print("X:", torch.bincount(torch.argmax(outputs, dim=1)).tolist())
        # # print("y:", np.bincount(y).tolist(), '\n')

        loss.backward()  # compute gradients
        optimizer.step()  # update weights

    time.sleep(0.1)
    avg_acc = sum_epoch_acc / len(train_loader)  # average accuracy
    avg_loss = sum_epoch_loss / len(train_loader)  # average loss
    print(f"Training Accuracy:   {avg_acc:>6.3f}")
    print(f"Training Loss:       {avg_loss:>6.3f}")

    ##############################################
    # VALIDATION
    ##############################################
    sum_epoch_loss = 0.0
    sum_epoch_acc = 0.0
    model.eval()
    vl = tqdm(validation_loader, desc=f"Validation", position=0, disable=False)
    for batch_idx, batch in enumerate(vl):
        X = batch['X'].to(DEVICE)
        y = batch['y'].to(DEVICE)

        # optimizer.zero_grad()  # TODO: Needed?

        with torch.no_grad():  # Disables tracking of gradient
            outputs = model(X)
        loss = criterion(outputs, y)  # avg loss in batch

        predictions = torch.argmax(outputs, dim=1)
        num_correct_predictions = torch.sum(torch.eq(predictions, y)).item()
        batch_acc = num_correct_predictions / y.shape[0]  # y.shape[0] = batch_size

        # all_loss.append(loss.item())
        sum_epoch_loss += loss.item()
        sum_epoch_acc += batch_acc

    time.sleep(0.1)
    avg_acc = sum_epoch_acc / len(validation_loader)  # average accuracy
    avg_loss = sum_epoch_loss / len(validation_loader)  # average loss
    print(f"Validation Accuracy: {avg_acc:>6.3f}")
    print(f"Validation Loss:     {avg_loss:>6.3f}")

plt.plot(all_loss)
plt.show()

##############################################
# TESTING
##############################################
print('\nTESTING')
model.eval()
n_correct = 0
n_wrong = 0
for batch_idx, batch in enumerate(tqdm(test_loader, desc="   Testing")):

    X = batch['X'].to(DEVICE)
    y = batch['y'].to(DEVICE)

    optimizer.zero_grad()
    with torch.no_grad():
        output = model(X)

    big_idx = torch.argmax(output)
    if big_idx == y:
        n_correct += 1
    else:
        n_wrong += 1

time.sleep(0.5)
acc = n_correct / (n_correct + n_wrong)
print("Accuracy:", acc)

##############################################
# GRAPHING
##############################################

# # Convert it into an numpy array.
# data_array = np.array(predicted)
#
# # Create a figure for plotting the data as a 3D histogram.
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# # Create an X-Y mesh of the same dimension as the 2D data. You can
# # think of this as the floor of the plot.
# x_data, y_data = np.meshgrid(np.arange(data_array.shape[1]),
#                              np.arange(data_array.shape[0]))
#
# # Flatten out the arrays so that they may be passed to "ax.bar3d".
# # Basically, ax.bar3d expects three one-dimensional arrays:
# # x_data, y_data, z_data. The following call boils down to picking
# # one entry from each array and plotting a bar to from
# # (x_data[i], y_data[i], 0) to (x_data[i], y_data[i], z_data[i]).
# x_data = x_data.flatten()
# y_data = y_data.flatten()
# z_data = data_array.flatten()
# ax.bar3d(x_data,
#          y_data,
#          np.zeros(len(z_data)),
#          1, 1, z_data)
# #
# # Finally, display the plot.
# #
# plt.show()
