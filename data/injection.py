import torch

from utils import idct_2d


def create_linf_carrier():
    V = torch.zeros([10, 3, 32, 32])
    V[0, :, 0, 1] = 1.0
    V[1, :, 1, 0] = 1.0
    V[2, :, 1, 1] = 1.0
    V[3, :, 2, 1] = 1.0
    V[4, :, 1, 2] = 1.0
    V[5, :, 0, 1] = -1.0
    V[6, :, 1, 0] = -1.0
    V[7, :, 1, 1] = -1.0
    V[8, :, 2, 1] = -1.0
    V[9, :, 1, 2] = -1.0
    V = torch.sign(idct_2d(V.view([-1, 32, 32])))
    V = V.reshape([-1, 3, 32, 32])
    return V


def generate_zigzag_indices(shape):
    indices = [[] for i in range(shape[0] + shape[1] - 1)]
    for i in range(shape[0]):
        for j in range(shape[1]):
            sum = i + j
            if sum % 2 == 0:

                # add at beginning
                indices[sum].insert(0, (i, j))
            else:

                # add at end of the list
                indices[sum].append((i, j))
    return indices


def create_general_linf_carrier(num_classes, shape):
    V = torch.zeros(
        [
            num_classes,
        ]
        + shape
    )

    indices = generate_zigzag_indices(shape[1:])

    count = 0
    for idx_list in indices:
        for j in idx_list:
            if count > 0:
                matrix = torch.zeros(*shape)
                matrix[:, j] = 1.0
                V_j = torch.sign(idct_2d(matrix))
                V[count - 1] = V_j

            count += 1

            if count == num_classes + 1:
                break
        if count == num_classes + 1:
            break

    return V


def inject_feature(
    trainset,
    epsilon,
    V,
    batch_size=128,
    num_workers=2,
    shuffle=True,
):
    num_classes = V.shape[0]

    x = torch.from_numpy(trainset.data.transpose([0, 3, 1, 2])).type(torch.float) / 255.0
    y = torch.tensor(trainset.targets, dtype=torch.long)

    x_poison = x.clone()

    for t in range(num_classes):
        x_poison[y == t] += epsilon * V[t]

    poisonset = torch.utils.data.TensorDataset(x_poison, y)
    poisonloader = torch.utils.data.DataLoader(
        poisonset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True
    )

    return poisonloader


def inject_perp_feature(
    trainset,
    epsilon,
    V,
    batch_size=128,
    num_workers=2,
    shuffle=True,
):
    num_classes = V.shape[0]

    x = torch.from_numpy(trainset.data.transpose([0, 3, 1, 2])).type(torch.float) / 255.0
    y = torch.tensor(trainset.targets, dtype=torch.long)

    Q, _ = torch.linalg.qr(V.reshape([num_classes, -1]).T)

    P = Q.matmul(Q.T)
    x_proj = P.matmul(x.view([-1, 3072]).T)

    x_poison = x - x_proj.T.view([-1, 3, 32, 32])

    for t in range(num_classes):
        x_poison[y == t] += epsilon * V[t]

    poisonset = torch.utils.data.TensorDataset(x_poison, y)
    poisonloader = torch.utils.data.DataLoader(
        poisonset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True
    )

    return poisonloader


def inject_reverse_feature(
    trainset,
    epsilon,
    V,
    batch_size=128,
    num_workers=2,
    shuffle=True,
):
    num_classes = V.shape[0]

    x = torch.from_numpy(trainset.data.transpose([0, 3, 1, 2])).type(torch.float) / 255.0
    y = torch.tensor(trainset.targets, dtype=torch.long)

    x_poison = x.clone()

    for t in range(num_classes):
        x_poison[y == t] -= epsilon * V[t]

    poisonset = torch.utils.data.TensorDataset(x_poison, y)
    poisonloader = torch.utils.data.DataLoader(
        poisonset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True
    )

    return poisonloader


def inject_reverse_perp_feature(
    trainset,
    epsilon,
    V,
    batch_size=128,
    num_workers=2,
    shuffle=True,
):
    num_classes = V.shape[0]

    x = torch.from_numpy(trainset.data.transpose([0, 3, 1, 2])).type(torch.float) / 255.0
    y = torch.tensor(trainset.targets, dtype=torch.long)

    Q, _ = torch.linalg.qr(V.reshape([num_classes, -1]).T)

    P = Q.matmul(Q.T)
    x_proj = P.matmul(x.view([-1, 3072]).T)

    x_poison = x - x_proj.T.view([-1, 3, 32, 32])

    for t in range(num_classes):
        x_poison[y == t] -= epsilon * V[t]

    poisonset = torch.utils.data.TensorDataset(x_poison, y)
    poisonloader = torch.utils.data.DataLoader(
        poisonset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True
    )

    return poisonloader


def carrier_loader(epsilon, V):
    num_classes = V.shape[0]
    x = torch.zeros([num_classes, 3, 32, 32])
    y = torch.arange(num_classes)

    for t in range(num_classes):
        x[y == t] += epsilon * V[t]

    poisonset = torch.utils.data.TensorDataset(x, y)
    poisonloader = torch.utils.data.DataLoader(
        poisonset, batch_size=num_classes, shuffle=False, num_workers=2, pin_memory=True
    )

    return poisonloader
