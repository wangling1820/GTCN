import numpy as np
import torch
from sklearn.utils import shuffle
import scipy.io as sio
import torch as t


def setup_seed(seed=3407):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def get_contend_info(Ct_x, T):
    Ct = []
    for j in range(T):
        idx = Ct_x._indices()[0] == j
        Ct.append(
            t.sparse_coo_tensor(Ct_x._indices()[1:3, idx], Ct_x._values()[idx], dtype=t.float64))
    return Ct


def get_random_idx(num_all, num_ones, random_state=3407):
    no_true = np.ones(num_ones) == 1
    no_false = np.zeros(num_all - num_ones) == 1
    temp = list(no_true) + list(no_false)
    idx = shuffle(torch.tensor(temp), random_state=random_state)
    return idx


def get_dataset(A, idx):
    sz = A.size()
    not_idx = idx == False

    # Selected subset
    index = torch.LongTensor(A._indices()[0:3, idx].size())
    index[0:3] = A._indices()[0:3, idx]
    values = A._values()[idx]
    sub = torch.sparse_coo_tensor(index, values, sz)

    # Remaining subset
    remain_index = torch.LongTensor(A._indices()[0:3, not_idx].size())
    remain_index[0:3] = A._indices()[0:3, not_idx]
    remain_values = A._values()[not_idx]
    remain = torch.sparse_coo_tensor(remain_index, remain_values, sz)

    return remain.coalesce(), sub.coalesce()


def print_tensor(A, name):
    """Print a summary of a sparse tensor."""
    print('------------------------')
    print(name)
    print(A)
    print("Sum of values:", torch.sum(A._values()))
    print('------------------------')


def get_X_features(A, use_pos=True):

    # Initialize the node feature tensor
    # Shape: [T, N, 2], dtype=float64
    X = t.zeros(A.shape[0], A.shape[1], 2, dtype=t.float64)

    if use_pos:
        # Compute sparse summation over adjacency along each axis
        # A_sin corresponds to aggregation along outgoing edges (dim=1)
        # A_cos corresponds to aggregation along incoming edges (dim=2)
        # Resulting shape for both: [T, N]
        A_sin = t.sparse.sum(A, 1)
        A_cos = t.sparse.sum(A, 2)

        # ======== Construct sinusoidal encodings ========
        # Extract time indices (the first dimension of sparse indices)
        A_sin_T = A_sin._indices()[0]  # Temporal index positions

        # Compute sine positional encodings for temporal indices
        A_sin_pos = torch.sin(A_sin_T)

        # Combine sine position values with existing adjacency weights
        # This adds a temporal prior to the degree-like aggregation
        A_sin_val = A_sin_pos + A_sin._values()

        # Rebuild a sparse tensor incorporating sinusoidal values
        A_sin_pe = torch.sparse_coo_tensor(A_sin._indices(), A_sin_val, A_sin.size())

        # Do the same for cosine positional encodings
        A_cos_T = A_cos._indices()[0]
        A_cos_val = torch.cos(A_cos_T) + A_cos._values()
        A_cos_pe = torch.sparse_coo_tensor(A_cos._indices(), A_cos_val, A_cos.size())

        # Convert sparse tensors to dense and store them into feature matrix
        # X[..., 0] : sine-based aggregated features
        # X[..., 1] : cosine-based aggregated features
        X[:, :, 0] = A_sin_pe.coalesce().to_dense()
        X[:, :, 1] = A_cos_pe.coalesce().to_dense()

        # Optionally, a third channel (e.g., normalized degree or time embedding)
        # could be added here in future implementations.
        # X[:, :, 2] = ...

    else:
        # Without positional priors, use degree-based aggregation
        # X[..., 0] = out-degree, X[..., 1] = in-degree
        X[:, :, 0] = t.sparse.sum(A, 1).to_dense()
        X[:, :, 1] = t.sparse.sum(A, 2).to_dense()

    return X


def func_make_symmetric(sparse_tensor, N, TT):
    tensor_idx = torch.LongTensor([])
    tensor_val = torch.FloatTensor([]).unsqueeze(1)

    A_idx = sparse_tensor._indices()
    A_val = sparse_tensor._values()

    for j in range(TT):
        idx = A_idx[0] == j
        mat = torch.sparse_coo_tensor(A_idx[1:3, idx], A_val[idx], torch.Size([N, N]))

        # Symmetrize by averaging A and A^T
        mat_t = mat.transpose(1, 0)
        sym_mat = (mat + mat_t) / 2

        # Rebuild indices and values with time dimension
        vertices = sym_mat._indices().clone().detach()
        time = torch.ones(sym_mat._nnz(), dtype=torch.long) * j
        time = time.unsqueeze(0)
        full = torch.cat((time, vertices), 0)

        tensor_idx = torch.cat((tensor_idx, full), 1)
        tensor_val = torch.cat((tensor_val, sym_mat._values().unsqueeze(1)), 0)

    tensor_val.squeeze_(1)
    A = torch.sparse_coo_tensor(tensor_idx, tensor_val, torch.Size([TT, N, N])).coalesce()
    return A


def func_edge_life(A, N, TT, edge_life_window):
    A_new = A.clone()
    A_new._values()[:] = 0

    for t in range(TT):
        idx = (A._indices()[0] >= max(0, t - edge_life_window + 1)) & (A._indices()[0] <= t)
        block = torch.sparse_coo_tensor(A._indices()[0:3, idx], A._values()[idx], torch.Size([TT, N, N]))
        block._indices()[0] = t
        A_new = A_new + block

    return A_new.coalesce()


def func_laplacian_transformation(B, N, TT):
    # Create identity tensor I
    vertices = torch.LongTensor([range(N), range(N)])
    tensor_idx = torch.LongTensor([])
    tensor_val = torch.DoubleTensor([]).unsqueeze(1)

    for j in range(TT):
        time = torch.ones(N, dtype=torch.long) * j
        time = time.unsqueeze(0)
        full = torch.cat((time, vertices), 0)
        tensor_idx = torch.cat((tensor_idx, full), 1)
        val = torch.ones(N, dtype=torch.float64)
        tensor_val = torch.cat((tensor_val, val.unsqueeze(1)), 0)

    tensor_val.squeeze_(1)
    I = torch.sparse_coo_tensor(tensor_idx, tensor_val, torch.Size([TT, N, N]))

    # Add self-loops
    C = B + I

    tensor_idx = torch.LongTensor([])
    tensor_val = torch.DoubleTensor([]).unsqueeze(1)

    for j in range(TT):
        idx = C._indices()[0] == j
        mat = torch.sparse_coo_tensor(C._indices()[1:3, idx], C._values()[idx], torch.Size([N, N]))
        vec = torch.ones([N, 1], dtype=torch.float64)

        # Compute degree normalization
        degree = 1 / torch.sqrt(torch.sparse.mm(mat, vec))
        index = torch.LongTensor(C._indices()[0:3, idx].size())
        values = torch.DoubleTensor(C._values()[idx].size())
        index[0] = j
        index[1:3] = mat._indices()
        values = mat._values()

        count = 0
        for i, j in index[1:3].transpose(1, 0):
            values[count] = values[count] * degree[i] * degree[j]
            count += 1

        tensor_idx = torch.cat((tensor_idx, index), 1)
        tensor_val = torch.cat((tensor_val, values.unsqueeze(1)), 0)

    tensor_val.squeeze_(1)
    C = torch.sparse_coo_tensor(tensor_idx, tensor_val, torch.Size([TT, N, N]))
    return C.coalesce()


def read_data(src_file, train_rate, val_rate, test_rate,
              edge_life, edge_life_window,
              make_symmetric,
              val_len=15, test_len=20, random_sample=True):
    setup_seed()
    print(f"Loading data from {src_file} ...")

    # Load raw data: each row = (src, dst, value, time)
    data = np.loadtxt(src_file, delimiter=',', skiprows=1)
    save_file_location = src_file[:-4]

    # Build output file name
    file_name = ""
    if edge_life:
        file_name += f"_edge_{edge_life_window}"
    if make_symmetric:
        file_name += "_sym"
    file_name += "_GTCN"
    save_file_name = f"{file_name}_{str(train_rate)[-1]}_{str(val_rate)[-1]}_{str(test_rate)[-1]}.mat"
    res_file = save_file_location + save_file_name

    data_len = data.shape[0]
    data = torch.tensor(data)

    # Extract unique time steps
    dates = np.unique(data[:, 3])
    TT = len(dates)
    N = int(max(max(data[:, 0]), max(data[:, 1]))) + 1

    # Calculate split sizes
    if random_sample:
        no_val_samples = int(data_len * val_rate)
        no_test_samples = int(data_len * test_rate)
    else:
        no_train_samples = TT - val_len - test_len
        no_val_samples = no_train_samples + val_len
        no_test_samples = no_val_samples + test_len

    # Build sparse tensor for adjacency matrix
    tensor_idx = torch.zeros([data.size()[0], 3], dtype=torch.long)
    tensor_val = torch.ones([data.size()[0]], dtype=torch.float64)
    tensor_labels = torch.zeros([data.size()[0]], dtype=torch.float64)

    # Fill temporal indices
    for t in range(TT):
        idx = data[:, 3] == dates[t]
        tensor_idx[idx, 1:3] = (data[idx, 0:2]).type('torch.LongTensor')
        tensor_idx[idx, 0] = t
        tensor_labels[idx] = data[idx, 2]

    A = torch.sparse_coo_tensor(tensor_idx.transpose(1, 0), tensor_val, torch.Size([TT, N, N])).coalesce()
    A_labels = torch.sparse_coo_tensor(tensor_idx.transpose(1, 0), tensor_labels, torch.Size([TT, N, N])).coalesce()

    # Random sampling
    val_idx_rand = get_random_idx(A_labels._nnz(), no_val_samples)
    test_idx_rand = get_random_idx(A_labels._nnz() - no_val_samples, no_test_samples)

    # Split label and adjacency tensors
    label_remain_data, label_val = get_dataset(A_labels, val_idx_rand)
    label_train, label_test = get_dataset(label_remain_data, test_idx_rand)
    A_test, A_val_temp = get_dataset(A, val_idx_rand)
    A_train, _ = get_dataset(A_test, test_idx_rand)
    A_val = torch.add(A_train, A_val_temp).coalesce()

    # Process symmetric and edge-life transformations
    print('Applying symmetry...')
    A_train_sym = func_make_symmetric(A_train, N, TT) if make_symmetric else A_train
    A_test_sym = func_make_symmetric(A_test, N, TT) if make_symmetric else A_test
    A_val_sym = func_make_symmetric(A_val, N, TT) if make_symmetric else A_val

    print('Applying edge life window...')
    A_train_sym_life = func_edge_life(A_train_sym, N, TT, edge_life_window) if edge_life else A_train_sym
    A_test_sym_life = func_edge_life(A_test_sym, N, TT, edge_life_window) if edge_life else A_test_sym
    A_val_sym_life = func_edge_life(A_val_sym, N, TT, edge_life_window) if edge_life else A_val_sym

    print('Applying Laplacian normalization...')
    Ct_train = func_laplacian_transformation(A_train_sym_life, N, TT)
    Ct_val = func_laplacian_transformation(A_val_sym_life, N, TT)
    Ct_test = func_laplacian_transformation(A_test_sym_life, N, TT)

    # Save results
    print(f'Saving processed data to {res_file} ...')
    sio.savemat(res_file, {
        'tensor_idx': np.array(tensor_idx.transpose(1, 0)),
        'tensor_labels': np.array(tensor_labels),

        'A_train_idx': np.array(A_train._indices()),
        'A_train_vals': np.array(A_train._values()),
        'A_val_idx': np.array(A_val._indices()),
        'A_val_vals': np.array(A_val._values()),
        'A_test_idx': np.array(A_test._indices()),
        'A_test_vals': np.array(A_test._values()),

        'train_label_idx': np.array(label_train._indices()),
        'train_label_vals': np.array(label_train._values()),
        'val_label_idx': np.array(label_val._indices()),
        'val_label_vals': np.array(label_val._values()),
        'test_label_idx': np.array(label_test._indices()),
        'test_label_vals': np.array(label_test._values()),

        'train_idx': np.array(Ct_train._indices()),
        'train_vals': np.array(Ct_train._values()),
        'val_idx': np.array(Ct_val._indices()),
        'val_vals': np.array(Ct_val._values()),
        'test_idx': np.array(Ct_test._indices()),
        'test_vals': np.array(Ct_test._values()),
    })

    print(f'Data successfully saved at: {res_file}\n')
