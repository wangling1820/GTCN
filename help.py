import math
import torch
import torch.nn as nn
import time
import scipy.io as sio
import torch as t
import numpy as np
import logging
from read_data_utils import get_X_features
import os

# Automatically detect CUDA availability and select the computing device.
device = t.device('cuda' if t.cuda.is_available() else 'cpu')


class EarlyStop:
    def __init__(self, patience):
        self.best_metric = float('inf')  # Initialize to a very large value
        self.count = 0
        self.patience = patience
        self.flag = False

    def is_stop(self, current_metric):

        if self.get_flag():
            return self.flag

        if current_metric < self.best_metric:  # Improvement
            self.best_metric = current_metric
            self.count = 0
        else:  # No improvement
            self.count += 1

        self.flag = self.count >= self.patience
        return self.flag


    def get_flag(self):
        """Return the current early-stop flag state."""
        return self.flag


def init_logger(data_name, s, fname):
    dt = time.strftime("%m%d%H%M%S", time.localtime())
    logname = f"lr_{s[0]}_embed_{s[1]}_layer_{s[2]}_GTCN_{dt}.log"
    log_dir = os.path.join('./logging', data_name)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, logname)
    logger = get_logging(log_path)
    logger.info(f"data name is: {fname}")
    logger.info(logname)
    return logger



def get_logging(file_name):
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(
        filename=file_name,
        level=logging.DEBUG,
        format=LOG_FORMAT,
        datefmt=DATE_FORMAT
    )
    return logging


def get_sparse_tensor(saved_content, idx, vals, sz):
    st = t.sparse_coo_tensor(
        t.tensor(np.array(saved_content[idx], dtype=int), dtype=t.long),   # Indices of non-zero entries
        t.squeeze(t.tensor(saved_content[vals], dtype=t.float64)),        # Corresponding values
        sz                                                                # Tensor size
    ).coalesce()  # Merge duplicate entries
    return st


def get_time_pos(T, N, d):
    def get_position_angle_vec(position, d_hid=2):
        # Compute positional angle for each embedding dimension
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    # Initialize encoding tensor
    X = t.zeros(T, N, d, dtype=t.float64).to(device)

    for i in range(T):
        # Compute angle vector for current timestep
        X[i] = t.tensor(get_position_angle_vec(i + 1, d_hid=d))
        # Apply sinusoidal transformation
        X[i, :, 0::2] = t.sin(X[i, :, 0::2])  # Even dimensions → sine
        X[i, :, 1::2] = t.cos(X[i, :, 1::2])  # Odd dimensions → cosine

    return X


def get_contend_info(Ct_x, T, C_sz, device=device):
    Ct = []
    for j in range(T):
        # Identify indices corresponding to the current timestep j
        idx = Ct_x._indices()[0] == j
        # Extract sub-tensor for current timestep
        Ct.append(
            t.sparse_coo_tensor(
                Ct_x._indices()[1:3, idx],
                Ct_x._values()[idx],
                C_sz
            ).to(device).coalesce()
        )
    return Ct


def DataLoder(filename, logger, device=device):
    # ===== Load serialized graph tensors from MATLAB file =====
    saved_content = sio.loadmat(filename)

    # ===== Determine the temporal and spatial ranges =====
    # Tensor indices are typically stored as [time, src_node, trg_node].
    # The number of timesteps T and node count N are computed as (max - min + 1).
    MT = np.min(saved_content["tensor_idx"][0])
    M1 = np.min(saved_content["tensor_idx"][1])
    M2 = np.min(saved_content["tensor_idx"][2])
    Min = min(M1, M2)

    XT = np.max(saved_content["tensor_idx"][0])
    N1 = np.max(saved_content["tensor_idx"][1])
    N2 = np.max(saved_content["tensor_idx"][2])
    Max = max(N1, N2)

    T = XT - MT + 1        # Number of time steps
    N = Max - Min + 1       # Number of nodes
    logger.info('len of time is %d, num of nodes is %d' % (T, N))

    # Define tensor sizes for temporal adjacency and condensed matrices
    sz = t.Size([T, N, N])
    C_sz = t.Size([N, N])

    # ===== Construct sparse tensors for adjacency (A) and candidate links (Ct) =====
    A_train = get_sparse_tensor(saved_content, 'A_train_idx', 'A_train_vals', sz)


    Ct_train = get_sparse_tensor(saved_content, 'train_idx', 'train_vals', sz)


    # ===== Load sparse label tensors (ground truth links) =====
    train_label = get_sparse_tensor(saved_content, 'train_label_idx', 'train_label_vals', sz)
    val_label = get_sparse_tensor(saved_content, 'val_label_idx', 'val_label_vals', sz)
    test_label = get_sparse_tensor(saved_content, 'test_label_idx', 'test_label_vals', sz)

    # Move tensors to target device (GPU or CPU)
    A_train = A_train.to(device=device)

    train_label = train_label.to(device=device)
    val_label = val_label.to(device=device)
    test_label = test_label.to(device=device)

    # ===== Compute condensed temporal structural representations =====
    # These condensed matrices summarize temporal edge connectivity
    # and can be used for efficient higher-order message aggregation.
    Ct_train_2 = get_contend_info(Ct_train, T, C_sz)


    # ===== Compute node feature matrices =====
    # Extract node-level features (e.g., degree, temporal statistics) from adjacency tensors.
    X_train = get_X_features(A_train).to(device)

    # ===== Extract edge indices and corresponding labels =====
    # Each label tensor contains triplets (time, src, trg, value),
    # which are converted into separate edge index lists and binary labels.
    edges_idx = train_label._indices().to(device)
    edges_train = [edges_idx[[0, 1]].tolist(), edges_idx[[0, 2]].tolist()]
    labels_train = train_label._values().to(device)

    edges_idx = val_label._indices().to(device)
    edges_val = [edges_idx[[0, 1]].tolist(), edges_idx[[0, 2]].tolist()]
    labels_val = val_label._values().to(device)

    edges_idx = test_label._indices().to(device)
    edges_test = [edges_idx[[0, 1]].tolist(), edges_idx[[0, 2]].tolist()]
    labels_test = test_label._values().to(device)

    # Reshape label values into column vectors (for BCE/CE loss functions)
    target_train = torch.reshape(labels_train, (-1, 1)).to(device)
    target_val = torch.reshape(labels_val, (-1, 1)).to(device)
    target_test = torch.reshape(labels_test, (-1, 1)).to(device)

    logger.info('num of train is %d, num of val is %d, num of test is %d' %
                (target_train.shape[0], target_val.shape[0], target_test.shape[0]))

    # ===== Package processed tensors into dictionaries =====
    # A, X, edges, target are returned for model training and evaluation.

    A = {'train': Ct_train_2}
    X = {'train': X_train}
    edges = {'train': edges_train, 'val': edges_val, 'test': edges_test}
    target = {'train': target_train, 'val': target_val, 'test': target_test}

    return A, X, edges, target



def train(gcn, A, edges, target, logger, opt_setting, device=device):

    train_time = 0.0  # Accumulate total training time

    # Move model to target device
    gcn.to(device)
    total = sum([param.nelement() for param in gcn.parameters()])
    print("Number of parameters: %f" % (total))
    logger.info("Number of parameters: %f" % (total))

    # ===== Optimizer selection =====
    if opt_setting['optim'] == 'Adam':
        optimizer = t.optim.Adam(gcn.parameters(), lr=opt_setting['lr'], weight_decay=0.01)
    elif opt_setting['optim'] == 'SGD':
        optimizer = t.optim.SGD(gcn.parameters(), lr=opt_setting['lr'], momentum=0.09, weight_decay=0.0001)
    else:
        exit('Optimizer not specified!')

    # ===== Loss function selection =====
    # Options: Mean Absolute Error (L1), Mean Squared Error, or Huber Loss
    if opt_setting['loss'] == 'MAE':
        criterion = nn.L1Loss()
    elif opt_setting['loss'] == 'MSE':
        criterion = nn.MSELoss()
    elif opt_setting['loss'] == 'huber':
        criterion = nn.HuberLoss(reduction='sum', delta=1)
    else:
        exit('Loss function not specified!')

    # Auxiliary loss metrics for evaluation
    mae = nn.L1Loss()
    mse = nn.MSELoss()

    # Initialize best metrics and early stopping handlers
    val_rmse_min = val_mae_min = 1e10
    rmse_break = EarlyStop(10)
    mae_break = EarlyStop(10)

    ep = 1
    while ep <= 500:
        # ===== Training step =====
        s_time = time.time()
        optimizer.zero_grad()

        # Forward pass: GTNN inference
        output_train = gcn(A['train'], edges['train'])
        loss_train = criterion(output_train, target['train'])

        # Backward propagation
        loss_train.backward(retain_graph=True)
        optimizer.step()

        torch.cuda.empty_cache()
        e_time = time.time()
        train_time += e_time - s_time

        # ===== Evaluation phase =====
        with t.no_grad():
            # Compute training metrics
            rmse_train = math.sqrt(mse(output_train, target['train']))
            mae_train = mae(output_train, target['train'])

            # Validation inference
            output_val = gcn(A['train'], edges['val'])
            rmse_val = math.sqrt(mse(output_val, target['val']))
            mae_val = mae(output_val, target['val']).item()

            # Test inference
            output_test = gcn(A['train'], edges['test'])
            rmse_test = math.sqrt(mse(output_test, target['test']))
            mae_test = mae(output_test, target['test']).item()

            # ===== Logging results =====
            logger.info("Ep %d. Train MAE %.4f. Train RMSE %.4f." % (ep, mae_train, rmse_train))
            logger.info("Ep %d. Val MAE %.4f. Val RMSE %.4f." % (ep, mae_val, rmse_val))
            logger.info("Ep %d. Test MAE %.4f. Test RMSE %.4f.\n" % (ep, mae_test, rmse_test))

            print("Ep %d. Train MAE %.4f. Train RMSE %.4f." % (ep, mae_train, rmse_train))
            print("Ep %d. Val MAE %.4f. Val RMSE %.4f." % (ep, mae_val, rmse_val))
            print("Ep %d. Test MAE %.4f. Test RMSE %.4f.\n" % (ep, mae_test, rmse_test))

            # ===== Early stopping logic =====
            if round(val_rmse_min, 4) > round(rmse_val, 4) and not rmse_break.get_flag():
                epoch_rmse = ep
                test_rmse_min = rmse_test
                val_rmse_min = rmse_val

            if round(val_mae_min, 4) > round(mae_val, 4) and not mae_break.get_flag():
                epoch_mae = ep
                val_mae_min = mae_val
                test_mae_min = mae_test

        # ===== Early stopping trigger =====
        if rmse_break.is_stop(round(rmse_val, 3)) and mae_break.is_stop(round(mae_val, 3)):
            break

        if rmse_break.get_flag() and mae_break.get_flag():
            break

        # ===== Invalid training detection =====
        if ep > 20 and rmse_val > 40 and mae_val > 40:
            logger.info('****** This setting cannot make model converge properly! *********')
            break

        ep += 1

    # ===== Final performance summary =====
    logger.info("Best MAE epoch is %d, Val MAE %.4f." % (epoch_mae, test_mae_min))
    logger.info("Best RMSE epoch is %d, Test RMSE %.4f.\n" % (epoch_rmse, test_rmse_min))
    logger.info("Train Time is: %0.6f s" % (train_time))

    print("Best MAE epoch is %d, Val MAE %.4f." % (epoch_mae, test_mae_min))
    print("Best RMSE epoch is %d, Test RMSE %.4f.\n" % (epoch_rmse, test_rmse_min))
    print("Train MAE %.4f. Train RMSE %.4f." % (mae_train, rmse_train))
    print("Val MAE %.4f. Val RMSE %.4f." % (mae_val, rmse_val))
    print("Test MAE %.4f. Test RMSE %.4f.\n" % (mae_test, rmse_test))
    print("Train Time is: %0.6f s" % (train_time))
