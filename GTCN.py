
import torch as t
import torch.nn as nn
import numpy as np
import random

unsq = t.unsqueeze
sq = t.squeeze


def setup_seed(seed=3407):
    t.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # For deterministic CUDA operations (optional):
    # t.cuda.manual_seed_all(seed)
    # t.backends.cudnn.deterministic = True


class M_transform(nn.Module):
    def __init__(self, T, len_M, N=None, device=None, xavier_normal_flag=False):
        super(M_transform, self).__init__()
        setup_seed()

        self.len_M = len_M
        self.T = T
        self.N = N

        # Automatically select GPU if available
        self.device = device if device is not None else (
            t.device('cuda' if t.cuda.is_available() else 'cpu')
        )

        # Register learnable temporal weights
        self.M = nn.ParameterDict()
        for length in range(1, len_M + 1):
            m = t.randn((1, length), dtype=t.float64, device=self.device)
            if xavier_normal_flag:
                nn.init.xavier_normal_(m)
            self.M[str(length)] = nn.Parameter(m)

        for length in range(len_M + 1, T + 1):
            m = t.randn((1, len_M), dtype=t.float64, device=self.device)
            if xavier_normal_flag:
                nn.init.xavier_normal_(m)
            self.M[str(length)] = nn.Parameter(m)

        self.to(self.device)

    def forward(self, X):
        if isinstance(X, t.Tensor):
            # Dense feature case
            X = X.to(self.device)
            T, N, F = X.shape
            out = t.zeros(T, N * F, dtype=t.float64, device=self.device)
            data = X.reshape(T, N * F)

            for row in range(T):
                if row < self.len_M:
                    weight = t.softmax(self.M[str(row + 1)], dim=1)
                    out[row] = weight.mm(data[0:row + 1])
                else:
                    weight = t.softmax(self.M[str(row)], dim=1)
                    out[row] = weight.mm(data[row - self.len_M + 1:row + 1])

            return out.reshape(X.shape)

        elif isinstance(X, list):
            # Sparse adjacency case
            sz = t.Size([self.N, self.N])
            res = []

            for tm in range(len(X)):
                m = t.softmax(self.M[str(tm + 1)], dim=1)
                temp = t.sparse_coo_tensor(size=sz, dtype=t.float64, device=self.device)

                if tm < self.len_M:
                    # Use all available history up to t
                    for i in range(tm + 1):
                        A = X[i].to(self.device)
                        val = m[:, i] * A._values()
                        temp = temp + t.sparse_coo_tensor(A._indices(), val, sz, device=self.device)
                else:
                    # Use last len_M steps
                    for i in range(self.len_M):
                        A = X[tm - self.len_M + i].to(self.device)
                        val = m[:, i] * A._values()
                        temp = temp + t.sparse_coo_tensor(A._indices(), val, sz, device=self.device)

                res.append(temp.coalesce())

            return res

        else:
            raise TypeError("M_transform forward() input type not supported!")


class Decoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim=1, device=None):
        super(Decoder, self).__init__()

        self.device = device if device is not None else t.device('cuda' if t.cuda.is_available() else 'cpu')

        # Linear transformation for concatenated node embeddings
        self.W_cat = nn.Parameter(t.randn(2 * in_dim, hidden_dim, dtype=t.float64, device=self.device))
        # Output transformation
        self.U = nn.Parameter(t.randn(hidden_dim, out_dim, dtype=t.float64, device=self.device))
        # Bias
        self.bias = nn.Parameter(t.zeros(hidden_dim, dtype=t.float64, device=self.device))

        # Activation
        self.relu = nn.ReLU()

        # Parameter initialization
        nn.init.xavier_uniform_(self.W_cat)
        nn.init.xavier_uniform_(self.U)
        nn.init.zeros_(self.bias)

        self.to(self.device)

    def forward(self, src_nodes, trg_nodes):
        src_nodes = src_nodes.to(self.device)
        trg_nodes = trg_nodes.to(self.device)

        rep_cat = t.matmul(t.cat((src_nodes, trg_nodes), dim=1), self.W_cat)
        rep_dot = src_nodes * trg_nodes
        rep_final = rep_dot + rep_cat + self.bias
        output = self.relu(t.matmul(rep_final, self.U))

        return output


class GTCN(nn.Module):
    def __init__(self, At, X, edges, fea_dim=10, layers=1, device=None):
        super(GTCN, self).__init__()

        self.T = X.shape[0]
        self.N = X.shape[1]
        self.layers = layers
        self.edges = edges
        self.At = At
        self.fea_dim = fea_dim

        self.device = device if device is not None else (
            t.device('cuda' if t.cuda.is_available() else 'cpu')
        )

        # Learnable node feature embeddings
        self.n_f = nn.Parameter(t.randn(self.N, self.fea_dim, dtype=t.float64))

        # Temporal transformation module
        self.M_trans = M_transform(T=self.T, len_M=20, N=self.N)

        # Multi-layer graph convolution weights
        self.W_layers = nn.ParameterList([
            nn.Parameter(t.randn(self.fea_dim, self.fea_dim, dtype=t.float64))
            for _ in range(self.layers)
        ])

        # Decoder for final link prediction or embedding scoring
        self.decoder = Decoder(in_dim=self.fea_dim, hidden_dim=self.fea_dim, out_dim=1)

        # Non-linearities and regularization
        self.relu = nn.ReLU(inplace=False)
        self.elu = nn.ELU()
        self.tanh = nn.Tanh()
        self.prelu = nn.PReLU()
        self.dp = nn.Dropout(0.1)

        # Parameter initialization
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize network parameters using Xavier normalization."""
        for W in self.W_layers:
            nn.init.xavier_normal_(W, gain=0.414)
        nn.init.xavier_normal_(self.n_f, gain=0.414)

    def __call__(self, At=None, edges=None):
        return self.forward(At, edges)

    def compute_AtXt(self, A, X):
        X_trans = self.M_trans(X)
        A_trans = self.M_trans(A)
        AtXt = t.zeros(self.T, self.N, X.shape[-1], dtype=t.float64).to(self.device)

        for k in range(self.T):
            AtXt[k] = t.sparse.mm(A_trans[k], X_trans[k])

        return AtXt

    def forward(self, At=None, edges=None):
        A = At if At is not None else self.At
        Y = self.n_f.repeat(self.T, 1, 1)

        # Multi-layer graph tensor convolution
        for l in range(self.layers):
            AtXt = self.compute_AtXt(A, Y)
            Y = t.matmul(AtXt, self.W_layers[l])
            Y = self.relu(Y) + Y # residual
            # Y = self.relu(Y)
            Y = self.dp(Y)

        H = Y  # Final node representations

        src_nodes = H[edges[0]]
        trg_nodes = H[edges[1]]

        output = self.decoder(src_nodes, trg_nodes)
        return output



