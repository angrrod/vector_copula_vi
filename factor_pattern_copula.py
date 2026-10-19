import torch
from pyro.distributions.torch_distribution import TorchDistribution
from torch.distributions import constraints
import torch.nn as nn

import torch.nn.functional as F
from abc import ABC, abstractmethod

def Blockdiag(B,dimList):
    #B has shape[M,D,D]
    Bdiag = B.clone()
    
    prevDim = 0
    #TODO: extend to multiple signals
    for dim in dimList:
        ind = dim + prevDim
        Bdiag[..., :ind,ind:] = 0
        Bdiag[..., ind:,:ind] = 0   
        prevDim += dim  
        
    return Bdiag

def positive_zeta(z, eps=1e-6):
    return F.softplus(z) + eps

#tests
def _assert_finite_tensor(name: str, x: torch.Tensor):
    if not torch.is_tensor(x):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(x)}")

    if not torch.isfinite(x).all():
        finite = torch.isfinite(x)

        if finite.any():
            finite_vals = x[finite]
            min_val = finite_vals.min().item()
            max_val = finite_vals.max().item()
        else:
            min_val = "no finite values"
            max_val = "no finite values"

        raise ValueError(
            f"{name} contains NaN or Inf. "
            f"shape={tuple(x.shape)}, "
            f"dtype={x.dtype}, "
            f"device={x.device}, "
            f"finite_min={min_val}, "
            f"finite_max={max_val}"
        )
    
def _assert_finite_module_parameters(name: str, module: torch.nn.Module):
    for param_name, param in module.named_parameters():
        _assert_finite_tensor(f"{name}.{param_name}", param)

    for buffer_name, buffer in module.named_buffers():
        _assert_finite_tensor(f"{name}.{buffer_name}", buffer)
    

class FactorPatternCorrelationModel(TorchDistribution,ABC):
    arg_constraints = {}
    support = constraints.real_vector
    has_rsample = True
    def __init__(self, flows:nn.ModuleList, B:torch.Tensor, z:torch.Tensor,blocksizes:list, debug_checks:bool = False, is_independent:bool = False):
        """
        args:
            flows           nn.ModuleList of Zuko flows modeling the margina
            B               M x D x P matrix, where P < D, D is event_shape and M is the batch shape denoting different distributions, used in amortization
            z               M vector parameter
            debug_checks    bool indicating whether or not to check the parameter inputs for nan's
            is_independent  boolean indicating if the copula is independent or not, in the latter case, B and zeta are fixed and non trainable
        """
        #test finitness of the input parameters
        if debug_checks:
            _assert_finite_tensor("B", B)
            _assert_finite_tensor("z", z)

            for i, flow in enumerate(flows):
                if not isinstance(flow, torch.nn.Module):
                    raise TypeError(
                        f"flows[{i}] must be a torch.nn.Module, got {type(flow)}"
                    )

                _assert_finite_module_parameters(f"flows[{i}]", flow)
                
        self.is_independent = is_independent
        self.flows          = flows
        
        self.B = B
        self.z = z
        self.blocksizes = blocksizes

        self.M = B.shape[0]
        self.D = B.shape[1]
        self.P = B.shape[2]

        self.device = B.device
        self.dtype = B.dtype

        if is_independent:
            self.B = torch.zeros_like(B)
            self.z = torch.zeros_like(z)

        if B.device != z.device:
            raise ValueError(
                f"Device mismatch: B={B.device}, z={z.device}"
            )

        if B.dtype != z.dtype:
            raise ValueError(
                f"Dtype mismatch: B={B.dtype}, z={z.dtype}"
            )

        if self.D <= self.P:
            raise ValueError(
                f"Require D > P, got D={self.D}, P={self.P}"
            )

        if z.shape[0] != B.shape[0]:
            raise ValueError(
                f"Batch mismatch: B has M={B.shape[0]}, "
                f"z has M={z.shape[0]}"
            )

        if sum(blocksizes) != self.D:
            raise ValueError(
                f"Block sizes sum to {sum(blocksizes)}, expected {self.D}"
            )
        batch_shape = torch.Size([self.M])
        event_shape = torch.Size([self.D])

        super().__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=None,
        )
    
    def WoodburyComponents(self):
        """
        Compute Omega_tilde^{-1} using the Woodbury identity.

        Omega_tilde = zeta I_D + B B^T

        Instead of inverting a D x D matrix, this requires
        a Cholesky decomposition of the P x P matrix

            K = zeta I_P + B^T B.

        Returns
        -------
        OmegaTildeInverse : [M, D, D]
        zeta              : [M]
        L_K               : [M, P, P]
        """

        zeta = positive_zeta(self.z)

        Bt = self.B.transpose(-1, -2)

        zeta_b = zeta.view(
            self.M, 1, 1
        )

        I_d = torch.eye(
            self.D,
            device=self.device,
            dtype=self.dtype,
        )

        I_p = torch.eye(
            self.P,
            device=self.device,
            dtype=self.dtype,
        )

        I_d_batch = I_d.expand(
            self.M,
            self.D,
            self.D,
        )

        I_p_batch = I_p.expand(
            self.M,
            self.P,
            self.P,
        )

        K = (
            zeta_b * I_p_batch
            + Bt @ self.B
        )

        L_K = torch.linalg.cholesky(K)

        # K^{-1} B^T
        Kinv_Bt = torch.cholesky_solve(
            Bt,
            L_K,
        )

        OmegaTildeInverse = (
            I_d_batch
            - self.B @ Kinv_Bt
        ) / zeta_b

        return (
            OmegaTildeInverse,
            zeta,
            L_K,
        )
        
    def OmegaComponents(self):
        """
        Construct

            Omega_tilde = zeta I + B B^T

        and the Cholesky factor of its block diagonal:

            A_inv = chol(bdiag(Omega_tilde)).

        Since A = A_inv^{-1},

            Omega = A Omega_tilde A^T

        has identity diagonal blocks.

        Returns
        -------
        OmegaTilde : [M, D, D]
        A_inv      : [M, D, D]
        """

        I = torch.eye(
            self.D,
            device=self.device,
            dtype=self.dtype,
        )

        I_batch = I.expand(
            self.M,
            self.D,
            self.D,
        )

        zeta = positive_zeta(self.z)

        OmegaTilde = (
            zeta[:, None, None] * I_batch
            + self.B @ self.B.transpose(-1, -2)
        )

        if not torch.isfinite(OmegaTilde).all():
            raise RuntimeError(
                "OmegaTilde contains NaN/Inf"
            )

        Bd = Blockdiag(
            OmegaTilde,
            self.blocksizes,
        )

        A_inv = torch.linalg.cholesky(Bd)

        return OmegaTilde, A_inv
    
    def getLogCopulaTerm(
        self,
        Q: torch.Tensor,
        Mmat: torch.Tensor,
    ):
        """
        Calculate

            Q^T Mmat Q

        for each sample S and model batch M.

        Args
        ----
        Q:
            [S, M, D]
        Mmat:
            [M, D, D]

        Returns
        -------
        quad:
            [S, M]
        """

        if Q.dim() != 3:
            raise ValueError(
                f"Expected Q shape [S, M, D], "
                f"got {tuple(Q.shape)}"
            )

        S, M, D = Q.shape

        if M != self.M:
            raise ValueError(
                f"Q has M={M}, expected {self.M}"
            )

        if D != self.D:
            raise ValueError(
                f"Q has D={D}, expected {self.D}"
            )

        expected_shape = (
            self.M,
            self.D,
            self.D,
        )

        if Mmat.shape != expected_shape:
            raise ValueError(
                f"Mmat must have shape {expected_shape}, "
                f"got {tuple(Mmat.shape)}"
            )

        Q_flat = Q.reshape(
            S * self.M,
            self.D,
        )

        Mmat_flat = (
            Mmat
            .unsqueeze(0)
            .expand(
                S,
                self.M,
                self.D,
                self.D,
            )
            .reshape(
                S * self.M,
                self.D,
                self.D,
            )
        )

        MQ = torch.bmm(
            Mmat_flat,
            Q_flat.unsqueeze(-1),
        ).squeeze(-1)

        quad = (
            Q_flat * MQ
        ).sum(dim=-1)

        return quad.reshape(
            S,
            self.M,
        )
    
    def _logProbCopula(self, Q):
        """
        Compute copula contribution for logprob
        Q   Vector quantiles    Q.shape == [S, self.M, self.D]
        """
        I        = torch.eye(self.D, device=self.device, dtype = self.dtype)
        I_batch  = I.expand(self.M,self.D,self.D)
        OmegaInv,A_inv, zeta, K = self.InvertOmega()  #[self.M, self.D, self.D]
        
        logDetTerm    = -0.5 * self.LogDetOmegaLowRank(A_inv,zeta,K)
        logCopulaTerm = -0.5 * self.getLogCopulaTerm(Q,Mmat = OmegaInv- I_batch)
        
        logDensity      = logDetTerm + logCopulaTerm # up to a constant term
        return logDensity, logDetTerm, logCopulaTerm
    
    def LogDetOmegaLowRank(
        self,
        A_inv: torch.Tensor,
        zeta: torch.Tensor,
        L_K: torch.Tensor,
        context = None
    ) -> torch.Tensor:
        """
        Computes logdet(Omega) without a D x D Cholesky.

        logdet(Omega) = logdet(OmegaTilde) - 2 logdet(A_inv), from definition Omega = A @ OmegaTilde @ A^T and A_inv = A^{-1}

        and:
            logdet(OmegaTilde)
            = (D - P) log(zeta) + logdet(zeta I_P + B^T B), from definition OmegaTilde = zeta I_D + B B^T
        """

        diag_L_K = torch.diagonal(L_K, dim1=-2, dim2=-1)
        logdet_K = 2.0 * torch.log(diag_L_K).sum(dim=-1)  # [M]
        logdet_OmegaTilde = (self.D - self.P) * torch.log(zeta) + logdet_K  # [M]
        diag_A_inv = torch.diagonal(A_inv, dim1=-2, dim2=-1)
        logdet_A_inv = torch.log(diag_A_inv).sum(dim=-1)  # [M]
        logdet_Omega = logdet_OmegaTilde - 2.0 * logdet_A_inv  

        return logdet_Omega
    
    def InvertOmega(self) -> torch.Tensor:
        """Inverts Omega using woodbury forumla and sequential rank-1 Cholesky
        factor updates

        Returns:
            torch.Tensor: inverted Omega
        """
        OmegaTildeInverse, zeta, L_K = self.WoodburyComponents(self.z,self.B) 
        _, A_inv = self.OmegaComponents(self.B,self.z,self.blocksizes) # we need first output to calculate A_inv so no speedup can be gained by trying to calculate A_inv only
        OmegaInverse = (
            A_inv.transpose(-1, -2)
            @ OmegaTildeInverse
            @ A_inv
        )
        return OmegaInverse,A_inv, zeta, L_K #return zeta, L_K for speed to do the calculation onely once
    
    def Omega(self, symmetrize: bool = False):
        OmegaTilde, A_inv = self.OmegaComponents()
        # Compute A @ OmegaTilde without explicitly forming A
        X = torch.linalg.solve_triangular(
            A_inv,
            OmegaTilde,
            upper=False,
            left=True
        )

        # Compute X @ A.T = X @ A_inv^{-T}
        Omega = torch.linalg.solve_triangular(
            A_inv,
            X.transpose(-1, -2),
            upper=False,
            left=True
        ).transpose(-1, -2)   
        
        #symetrize to enforce symmetry constraint
        if symmetrize:  
            Omega = 0.5 * (
                Omega + Omega.transpose(-1, -2)
            )
         
        return Omega # [self.M, self.D, self.D]
    
    @abstractmethod
    def log_prob(
        self,
        value: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError
    
    @abstractmethod
    def rsample(
        self,
        sample_shape=torch.Size(),
    ) -> torch.Tensor:
        raise NotImplementedError
    
    @abstractmethod
    def sample(
        self,
        sample_shape=torch.Size(),
    ):
        raise NotImplementedError
    
    @abstractmethod
    def sample_and_log_prob(
        self,
        sample_shape=torch.Size(),
    ):
        raise NotImplementedError
