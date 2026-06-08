from pyro.distributions.torch_distribution import TorchDistribution
import torch
from torch.distributions import constraints
import torch.nn.functional as F

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

def getLogCopulaTerm(Q: torch.Tensor,Mmat: torch.Tensor):
    """calculate the copula contribution to the logprobability, where both the quantiles and the model matrix are involved.
    [S, self.M, self.D] @ [self.M, self.D, self.D] @ [S, self.M, self.D]
    -> for each M and for each S do Q @ (OmegaInv- I_batch) @ Q.T
    We use bmm, to obtain maximal efficienty but one needs flattening for this.

    Args:
        Q (torch.Tensor): tensor of size [S, self.M, self.D] representing vector quantiles
        Mmat (torch.Tensor): tensor of size [self.M, self.D, self.D] represent omega-I

    Returns:
        torch.Tensor: result of shape [S*M]
    """
    S, M, D = Q.shape
    Q_flat = Q.reshape(S * M, D)  # [S*M, D]
    Mmat_flat = (
        Mmat.unsqueeze(0)
            .expand(S, M, D, D)
            .reshape(S * M, D, D)
    )  # [S*M, D, D]
    
    MQ = torch.bmm(
        Mmat_flat,
        Q_flat.unsqueeze(-1),
    ).squeeze(-1)  # [S*M, D]
    
    quad = (Q_flat * MQ).sum(dim=-1)  # [S*M]
    return quad.reshape(S, M)          # [S, M]

def WoodburyComponents(z: torch.Tensor,B: torch.Tensor) -> torch.Tensor:
    """returns the inverse of (zeta*I + B*B^T) using the woodbury formula, this inverts a PxP matrix instead of a DxD matrix which is faster for low rank approximations 

    Args:
        z (torch.Tensor): matrix of shape [M], used to guarantee PD'ness
        B (torch.Tensor): matrix of shape [M,D,P]

    Returns:
        torch.Tensor: inverse of (zeta*I + B*B^T)
    """
    zeta    = F.softplus(z)
    M, D, P = B.shape
    Bt      = B.transpose(-1, -2)
    zeta_b  = zeta.view(M, 1, 1)  # [M, 1, 1]
    
    I_d     = torch.eye(D, device=B.device, dtype=B.dtype)
    I_p     = torch.eye(P, device=B.device, dtype=B.dtype)
    
    I_d_batch = I_d.expand(M,D,D)  #[M,D,D]
    I_p_batch = I_p.expand(M,P,P)  #[M,P,P]
    
    K       = zeta_b * I_p_batch + Bt @ B  #[M,P,P]
    L_K     = torch.linalg.cholesky(K) # [M, P, P]
    Kinv_Bt = torch.cholesky_solve(Bt, L_K) #[M, P, D] #obtain K^-1 @ B^T, #only need to calc L_K once in the stack (other place would be LogDetOmegaLowRank)
    
    Woodbury = (I_d_batch - B @ Kinv_Bt)/zeta_b  #[M,D,D]
    return Woodbury, zeta, L_K  #return zeta, L_K for speed to do the calculation onely once

class VectorCopulaFlow_V2(TorchDistribution): 
    """
    Implementation of the VectorCopula distribution with 
    Zuko normalizing flows as marginals
    """
    arg_constraints = {}  # fill if you have constrained params
    support         = constraints.real_vector
    has_rsample     = True  # set True if you implement rsample()
    
    def __init__(self, flows:list, B:torch.Tensor, z:torch.Tensor):
        """
        flow_n  Python list of Zuko flows modeling the margina
        B       M x D x P matrix, where P < D, D is event_shape and M is the batch shape denoting different distributions, used in amortization
        z       M vector parameter
        """
        self.flows = flows
        self.distribs = [flow() for flow in flows]
        self.B      = B
        self.z      = z #[self.M]; we will take the softplus of this to obtain zeta to make it strictly positive

        assert B.device == z.device, (
            f"Device mismatch: z is on {z.device}, B is on {B.device}"
        )
        assert B.dtype == z.dtype, (
            f"Dtype mismatch: B has dtype {B.dtype}, z has dtype {z.dtype}"
        )
        assert self.B.shape[1] > self.B.shape[2], (
            f"B must have shape D > P but got: D = {self.B.shape[1]}, P = {self.B.shape[2]}"
        )
        assert self.z.shape[0] == self.B.shape[0], (
            f"B must have shape batch size M as z, but got: M_B = {self.B.shape[0]}, M_z = {self.z.shape[0]}"
        )
        
        self.M          = self.B.shape[0]
        self.D          = self.B.shape[1]
        self.P          = self.B.shape[2] # low rank approximation
        
        self.blocksizes = [dist.event_shape[0] for dist in self.distribs]
        self.device     = B.device 
        self.dtype      = B.dtype
        batch_shape     = torch.Size([self.M]) #TODO: imlement batching support 
        event_shape     = torch.Size([self.D])  
        
        assert self.D == torch.tensor(self.blocksizes, dtype=torch.int, device=self.device).sum(), "Block sizes should add up to D"
        super().__init__(batch_shape=batch_shape, event_shape=event_shape, validate_args=None)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """
        calculate log probability for the Vector copula model
        value:    torch.Tensor,  value.shape == [S, self.M, self.D]    # sample_shape (nbr samples, for minibatches) + batch_shape (nbr paralell models) + event_shape
        """
        return self.log_prob_components(value)["log_prob_total"]
    
    def log_prob_components(self,value: torch.tensor) -> dict:
        """returns the separate log probability components
        
        Args:
            value (torch.tensor): [S, M, D]
            
        Returns:
            dict: with:         
        log_prob_total:     [S, M]
        logp_marg_total:    [S, M]
        logp_marginals:     list of tensors, each [S, M]
        log_copula_total:   [S, M]
        log_det_term:       [S, M]
        logCopulaTerm:      [S, M]
        """
        #->TODO REMOVE
        added_sample_dim = False
        if value.dim() == 2:
            # Pyro often passes a single draw with shape [M, D].
            # Internally this class expects [S, M, D], so add S = 1.
            value = value.unsqueeze(0)
            added_sample_dim = True

        elif value.dim() == 3:
            # Already in the internal shape convention [S, M, D].
            pass

        else:
            raise ValueError(
                f"Expected value shape [M, D] or [S, M, D], got {tuple(value.shape)}"
            )
        #<- end remove
        
        value_split = torch.split(value, self.blocksizes,dim = -1) #[S, self.M, blocksize]
        
        #Compute contribution to logprob for each marginal 
        logp_marg_list  = []
        Qlist = []
        for dist, value_i in zip(self.distribs, value_split):
            logp_marg_list.append(dist.log_prob(value_i))      # [S, M]
            Qlist.append(dist.transform.inv(value_i))          # [S, M, d_i]
            
        logp_marg_total = torch.stack(logp_marg_list, dim=0).sum(dim=0)  # [S, M]
        Q = torch.cat(Qlist, dim=-1)                               # [S, M, D]
        #Compute copula contribution to logprob
        log_copula_total,log_det_term, log_copula_quad  = self._logProbCopula(Q) #[S, self.M]
        out = log_copula_total + log_copula_total

        
        components = {
            "log_prob_total": out,
            "logp_marg_total": logp_marg_total,
            "logp_marginals": logp_marg_list,
            "log_copula_total": log_copula_total,
            "log_det_term": log_det_term,
            "log_copula_quad": log_copula_quad,
        }

        # -> TODO: Remove
        if added_sample_dim:
            for key, val in components.items():
                if isinstance(val, torch.Tensor):
                    components[key] = val.squeeze(0)

            components["logp_marginals"] = [
                x.squeeze(0) for x in components["logp_marginals"]
            ]
        # <-    
        
        return components
    
    def _logProbCopula(self, Q):
        """
        Compute copula contribution for logprob
        Q   Vector quantiles    Q.shape == [S, self.M, self.D]
        """
        I        = torch.eye(self.D, device=self.device, dtype = self.dtype)
        I_batch  = I.expand(self.M,self.D,self.D)
        OmegaInv,A_inv, zeta, K = self.InvertOmega()  #[self.M, self.D, self.D]
        
        logDetTerm    = -0.5 * self.LogDetOmegaLowRank(A_inv,zeta,K)
        logCopulaTerm = -0.5 * getLogCopulaTerm(Q,Mmat = OmegaInv- I_batch)
        
        logDensity      = logDetTerm + logCopulaTerm # up to a constant term
        return logDensity, logDetTerm, logCopulaTerm
    
    def LogDetOmegaLowRank(
        self,
        A_inv: torch.Tensor,
        zeta: torch.Tensor,
        L_K: torch.Tensor,
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
        OmegaTildeInverse, zeta, L_K = WoodburyComponents(self.z,self.B) 
        _, A_inv = self.OmegaComponents() # we need first output to calculate A_inv so no speedup can be gained by trying to calculate A_inv only
        OmegaInverse = (
            A_inv.transpose(-1, -2)
            @ OmegaTildeInverse
            @ A_inv
        )
        return OmegaInverse,A_inv, zeta, L_K #return zeta, L_K for speed to do the calculation onely once
    
    def OmegaComponents(self):
        I          = torch.eye(self.D, device=self.device, dtype = self.dtype)
        I_batch    = I.expand(self.M,self.D,self.D)  # [self.M, self.D, self.D]
        zeta       = F.softplus(self.z)#force zeta to be positive as it acts as regulartor fo positive definiteness
        OmegaTilde = zeta[:, None, None] * I_batch + self.B @ self.B.transpose(-1, -2) # [self.M, self.D, self.D]

        if not torch.isfinite(OmegaTilde).all():
            raise RuntimeError("OmegaBar contains NaN/Inf")
        
        Bd = Blockdiag(OmegaTilde, self.blocksizes)  # [self.M, self.D, self.D]
        A_inv = torch.linalg.cholesky(Bd)      # [self.M, self.D, self.D]
        return OmegaTilde, A_inv
    
    def Omega(self):
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
        return Omega # [self.M, self.D, self.D]
    
    def rsample(self, sample_shape=torch.Size()):
        Omega = self.Omega() # [self.M, self.D, self.D]
        MultiNormal = torch.distributions.MultivariateNormal(
            loc=torch.zeros(self.batch_shape + self.event_shape, # [M, D]
                                device=self.device,
                                dtype = self.dtype
                            ),
            covariance_matrix=Omega
            )
        sample = MultiNormal.rsample(sample_shape) #[sample_shape, self.M, self.D] #sample_shape can be empty
        Z = torch.split(sample, self.blocksizes, dim=-1)
        sample_list = []
        for dist, Z_i in zip(self.distribs, Z):  #use zip because split creates a list.
            sample_i = dist.transform(Z_i)
            sample_list.append(sample_i)
        return torch.cat(sample_list, dim=-1) #[sample_shape, self.M, self.D]
    
    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            return self.rsample(sample_shape)#[sample_shape, self.M, self.D]
    