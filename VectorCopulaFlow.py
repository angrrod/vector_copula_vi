from pyro.distributions.torch_distribution import TorchDistribution
import torch

def Blockdiag(B,dimList):
    Bdiag = B.clone()
    
    prevDim = 0
    for dim in dimList:
        ind = dim + prevDim
        Bdiag[:ind,ind:] = 0
        Bdiag[ind:,:ind] = 0   
        prevDim += dim  
        
    return Bdiag


class VectorCopulaFlow(TorchDistribution):
    """
    Implementation of the VectorCopula distribution with 
    Zuko normalizing flows as marginals
    """
    arg_constraints = {}  # fill if you have constrained params
    support         = constraints.real_vector
    has_rsample     = True  # set True if you implement rsample()

    def __init__(self, flows:list, B:torch.Tensor, zeta:torch.Tensor):
        """
        flow_n  Python list of Zuko flows modeling the margina
        B       D x P matrix, where P < D, and D is event_shape
        zeta    Scalar parameter
        """
        self.flows = flows
        self.distribs = [flow() in flows]
        self.B      = B
        self.zeta   = zeta
        self.blocksizes = [dist.event_shape for dist in distribs]
        self.N      = self.B.shape[0]
        self.device = B.device #TODO: Use this variable every time you create a tensor!
        batch_shape = torch.Size() #TODO: imlement batching support
        event_shape = torch.Size([self.B.shape[0]])
        assert self.N == self.blocksizes.sum(), "Block sizes should add up to D"
        super().__init__(batch_shape=batch_shape, event_shape=event_shape, validate_args=None)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        value_split = torch.split(values, self.blocksizes)
        #Compute marginal contribution to logprob
        logp_marg = torch.zeros(len(self.distribs))
        Qlist = []
        for i in range(len(self.distribs)):
            logp_marg[i] = self.distribs[i].log_prob(value_split[i])
            Qlist.append(self.distribs[i].transform.inv(value_split[i]))
        Q = torch.concat(Qlist)
        #Compute copula contribution to logprob
        logDensity,_,_  = self._logProbCopula(Q)
        return logp_marg.sum() + logDensity
        
    def _logProbCopula(self, Q):
        """
        Compute copula contribution for logprob
        Q   Vector quantiles
        """

        Omega = self.Omega()
        I = torch.eye(self.N)
        L = torch.linalg.cholesky(Omega) #TODO: Use Woodbury formula 
        OmegaInv = torch.cholesky_solve(I, L)
        
        logDetTerm      = - torch.log(torch.diagonal(L)).sum()
        logCopulaTerm   = (-1/2) * Q @ (OmegaInv- I) @ Q.T
        logDensity      = logDetTerm + logCopulaTerm # up to a constant term
        return logDensity, logDetTerm, logCopulaTerm

    def Omega(self):
        OmegaTilde = self.zeta * torch.eye(self.N) + self.B @ self.B.T

        if not torch.isfinite(OmegaBar).all():
            raise RuntimeError("OmegaBar contains NaN/Inf")
        
        Bd = Blockdiag(OmegaBar, self.blocksizes)
        #Bd = Bd + 1e-6 * eye #TODO: Se if it works without
        L = torch.linalg.cholesky(Bd)

        A = torch.inverse(L)
        return A @ OmegaBar @ A.T

    def rsample(self, sample_shape=torch.Size()):
        Omega = self.Omega()
        MultiNormal = torch.distributions.MultivariateNormal(loc=torch.zeros((self.N))
                                                    , covariance_matrix=Omega)
        sample = dist.rsample(sample_shape)
        Z = torch.split(sample, self.blocksizes, dim=-1) #TODO:Check this
        sample_list = []
        for i in range(len(self.distribs)):
            sample_list.append(self.distribs[i].transform(Z[i]))
        return torch.cat(sample_list, dim=-1)

    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            return self.rsample(sample_shape)


