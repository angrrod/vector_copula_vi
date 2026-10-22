import torch

from abc import ABC, abstractmethod
from .factor_pattern_copula import FactorPatternCorrelationModel
import torch.nn as nn
class GaussianVectorCopulaFlow(FactorPatternCorrelationModel,ABC): 
    """
    Implementation of the VectorCopula distribution with 
    Zuko normalizing flows as marginals
    """
    def __init__(self, flows:nn.ModuleList, B:torch.Tensor, z:torch.Tensor,blocksizes:list, debug_checks:bool = False, is_independent:bool = False):
        """
        args:
            flows           nn.ModuleList of Zuko flows modeling the marginal
            B               M x D x P matrix, where P < D, D is event_shape and M is the batch shape denoting different distributions, used in amortization
            z               M vector parameter
            debug_checks    bool indicating whether or not to check the parameter inputs for nan's
            is_independent  boolean indicating if the copula is independent or not, in the latter case, B and zeta are fixed and non trainable
        """
        super().__init__(flows=flows, B=B, z=z,blocksizes=blocksizes, debug_checks=debug_checks, is_independent=is_independent)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """
        calculate log probability for the Vector copula model
        value:    torch.Tensor,  value.shape == [S, self.M, self.D]    # sample_shape (nbr samples, for minibatches) + batch_shape (nbr paralell models) + event_shape
                  shape can be 2d, then self.M == 1 is assumed
        """
        return self.log_prob_components(value)["log_prob_total"]
    
    def sample(self, sample_shape=torch.Size(), symmetrize_omega:bool = True):
        with torch.no_grad():
            return self.rsample(sample_shape, symmetrize_omega)#[sample_shape, self.M, self.D]
    
    def sample_and_log_prob(self,N):
        samples = self.sample(sample_shape=N)
        log_q   = self.log_prob(samples)
        return samples,log_q
    
    @abstractmethod
    def _marginal_log_prob(self, i: int, value_i: torch.Tensor) -> torch.Tensor:
        """
        returns log prabaility of the marginals
        
        Args:
            i (int): i-th flow
            value_i (torch.Tensor): value for the i-th flow
        """
        raise NotImplementedError

    @abstractmethod
    def _to_base(self, i: int, value_i: torch.Tensor,input_was_2d:bool = False) -> torch.Tensor:
        """
        Transforms to base distribution
        
        Args:
            i (int): i-th flow
            value_i (torch.Tensor): value for the i-th flow
        """
        raise NotImplementedError

    @abstractmethod
    def _from_base(self, i: int, z_i: torch.Tensor) -> torch.Tensor:
        """
        Transforms from base to the complex distribution
        
        Args:
            i (int): i-th flow
            z_i (torch.Tensor): value for the i-th flow
        """
        raise NotImplementedError

    def log_prob_components(self, value: torch.Tensor) -> dict:
        #check if input is 2d or not and add middle dimension of needed
        input_was_2d  = False

        if value.dim() == 2:
            input_was_2d  = True
        elif value.dim() != 3:
            raise ValueError(
                f"Expected value shape [M, D] or [S, M, D], got {tuple(value.shape)}"
            )

        value_split = torch.split(value, self.blocksizes, dim=-1)

        logp_marg_list = []
        Qlist = []

        for i, value_i in enumerate(value_split):
            logp_marg_val = self._marginal_log_prob(i, value_i)
            #if marginal_log_prob reduced the shape because of mismatch between value and context, add model batch back
            if len(logp_marg_val.shape) == 1:
                logp_marg_val = logp_marg_val.unsqueeze(1)
            logp_marg_list.append(logp_marg_val)
            Qlist.append(self._to_base(i, value_i,input_was_2d))

        logp_marg_total = torch.stack(logp_marg_list, dim=0).sum(dim=0) 
        Q = torch.cat(Qlist, dim=-1)

        # The copula algebra expects [S, M, D].
        # For ordinary training input [M, D], insert implicit S = 1 only here.
        if input_was_2d:
            Q_for_copula = Q.unsqueeze(0)                         # [1, M, D]
            logp_marg_total_for_copula = logp_marg_total.unsqueeze(0)  # [1, M]
        else:
            Q_for_copula = Q                                      # [S, M, D]
            logp_marg_total_for_copula = logp_marg_total          # [S, M]

        log_copula_total, log_det_term, log_copula_quad = self._logProbCopula(
            Q_for_copula
        )

        out = log_copula_total + logp_marg_total_for_copula

        components = {
            "log_prob_total": out, #[S,M]
            "logp_marg_total": logp_marg_total_for_copula, #[S,M]
            "logp_marginals": logp_marg_list, #[S]
            "log_copula_total": log_copula_total, #[S,M]
            "log_det_term": log_det_term, #[1]
            "log_copula_quad": log_copula_quad, #[S,M]
        }

        # Return [M] if the input was [M, D].
        if input_was_2d:
            components["log_prob_total"] = components["log_prob_total"].squeeze(0)
            components["logp_marg_total"] = components["logp_marg_total"].squeeze(0)
            components["log_copula_total"] = components["log_copula_total"].squeeze(0)
            components["log_det_term"] = components["log_det_term"].squeeze(0)
            components["log_copula_quad"] = components["log_copula_quad"].squeeze(0)

            # These were computed directly as [M], so do not squeeze them.
            components["logp_marginals"] = logp_marg_list

        return components

    def rsample(self, sample_shape=torch.Size(), symmetrize_omega:bool = False):
        if isinstance(sample_shape, int):
            sample_shape = (sample_shape,)
        
        Omega = self.Omega(symmetrize_omega)

        multi_normal = torch.distributions.MultivariateNormal(
            loc=torch.zeros(
                self.batch_shape + self.event_shape,
                device=self.device,
                dtype=self.dtype,
            ),
            covariance_matrix=Omega,
        )

        sample = multi_normal.rsample(sample_shape)
        Z_split = torch.split(sample, self.blocksizes, dim=-1)

        sample_list = [
            self._from_base(i, z_i)
            for i, z_i in enumerate(Z_split)
        ]

        return torch.cat(sample_list, dim=-1)
    
class ZukoVectorCopulaFlow(GaussianVectorCopulaFlow):
    """
    Common copula algebra and shape checks.
    Does not assume a particular marginal-flow interface.
    """
    def __init__(self, flows:nn.ModuleList, B:torch.Tensor, z:torch.Tensor,distribs = None, debug_checks:bool = False,is_independent:bool=False):
        if distribs is not None:
            self.distribs = distribs
        else:
            self.distribs = [flow() for flow in flows] # condition 
        blocksizes = [dist.event_shape[0] for dist in self.distribs]
        super().__init__(flows, B, z,blocksizes,debug_checks,is_independent=is_independent)
        
    def _marginal_log_prob(self, i, value_i):
        return self.distribs[i].log_prob(value_i)

    def _to_base(self, i, value_i, input_was_2d:bool):
        return self.distribs[i].transform(value_i)

    def _from_base(self, i, z_i):
        return self.distribs[i].transform.inv(z_i)
    
class DingoVectorCopulaFlow(GaussianVectorCopulaFlow):
    """
    Marginals are Dingo FlowWrapper / nflows objects conditioned on context.
    """
    def __init__(self, flows:nn.ModuleList, B:torch.Tensor, z:torch.Tensor,context, debug_checks:bool = False,is_independent:bool=False):
        blocksizes = [self._infer_flow_event_dim(flow) for flow in flows]
        super().__init__(flows, B, z, blocksizes, debug_checks,is_independent=is_independent)
        self.context = context
        
    def _infer_flow_event_dim(self,flow_wrapper):
        distribution = flow_wrapper.flow._distribution

        if hasattr(distribution, "_shape"):
            return distribution._shape[0]

        if hasattr(distribution, "shape"):
            return distribution.shape[0]

        raise ValueError("Could not infer event dimension from FlowWrapper.")
    
    def _marginal_log_prob(self, i, value_i):
        context,value_i = self._shape_for_log_prob(value_i)
        return self.flows[i].log_prob(value_i, context)

    def _to_base(self, i, value_i,input_was_2d:bool):
        context,value_i = self._shape_for_log_prob(value_i)
        q_i, _ = self.flows[i].flow._transform(
            value_i,
            context=context,
        )
        if (len(q_i.shape) == 2) and not input_was_2d:
            q_i = q_i.unsqueeze(1)
        return q_i

    def _from_base(self, i, z_i):
        z_i_mod,context = self._shape_for_sampling(z_i)
        value_i, _ = self.flows[i].flow._transform.inverse(
            z_i_mod,
            context = context,
        )
        value_i_out = self._reshape_for_sampling(value_i)
        return value_i_out
    
    #TODO: test expands if ok? -> multiple contexts?
    def _shape_for_log_prob(self, value: torch.Tensor):
        """fixes the length of tensors by removing batches of models
            value and self.context
        Args:
            value (torch.Tensor): the tensor used for evaluation of the log_prob (so a parameter tensor)
        """
        context = self.context
        if context is not None:
            if (len(self.context.shape) == 2) & (len(value.shape) == 3): #hit
                value = value.squeeze(1)
            if (len(self.context.shape) == 3) & (len(value.shape) == 2):
                context = context.squeeze(1)
            if (context.shape[0] == 1) & (value.shape[0] != 1):
                context = context.expand(value.shape[0],-1)
            #TODO: test which sizes fit -> bring out of function
            # self.flows[0].log_prob(value_i.squeeze(1), context.squeeze(1))
        return context,value
    
    def _reshape_for_sampling(self, value: torch.Tensor):
        return value.unsqueeze(1)
    
    #TODO: test expands if ok? -> multiple contexts?
    def _shape_for_sampling(self, value: torch.Tensor):
        """
        shape the context and value for sampling

        Args:
            value (torch.Tensor): value of the log probability, to be evaluated
        """
        context = self.context
        if context is not None:
            if (len(self.context.shape) == 2) & (len(value.shape) == 3):
                value = value.squeeze(1)
            if (self.context.shape[0] == 1) & (value.shape[0] != 1):
                context = self.context.expand(value.shape[0], -1)
        else:
            value = value.squeeze(1)
        return value, context


    