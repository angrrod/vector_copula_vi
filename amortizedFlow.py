#TODO: Test too many calls to vectorCopulaFlow
import torch.nn as nn
import torch
from .VectorCopulaFlow_V2 import ZukoVectorCopulaFlow, DingoVectorCopulaFlow
from .EllipticalVectorCopulaFlow import EllipticalVectorCopulaFlow
from abc import ABC, abstractmethod

class AmortizedCopulaFlow(nn.Module,ABC):
    """
    extenstion of the vector copula flow to amortized zetting, for both zuko and Dingo
    """
    def __init__(self, flows:nn.ModuleList, parameter_net:nn.Module, debug_checks:bool=False,is_independent:bool = False):
        """
            args:
            flows              nn.ModuleList of zuko,flsow modeling the marginals
            parameter_net      neural network, that allows to obtain the model parameters conditional on the context
            debug_checks       bool, allow certain test options
            marginal_backend   str, indicating what kind of backend is needed to be used: accepted values are 'zuko' or 'dingo'
            is_independent     boolean, indicating if the copula is independent or not, in the latter case, B and zeta are fixed and non trainable
        """
        super().__init__()
        if isinstance(flows, nn.ModuleList):
            self.flows = flows
        else:
            self.flows = nn.ModuleList(flows)
        self.parameter_net    = parameter_net
        self.debug_checks     = debug_checks
        self.is_independent   = is_independent
    
    @abstractmethod
    def distribution(self, context:torch.Tensor):
        raise NotImplementedError
    
    def log_prob(self, value, context):
        return self.distribution(context).log_prob(value)
    
    def log_prob_components(self, value, context):
        return self.distribution(context).log_prob_components(value)
    
    def rsample(self, sample_shape=torch.Size(), context=None):
        res = self.distribution(context).rsample(sample_shape)
        return res
    
    def sample(self, sample_shape=torch.Size(), context=None):
        with torch.no_grad():
            return self.rsample(sample_shape, context)
    
    def sample_and_log_prob(self,N, context):
        return self.distribution(context).sample_and_log_prob(N)
    
class AmortizedVectorCopulaFlow(AmortizedCopulaFlow):
    """
    extenstion of the vector copula flow to amortized zetting, for both zuko and Dingo
    """
    def __init__(self, flows:nn.ModuleList, parameter_net:nn.Module, debug_checks:bool=False, marginal_backend: str = "zuko",is_independent:bool = False):
        """
            args:
            flows              nn.ModuleList of zuko,flsow modeling the marginals
            parameter_net      neural network, that allows to obtain the model parameters conditional on the context
            debug_checks       bool, allow certain test options
            marginal_backend   str, indicating what kind of backend is needed to be used: accepted values are 'zuko' or 'dingo'
            is_independent     boolean, indicating if the copula is independent or not, in the latter case, B and zeta are fixed and non trainable
        """
        super().__init__(flows, parameter_net, debug_checks=debug_checks,is_independent = is_independent)
        assert marginal_backend in ('zuko', 'dingo'), f"marginal_backend option not in (zuko, dingo); got {marginal_backend}"
        self.marginal_backend = marginal_backend
    
    def distribution(self, context:torch.Tensor):
        """
        Args:
            context (torch.Tensor):   tensor describing the parameters of B and z of size [S,I] #I is the input of the parameter net obtained from the compression of the GW form
        
        """
        if self.is_independent:
            # We only need B and z here to recover the required tensor shapes.
            # Do not construct a gradient graph through parameter_net.
            with torch.no_grad():
                B, z = self.parameter_net(context)
        else:
            B, z = self.parameter_net(context)
        
        #initialize a new object everytime, but we need to make the constructor as light as possible
        if self.marginal_backend == "zuko":
            distribs = [flow(context) for flow in self.flows] #needed for amortization
            return ZukoVectorCopulaFlow(
                flows          = self.flows,
                B              = B,
                z              = z,
                distribs       = distribs,
                debug_checks   = self.debug_checks,
                is_independent = self.is_independent
            )
        else: #dingo
            return DingoVectorCopulaFlow(
                flows          = self.flows,
                B              = B,
                z              = z,
                context        = context,
                debug_checks   = self.debug_checks,
                is_independent = self.is_independent
            )
            
class AmortizedMixtureVectorCopulaFlow(AmortizedCopulaFlow):
    def __init__(self, flows:nn.ModuleList, parameter_net:nn.Module, debug_checks:bool=False,is_independent:bool = False):
        """
            args:
            flows              nn.ModuleList of zuko,flsow modeling the marginals
            parameter_net      neural network, that allows to obtain the model parameters conditional on the context
            debug_checks       bool, allow certain test options
            marginal_backend   str, indicating what kind of backend is needed to be used: accepted values are 'zuko' or 'dingo'
            is_independent     boolean, indicating if the copula is independent or not, in the latter case, B and zeta are fixed and non trainable
        """
        super().__init__(flows, parameter_net, debug_checks=debug_checks,is_independent = is_independent)
        self.blocksizes = [self._infer_flow_event_dim(flow) for flow in flows]

    def _infer_flow_event_dim(self,flow_wrapper):
        distribution = flow_wrapper.flow._distribution

        if hasattr(distribution, "_shape"):
            return distribution._shape[0]

        if hasattr(distribution, "shape"):
            return distribution.shape[0]

        raise ValueError("Could not infer event dimension from FlowWrapper.")
    
    def distribution(self, context:torch.Tensor):
        """
        Args:
            context (torch.Tensor):   tensor describing the parameters of B and z of size [S,I] #I is the input of the parameter net obtained from the compression of the GW form
        
        """
        if self.is_independent:
            # We only need B and z here to recover the required tensor shapes.
            # Do not construct a gradient graph through parameter_net.
            with torch.no_grad():
                B, z, raw_scale_gaps, weight_logits = self.parameter_net(context)
        else:
            B, z, raw_scale_gaps, weight_logits  = self.parameter_net(context)

        #initialize a new object everytime, but we need to make the constructor as light as possible
        return EllipticalVectorCopulaFlow(
                flows                = self.flows,
                B                    = B,
                z                    = z,
                context              = context,
                debug_checks         = self.debug_checks,
                is_independent       = self.is_independent,
                blocksizes           = self.blocksizes,
                raw_scale_gaps       = raw_scale_gaps,
                weight_logits        = weight_logits,
                learn_mixture_params = False,
            )
            