from pyro.distributions.torch_distribution import TorchDistribution
import torch
import torch.nn as nn
import torch.distributions as D
from .factor_pattern_copula import FactorPatternCorrelationModel

class EllipticalVectorCopulaFlow(FactorPatternCorrelationModel):
    def __init__(self,B:torch.tensor,z:torch.tensor, flows:nn.ModuleList,blocksizes:list,raw_scale_gaps:torch.tensor,weight_logits:torch.tensor,is_independent:bool = False,debug_checks:bool = False,context = None, learn_mixture_params:bool=True):
        """_summary_

        Args:
            device: torch device
            flows (nn.ModuleList): module list of all the marginal NF's
            blocksizes (list): list containing the sizes of the marginal (inputs)
            B               M x D x P matrix, where P < D, D is event_shape and M is the batch shape denoting different distributions, used in amortization
            z               M vector parameter
            scales (torch.tensor): scale variables for the magnitude of the shape tensor
            weights (torch.tensor): weight tensor for the different tensors
            is_independent (bool): not implemented
            debug_checks (bool): indicating wether debug checks should be done on the input
            context (torch.tensor) : context for conditional model used in SBI
            learn_mixture_params (boolean) : indicator if mixture parameters are constraint in this object or not
        """
        
        ### B               M x D x P matrix, where P < D, D is event_shape and M is the batch shape denoting different distributions, used in amortization        
        super().__init__(
            flows=flows,
            B=B,
            z=z,
            blocksizes=blocksizes,
            is_independent=is_independent,
            debug_checks=debug_checks
        )
        #force them on the simplex -> sum will be one
        self.context = context
        
        self.learn_mixture_params = learn_mixture_params
        if weight_logits.shape[-1] != raw_scale_gaps.shape[-1] + 1:
            raise ValueError(
                "Expected J mixture logits and J-1 raw scale gaps."
            )
            
        if learn_mixture_params:
            self.weight_logits = nn.Parameter(
                weight_logits.detach().clone()
            ) #[M, J] 

            self.raw_scale_gaps = nn.Parameter( #new param is being created
                raw_scale_gaps.detach().clone() #also forces the shapes to be the same
            ) #[M, J-1]
        else:
            # Amortized: retain gradients through parameter_net.
            # Expected shape: [M, J].
            self._weight_logits = weight_logits
            self._raw_scale_gaps = raw_scale_gaps
            
        self._refresh_marginals()
        self._refresh_elliptical_state()
    
    @property
    def weights(self):
        if self.learn_mixture_params:
            logits = self.weight_logits
        else:
            logits = self._weight_logits

        return torch.softmax(logits, dim=-1)
    
    @property
    def log_scales(self):
        if self.learn_mixture_params:
            raw = self.raw_scale_gaps 
        else:
            raw = self._raw_scale_gaps

        gaps = torch.nn.functional.softplus(raw)

        zero = raw.new_zeros(
            raw.shape[:-1] + (1,)
        )

        return torch.cat(
            [
                zero,
                torch.cumsum(gaps, dim=-1),
            ],
            dim=-1,
        )

    @property
    def scales(self):
        return torch.exp(self.log_scales)
    
    def elliptical_distribution(self):
        """
        returns the ellitpical distribution that is the basis of this vector copula model

        Returns:
            torch.distributions: vector copula model basis
        """
        weights, scales = self._batched_mixture_parameters()

        J = weights.shape[-1]

        covariance_matrix = (
            scales[..., None, None].square()
            * self.Omega_matrix[:, None, :, :]
        )

        # [M, J, D]
        loc = torch.zeros(
            self.M,
            J,
            self.D,
            device=self.device,
            dtype=self.dtype,
        )

        components = torch.distributions.MultivariateNormal(
            loc=loc,
            covariance_matrix=covariance_matrix,
        )

        mixture = torch.distributions.Categorical(
            probs=weights.expand(self.M, -1)
        )

        return torch.distributions.MixtureSameFamily(
            mixture_distribution=mixture,
            component_distribution=components,
        )
    
    def _create_omega_cholesky_blocks(self):
        """
        creates lists of cholesky factors and inverse cholesky factors of the submatrices on the diagonal.
        used in the transport and inverse transport maps
        """
        
        # Slices corresponding to each marginal block
        self.block_slices = []

        start = 0
        for d_i in self.blocksizes:
            stop = start + d_i
            self.block_slices.append(slice(start, stop))
            start = stop


        # Cholesky factor of each marginal covariance block
        self.L_omega_blocks = []
        self.L_omega_inv_blocks = []

        for idx in self.block_slices:

            Omega_i = self.Omega_matrix[..., idx, idx]

            # Numerical symmetry safeguard
            Omega_i = 0.5 * (
                Omega_i + Omega_i.transpose(-1, -2)
            )

            L_i = torch.linalg.cholesky(Omega_i)

            d_i = L_i.shape[-1]

            eye = torch.eye(
                d_i,
                device=L_i.device,
                dtype=L_i.dtype,
            )

            # Match possible batch dimensions of L_i
            eye = eye.expand(
                *L_i.shape[:-2],
                d_i,
                d_i,
            )

            L_i_inv = torch.linalg.solve_triangular(
                L_i,
                eye,
                upper=False,
            )
            self.L_omega_blocks.append(L_i)
            self.L_omega_inv_blocks.append(L_i_inv)
    
    def inverse_elliptical_transport(self, inv_value, i:int,eps = 1e-7):
        """inverse elliptical transport map

        Args:
            inv_value (torch.tensor): (S x M x d_i) value to be transported back
            i (int): it'h marginal to be transported back
            eps (float, optional): error tolerance. Defaults to 1e-7.

        Returns:
            torch.tensor: inverse transported value
        """
        m    = self.blocksizes[i]
        v    = torch.matmul(self.L_omega_inv_blocks[i],inv_value.unsqueeze(-1)).squeeze(-1)
        rnge = torch.linalg.vector_norm(v,dim=-1)
        dir  = v/rnge.clamp_min(eps)[..., None]
        chi2 = torch.distributions.Chi2(
            torch.as_tensor(
                float(m),
                device=inv_value.device,
                dtype=inv_value.dtype,
            )
        )
        
        #no pdf for chi2 -> use log-prob
        rho_sq,_ = self._invert_monotone_cdf(p = self.radial_cdf(rnge,m),cdf_fn = chi2.cdf,pdf_fn = lambda x: torch.exp(chi2.log_prob(x)),initial_hi =float(m))
        rho      = torch.sqrt(rho_sq)
        return rho[..., None] * dir
    
    def elliptical_transport(self,value,i):
        """elliptical transport map

        Args:
            value (torch.tensor): (S x M x d_i) value to be transported forward
            i (int): it'h marginal to be transported forward

        Returns:
            torch.tensor: transported value
        """
        m = self.blocksizes[i]
        
        if value.shape[-1] != m:
            raise ValueError(
                f"Block {i} has dimension {m}, "
                f"but value has last dimension {value.shape[-1]}"
            )
        
        chi2          = torch.distributions.Chi2(df=torch.tensor(float(m)))
        z             = self.marginals[i].transform(value)  #transform.inv
        # Gaussian radius
        rho           = torch.linalg.vector_norm(z,dim=-1)
        p             = chi2.cdf(rho.square())
        r,_           = self._invert_monotone_cdf(
                            p, 
                            cdf_fn=lambda x: self.radial_cdf(x, m),
                            pdf_fn=lambda x: self.radial_pdf(x, m),
                            initial_hi=self.scales.max()
                        )
        direction     = z/rho[...,None]
        L_i           = self.L_omega_blocks[i]
        direction     = torch.matmul(L_i, direction.unsqueeze(-1),).squeeze(-1)
        return r[..., None] * direction
    
    def _invert_monotone_cdf(self,p,cdf_fn,pdf_fn,initial_hi,n_steps=30,bracket_steps=30,eps=1e-8,return_diagnostics=True):
        """
        Numerically invert a monotonically increasing CDF.

        Parameters
        ----------
        p:
            Target probabilities, arbitrary batch shape.

        cdf_fn:
            Callable x -> F(x).

        pdf_fn:
            Callable x -> F'(x).

        initial_hi:
            Scalar or tensor broadcastable to p giving the initial
            upper bracket.

        Returns
        -------
        x:
            Approximation to F^{-1}(p).

        diagnostics:
            Optional dictionary containing numerical inversion errors.
        """

        p = p.clamp(eps, 1.0 - eps)

        lo = torch.zeros_like(p)

        hi = torch.ones_like(p) * torch.as_tensor(
            initial_hi,
            device=p.device,
            dtype=p.dtype,
        )

        # ------------------------------------------------------------
        # Construct bracket
        # ------------------------------------------------------------

        for _ in range(bracket_steps):
            F_hi = cdf_fn(hi)

            hi = torch.where(
                F_hi < p,
                2.0 * hi,
                hi,
            )

        # Make sure the requested probability was actually bracketed.
        F_hi = cdf_fn(hi)
        bracketed = F_hi >= p

        # Initial point
        x = 0.5 * (lo + hi)

        # ------------------------------------------------------------
        # Safeguarded Newton
        # ------------------------------------------------------------

        for i in range(n_steps):
            # if i % 5 == 0:
            #     print(f"step: {i}")
            F = cdf_fn(x)
            f = pdf_fn(x)

            residual = F - p

            x_newton = (
                x
                - residual / f.clamp_min(eps)
            )

            valid_newton = (
                torch.isfinite(x_newton)
                & (x_newton > lo)
                & (x_newton < hi)
            )

            x_bisect = 0.5 * (lo + hi)

            candidate = torch.where(
                valid_newton,
                x_newton,
                x_bisect,
            )

            F_candidate = cdf_fn(candidate)

            candidate_too_low = (
                F_candidate < p
            )

            lo = torch.where(
                candidate_too_low,
                candidate,
                lo,
            )

            hi = torch.where(
                candidate_too_low,
                hi,
                candidate,
            )

            x = candidate

        if not return_diagnostics:
            return x, None

        F_x = cdf_fn(x)

        diagnostics = {
            "cdf_error": (F_x - p).abs(),
            "cdf_rel_error" : (F_x - p).abs()/F_x.abs(),
            "bracket_width": hi - lo,
            "bracketed": bracketed,
        }
        
        error_tol = 1e-3
        if torch.any(diagnostics["cdf_error"] > error_tol) or torch.any(diagnostics["cdf_rel_error"] > error_tol):
            # cdf_error = diagnostics["cdf_error"][torch.where(diagnostics["cdf_error"] > error_tol)]

            # print(f"$$$ cdf_error: {cdf_error}")
            print(
                "CDF error mean:",
                diagnostics["cdf_error"].mean().item()
            )

            print(
                "CDF error max:",
                diagnostics["cdf_error"].max().item()
            )

            print(
                "bracket width mean:",
                diagnostics["bracket_width"].mean().item()
            )

            print(
                "bracket width max:",
                diagnostics["bracket_width"].max().item()
            )

            print(
                "all bracketed:",
                diagnostics["bracketed"].all().item()
            )
        return x, diagnostics
    
    def _refresh_elliptical_state(self):
        """
        Recompute all quantities that depend on trainable
        correlation parameters B and z.
        """

        self.Omega_matrix = self.Omega()

        self.basis_elliptical_distribution = (
            self.elliptical_distribution()
        )
        self._create_omega_cholesky_blocks()
    
    def _refresh_marginals(self):
        self.marginals = [
            flow(self.context)
            if self.context is not None else flow()
            for flow in self.flows
        ]
    
    def _batched_mixture_parameters(self):
        #dynamically enforce batching in variables
        weights = self.weights
        scales = self.scales

        if weights.ndim == 1:
            weights = weights.unsqueeze(0).expand(self.M, -1)

        if scales.ndim == 1:
            scales = scales.unsqueeze(0).expand(self.M, -1)

        if weights.shape != scales.shape:
            raise ValueError("Mixture shape mismatch")

        if weights.shape[0] != self.M:
            raise ValueError("Context batch size mismatch")

        return weights, scales
        
    def radial_cdf(self,value,m):
        """radial cdf of the elliptical distribution we use the chi-squared distribution but the actual formula uses a chi distribution

        Args:
            value (torch.tensor): input for he radial_cdf of the elliptical distribution
            m (int): amount of degrees of freedom for the radial cdf

        Returns:
            torch.tensor: radial cdf of value
        """
        
        chi2     = torch.distributions.Chi2(df=torch.tensor(float(m)))
        value_j  = value[..., None]
        X        = (value_j / self.scales).square()
        chi_sq_F = chi2.cdf(X)
        return (
            self.weights * chi_sq_F
        ).sum(dim=-1)
    
    def radial_pdf(self,value,m):
        """radial pdf of the elliptical distribution we use the chi-squared distribution but the actual formula uses a chi distribution

        Args:
            value (torch.tensor): input for he radial_pdf of the elliptical distribution
            m (int): amount of degrees of freedom for the radial pdf

        Returns:
            torch.tensor: radial cdf of value
        """
        chi2       = torch.distributions.Chi2(df=torch.tensor(float(m)))
        value_j    = value[..., None]
        X          = (value_j / self.scales).square()
        chi_sq_pdf = torch.exp(chi2.log_prob(X))
        component_pdf = (
            chi_sq_pdf
            * 2.0
            * value_j
            / self.scales.square()
        )
        return (self.weights * component_pdf ).sum(dim=-1)
    
    def block_slice(self, i):
        """returns the slice that describes the subblock for the marginal transport blocks

        Args:
            i (int): i'th marginal

        Returns:
            slice: subblock slice
        """
        start = sum(self.blocksizes[:i])
        stop  = start + self.blocksizes[i]
        return slice(start, stop)
    
    def marginal_log_pdf(self, value, i):
        """
            the log pdf of the i'th marginal density 

        Args:
            value (torch.tensor): S x M x d_i tensor to be evaluated
            i (int): indicating the i'th marginal density

        Returns:
            torch.tensor: log pdf
        """
        weights, scales = self._batched_mixture_parameters()
        idx = self.block_slice(i)
        Omega_i = self.Omega_matrix[..., idx, idx]

        # [J, d, d]
        cov = (
            scales[:, None, None].square()
            * Omega_i[None, :, :]
        )
        
        # total log probs
        
        mvn = torch.distributions.MultivariateNormal(
            loc=torch.zeros(
                weights.shape[-1], 
                value.shape[-1],
                device=value.device,
                dtype=value.dtype,
            ),
            covariance_matrix=cov,
        )
        
        # mvn.batch_shape
        # # [J]
        # mvn.event_shape
        # # [d_i]
        
        component_log_probs = mvn.log_prob(
            value.unsqueeze(-2) # [S, 1, d_i]
        )
        
        #add weights for the correct log exp's
        # sum over batches -> sum over mixture components
        return torch.logsumexp(
            torch.log(weights) + component_log_probs,
            dim=-1,
        )
    
    def log_prob_elliptical(self,value):
        """
        log prob part of the elliptical copula. consisting of marginal + elliptical distribution part

        Args:
            value (torch.tensor): value to be evalutated

        Returns:
            torch.tensor: log prob
        """
        # value: [S, M, D]
        value_blocks = torch.split(
            value,
            self.blocksizes,
            dim=-1,
        )
        y_blocks = [
            self.elliptical_transport(value_i, i)
            for i, value_i in enumerate(value_blocks)
        ]
        y = torch.cat(y_blocks, dim=-1)
        log_joint = self.basis_elliptical_distribution.log_prob(y)
        log_marginals = 0.0

        for i, y_i in enumerate(y_blocks):
            log_marginals = (
                log_marginals
                + self.marginal_log_pdf(y_i, i)
            )
            
        components = {
            "log_joint" : log_joint,
            "log_marginals" : log_marginals,
            "log_copula" : log_joint - log_marginals,
        }
        return components
    
    def log_prob_NF(self, value):
        """log probability part generated by the normalizing flow,

        Args:
            value (torch.tensor): value to be evalutated

        Returns:
            torch.tensor: log prob
        """
        
        # value: [S, M, D]
        value_blocks = torch.split(
            value,
            self.blocksizes,
            dim=-1,
        )

        log_probs = [
            self.marginals[i].log_prob(value_i)
            for i, value_i in enumerate(value_blocks)
        ]

        # each log_prob should be [S, M]
        return torch.stack(
            log_probs,
            dim=-1,
        ).sum(dim=-1)
    
    def log_prob_components(self, value: torch.Tensor) -> torch.Tensor:
        if self.learn_mixture_params:
            self._refresh_elliptical_state()
            self._refresh_marginals()
            
        components = self.log_prob_elliptical(value) 
        components['log_prob_NF'] = self.log_prob_NF(value)
        components['log_prob_total'] = components["log_copula"] + components['log_prob_NF']
        return components
    
    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        components = self.log_prob_components(value)
        return components['log_prob_total']
    
    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            return self.rsample(sample_shape)#[sample_shape, self.M, self.D]
    
    def rsample(self, sample_shape=torch.Size()):
        if self.learn_mixture_params:
            self._refresh_elliptical_state()
            self._refresh_marginals()
            
        
        Y = self.basis_elliptical_distribution.sample(sample_shape)
        # Split full elliptical sample into marginal blocks
        Y_blocks = torch.split(
            Y,
            self.blocksizes,
            dim=-1,
        )

        samples = []

        for i, Y_i in enumerate(Y_blocks):

            # Elliptical marginal -> standard Gaussian latent
            z_i = self.inverse_elliptical_transport(
                Y_i,
                i,
            )
            
            # Standard Gaussian latent -> NF marginal
            x_i = self.marginals[i].transform.inv(z_i)  #transform

            samples.append(x_i)

        return torch.cat(
            samples,
            dim=-1,
        )
    
    def sample_and_log_prob(self,N):
        samples = self.sample(sample_shape=N)
        log_q   = self.log_prob(samples)
        return samples,log_q    
