
import zuko
import matplotlib.pyplot as plt
import torch.optim as optim
from VectorCopulaFlow import VectorCopulaFlow
from VectorCopulaFlow_V2 import VectorCopulaFlow_V2
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torch.distributions as dist
import torch.nn as nn
from torch.distributions import constraints
import numpy as np
import corner
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam,ClippedAdam
from pyro.optim import MultiStepLR
import os
import math
import torch.nn.functional as F
from tqdm import trange

class IdentityTransform:
    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True
    sign = +1

    def __call__(self, z):
        return z

    def inv(self, x):
        return x

    def log_abs_det_jacobian(self, x, y):
        # Identity map has Jacobian determinant 1.
        # Therefore log |det J| = 0.
        return torch.zeros(
            x.shape[:-1],
            device=x.device,
            dtype=x.dtype,
        )
class IdentityFlowDistribution:
    arg_constraints = {}
    support = constraints.real_vector
    has_rsample = True
    def __init__(self, features: int, device=None, dtype=None):
        self.features = features
        self.base = dist.Independent(
            dist.Normal(
                torch.zeros(features, device=device, dtype=dtype),
                torch.ones(features, device=device, dtype=dtype),
            ),
            reinterpreted_batch_ndims=1,
        )
        self.event_shape = torch.Size([features])
        self.transform = IdentityTransform()
    
    def log_prob(self, x):
        z = self.transform.inv(x)
        log_abs_det = self.transform.log_abs_det_jacobian(z, x)
        return self.base.log_prob(z) - log_abs_det

    def sample(self, sample_shape=torch.Size()):
        if isinstance(sample_shape, int):
            sample_shape = torch.Size([sample_shape])

        z = self.base.sample(sample_shape)
        return self.transform(z)

    def rsample(self, sample_shape=torch.Size()):
        if isinstance(sample_shape, int):
            sample_shape = torch.Size([sample_shape])

        z = self.base.rsample(sample_shape)
        return self.transform(z)
class IdentityFlow(nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.features = features

    def forward(self, context=None):
        device = None
        dtype = None

        # Try to infer device/dtype from context if given
        if isinstance(context, torch.Tensor):
            device = context.device
            dtype = context.dtype

        return IdentityFlowDistribution(
            features=self.features,
            device=device,
            dtype=dtype,
        )

def known_cov_for_target_posterior_corr(R, N=10_000, priorVar=2.0):
    """
    Construct knownCov such that Sigma_post == R.

    R must be a positive definite correlation matrix.
    """
    d = R.shape[0]
    device = R.device
    dtype = R.dtype

    I = torch.eye(d, device=device, dtype=dtype)
    prior_precision = 1.0 / priorVar**2

    A = torch.linalg.inv(R) - prior_precision * I

    # Must be positive definite, otherwise knownCov is invalid
    eigvals = torch.linalg.eigvalsh(A)
    if torch.any(eigvals <= 0):
        raise ValueError(
            f"Invalid target R for priorVar={priorVar}. "
            f"Need inv(R) - prior_precision * I positive definite. "
            f"Smallest eigenvalue: {eigvals.min().item()}"
        )

    knownCov = N * torch.linalg.inv(A)
    return knownCov

def cosine_cycle(step, cycle_length, min_factor=0.05, max_factor=1.0):
    """
    Periodic cosine multiplier.

    Returns a factor between min_factor and max_factor.
    max -> min -> max
    """
    phase = (step % cycle_length) / cycle_length

    # 1 -> 0 -> 1 over one full cycle
    cosine = 0.5 * (1.0 + math.cos(2.0 * math.pi * phase))

    return min_factor + (max_factor - min_factor) * cosine

def shifted_cosine_cycle(step, cycle_length, min_factor=0.05, max_factor=1.0):
    """
    Periodic cosine multiplier.

    Returns a factor between min_factor and max_factor.
    min -> max -> min
    """
    phase = (step % cycle_length) / cycle_length

    # 0 -> 1 -> 0 over one full cycle
    wave = 0.5 * (1.0 - math.cos(2.0 * math.pi * phase))
    return min_factor + (max_factor - min_factor) * wave

def optim_args(param_name):
    if param_name == "B":
        lr = 5e-2
    elif param_name == "z":
        lr = 5e-2
    else:
        lr = 1e-3

    return {
        "lr": lr,
        "clip_norm": 10.0,
        "lrd": 1.0,   # no internal monotone decay
    }

def get_lr_for_param(param_name, step):
    if param_name == "B":
        base_lr = 1e-3
        factor = cosine_cycle(
            step,
            cycle_length=500,
            min_factor=0.05,
            max_factor=1.0,
        )

    elif param_name == "z":
        base_lr = 1e-3
        factor = cosine_cycle(
            step,
            cycle_length=500,
            min_factor=0.05,
            max_factor=1.0,
        )

    else:
        base_lr = 1e-3
        factor = shifted_cosine_cycle(
            step,
            cycle_length=500,
            min_factor=0.05,
            max_factor=1.0,
        )

    return base_lr * factor
class pyroImplementation:
    def __init__(self,trainData,knownCov):
        self.priorVar  = 2.0

        flows,B,z      = getInitModelParams()
        self.B_init    = B.detach().clone().to(trainData.device, trainData.dtype)
        self.z_init    = z.detach().clone().to(trainData.device, trainData.dtype)
        self.flows     = nn.ModuleList(flows)
        self.trainData = trainData
        self.N, self.M, self.D = trainData.shape  # [N,M,D]
        
        # d = 4
        # N = 2000
        # priorVar = 2.0
        # rho = 0.5

        # R = torch.full((d, d), rho)
        # R.fill_diagonal_(1.0)

        # knownCov = known_cov_for_target_posterior_corr(
        #     R,
        #     N=N,
        #     priorVar=priorVar,
        # )
        
        self.knownCov = knownCov.expand(1,self.D,self.D)
        self.loss   = Trace_ELBO(num_particles=5) #

        #optimizer args
        self.optimizer = ClippedAdam(optim_args) 
        
    def model(self,data):
        ### this defines your unnormalized posterior
        ### priors
        theta_prior = dist.MultivariateNormal(
                loc=torch.zeros(
                    self.M,
                    self.D,
                    device=data.device,
                    dtype=data.dtype,
                ),
                covariance_matrix=(self.priorVar ** 2)
                * torch.eye(
                    self.D,
                    device=data.device,
                    dtype=data.dtype,
                ),
            )
        theta = pyro.sample("theta", theta_prior.to_event(1)) #[M,D]
        
        ### likelihood
        likl  = dist.MultivariateNormal(
                loc=theta,
                covariance_matrix=self.knownCov,
            )
        with pyro.plate("data", data.shape[0], dim=-2):
            pyro.sample("obs", likl, obs=data) #[S,M,D]
    
    def guide(self,data):
        pyro.module("flow_1", self.flows[0])
        pyro.module("flow_2", self.flows[1])
        B     = pyro.param("B", self.B_init) #B = to existing parameter in param store or if it is not present, self.B_init
        z     = pyro.param("z", self.z_init)
            
        q = VectorCopulaFlow_V2(
            flows                = self.flows,
            B                    = B,
            z                    = z,
        )
        pyro.sample("theta",  q.to_event(1))
    
    def train(self,n_steps):
        loss_list   = []
        svi         = SVI(self.model, self.guide,  self.optimizer, loss=self.loss)
        component_history = {
            "step": [],
            "q_total": [],
            "q_marginal_total": [],
            "q_copula_total": [],
            "q_log_det": [],
            "q_copula_quad": [],
        }

        for i in range(len(self.flows)):
            component_history[f"q_marginal_{i}"] = []
            
        #component logging, for debugging
        diag_every = 1
        n_diag_samples = 1025
        n_plots = 1
        for step in tqdm(range(n_steps)):
            loss = svi.step(self.trainData)
            
            self.update_learning_rates(step + 1)
            
            loss_list.append(loss)
            
            if step % diag_every == 0:
                diagnostics = self.compute_q_log_prob_diagnostics(
                    n_diag_samples=n_diag_samples
                )

                component_history["step"].append(step)

                for key, value in diagnostics.items():
                    component_history[key].append(value)   
            if step % (n_steps//n_plots) == 0:
                q = self.get_trained_variational_model(detach=False)
                plot_each_marginal_flow_output(q,suffix= f'_{step}')
                    

        return loss_list, component_history
    
    def get_trained_variational_model(self, detach=True):
        B = pyro.param("B")
        z = pyro.param("z")

        if detach:
            B = B.detach().clone()
            z = z.detach().clone()

        q = VectorCopulaFlow_V2(
            flows=self.flows,
            B=B,
            z=z,
        )
        return q
    
    def samplePosteriorTarget(self,N):
        N,_, d = self.trainData.shape
        Sigma0 = (self.priorVar ** 2) * torch.eye(d)
        Sigma0_inv = torch.linalg.inv(Sigma0)
        cov_inv = torch.linalg.inv(self.knownCov)

        Sigma_post = torch.linalg.inv(Sigma0_inv + N * cov_inv)
        mu_post = Sigma_post @ cov_inv @ self.trainData.squeeze(1).sum(dim=0)
        dist = torch.distributions.MultivariateNormal(mu_post, covariance_matrix=Sigma_post)
        return dist.sample((N,)),Sigma_post,mu_post
    
    def compute_q_log_prob_diagnostics(self, n_diag_samples: torch.Tensor = 512):
        """
        Compute diagnostic components of log q(value).
        value: torch.Tensor -> 
        
        Returns:
            dict[str, float]
        """

        q = self.get_trained_variational_model(detach=False)

        with torch.no_grad():
            theta_diag = q.rsample((n_diag_samples,))
            components = q.log_prob_components(theta_diag)

            diagnostics = {
                "q_total": components["log_prob_total"].mean().detach().cpu().item(),
                "q_marginal_total": components["logp_marg_total"].mean().detach().cpu().item(),
                "q_copula_total": components["log_copula_total"].mean().detach().cpu().item(),
                "q_log_det": components["log_det_term"].mean().detach().cpu().item(),
                "q_copula_quad": components["log_copula_quad"].mean().detach().cpu().item(),
            }

            for i, logp_i in enumerate(components["logp_marginals"]):
                diagnostics[f"q_marginal_{i}"] = logp_i.mean().detach().cpu().item()

        return diagnostics

    def update_learning_rates(self, step):
        param_store = pyro.get_param_store()

        for param, torch_optimizer in self.optimizer.optim_objs.items():
            param_name = param_store._param_to_name.get(param, None)

            if param_name is None:
                continue

            lr = get_lr_for_param(param_name, step)

            for group in torch_optimizer.param_groups:
                group["lr"] = lr


#ML code + general functions
def split_X_into_blocks(X, block_sizes):
    """
    Split X along the final dimension.

    Accepts X with shape:
        [N, D]
        [N, 1, D]
        [N, M, D]

    Returns:
        list of tensors, each with shape [N*, block_size]
    """

    X = X.detach()

    if X.ndim == 3:
        # For your common case [N, 1, D], this becomes [N, D].
        # For [N, M, D], this becomes [N*M, D].
        N, M, D = X.shape
        X = X.reshape(N * M, D)

    elif X.ndim == 2:
        pass

    else:
        raise ValueError(f"Expected X shape [N, D] or [N, M, D], got {tuple(X.shape)}")

    if sum(block_sizes) != X.shape[-1]:
        raise ValueError(
            f"sum(block_sizes)={sum(block_sizes)} does not match X.shape[-1]={X.shape[-1]}"
        )

    return list(torch.split(X, block_sizes, dim=-1))

def compute_log_prob_diagnostics_from_model(
    model,
    value=None,
    n_diag_samples=512,
    use_samples=False,
):
    """
    Compute diagnostic components of log_prob for a VectorCopulaFlow_V2 model.

    Parameters
    ----------
    model:
        VectorCopulaFlow_V2 instance.

    value:
        Tensor used for diagnostics. Usually training data X.
        Required if use_samples=False.

    n_diag_samples:
        Number of samples to draw if use_samples=True.

    use_samples:
        If False, compute diagnostics on provided value.
        If True, sample from model and compute diagnostics on model samples.

    Returns
    -------
    diagnostics : dict[str, float]
    """

    with torch.no_grad():
        if use_samples:
            value = model.rsample((n_diag_samples,))
        else:
            if value is None:
                raise ValueError("value must be provided when use_samples=False")

        components = model.log_prob_components(value)

        diagnostics = {
            "q_total": components["log_prob_total"].mean().detach().cpu().item(),
            "q_marginal_total": components["logp_marg_total"].mean().detach().cpu().item(),
            "q_copula_total": components["log_copula_total"].mean().detach().cpu().item(),
            "q_log_det": components["log_det_term"].mean().detach().cpu().item(),
            "q_copula_quad": components["log_copula_quad"].mean().detach().cpu().item(),
        }

        for i, logp_i in enumerate(components["logp_marginals"]):
            diagnostics[f"q_marginal_{i}"] = logp_i.mean().detach().cpu().item()

    return diagnostics

def count_trainable_params(model):
    flow_params = [
        p
        for flow in model.flows
        for p in flow.parameters()
        if p.requires_grad
    ]
    extra_params = [model.B, model.z]
    extra_params = [
        p for p in extra_params
        if isinstance(p, torch.Tensor) and p.requires_grad
    ]
    total = sum(p.numel() for p in flow_params + extra_params)
    return total

def getInitModelParams(TrainIdentity = False):
    flow_1 = zuko.flows.NSF(
        features=2,
        transforms=6,
        context=0,
        hidden_features=(8, 8),
        bins=32,
        randperm=True,
    )
    
    flow_2 = zuko.flows.NSF(
        features=2,
        transforms=6,
        context=0,
        hidden_features=(8, 8),
        bins=32,
        randperm=False,
    )
    
    # TrainIdentity = True
    if TrainIdentity:
        history = train_flows_to_identity(
            flows=[flow_1, flow_2],
            steps=5000,
            batch_size=4096,
            lr=5e-3,
            device="cpu",
            sample_scale=1.0,
        )
        check_identity(flow_1)
        
        flow_1.load_state_dict(torch.load("/root/phd/GW_seperation_analysis/vector_copula_vi_v2/vector_copula_vi/Data/flow_1_identity.pt"))
    # flow_1 = IdentityFlow(features=2)
    # flow_2 = IdentityFlow(features=2)
    flow_2.load_state_dict(flow_1.state_dict())  #break symmetry
    flows = [flow_1,flow_2]
    B     = torch.nn.Parameter(torch.randn(1, 4, 3) * 0.5)
    z     = torch.nn.Parameter(torch.tensor([1.0]))
    return flows,B,z

def buildModel():
    flows,B,z = getInitModelParams()
    model = VectorCopulaFlow_V2(
        flows = flows,       
        B     = B,
        z     = z
    )
    
    print(f"amount of trainable params: {count_trainable_params(model)}")
    return model

@torch.no_grad()
def check_identity(flow, n=10_000, device="cpu"):
    x = torch.randn(n, 2, device=device)

    dist = flow()
    transform = dist.transform

    y = transform(x)
    log_det = transform.log_abs_det_jacobian(x, y)

    print("MSE(T(x), x):", F.mse_loss(y, x).item())
    print("mean |T(x)-x|:", (y - x).abs().mean().item())
    print("max  |T(x)-x|:", (y - x).abs().max().item())
    print("mean log det:", log_det.mean().item())
    print("std  log det:", log_det.std().item())
    
def identity_loss_for_flow(flow, x):
    """
    Train the Zuko flow transform to behave like identity:
        T(x) ≈ x
        log |det J_T(x)| ≈ 0
    """

    # For unconditional flows with context=0, flow() should instantiate the distribution.
    dist = flow()

    # Zuko distributions expose the underlying transform.
    transform = dist.transform

    y = transform(x)

    # log_abs_det_jacobian usually returns shape [batch]
    log_det = transform.log_abs_det_jacobian(x, y)

    reconstruction_loss = F.mse_loss(y, x)
    volume_loss = (log_det ** 2).mean()

    return reconstruction_loss + 0.01 * volume_loss, {
        "reconstruction": reconstruction_loss.detach(),
        "volume": volume_loss.detach(),
    }

def train_flows_to_identity(
    flows,
    steps=10_000,
    batch_size=4096,
    lr=1e-3,
    device="cpu",
    dtype=torch.float32,
    sample_scale=1.0,
):
    """
    flows: list of Zuko flow modules, e.g. [flow_1, flow_2]
    """

    flow = flows[0]
    flow.to(device=device, dtype=dtype)

    params = [
        p
        for p in flow.parameters()
        if p.requires_grad
    ]

    optimizer = torch.optim.Adam(params, lr=lr)

    history = {
        "loss": [],
        "reconstruction": [],
        "volume": [],
    }

    for step in trange(steps):
        x = sample_scale * torch.randn(
            batch_size, 2,
            device=device,
            dtype=dtype,
        )

        optimizer.zero_grad()

        total_loss = 0.0
        total_reconstruction = 0.0
        total_volume = 0.0

        loss, components = identity_loss_for_flow(flow, x)
        total_loss = total_loss + loss
        total_reconstruction = total_reconstruction + components["reconstruction"]
        total_volume = total_volume + components["volume"]

        total_loss.backward()
        optimizer.step()

        history["loss"].append(total_loss.item())
        history["reconstruction"].append(total_reconstruction.item())
        history["volume"].append(total_volume.item())
    torch.save(flow.state_dict(), "/root/phd/GW_seperation_analysis/vector_copula_vi_v2/vector_copula_vi/Data/flow_1_identity.pt")
    return history

#generate flexible learning rates for ML optimization
def update_learning_rates_pytorch(optimizer, get_lr_for_group, step):
    for group in optimizer.param_groups:
        group_name = group.get("name", None)

        if group_name is None:
            raise ValueError("Every optimizer param_group should have a 'name' field.")

        group["lr"] = get_lr_for_group(group_name, step)

def pretrain_marginal_flows(
    flows,
    X,
    block_sizes=None,
    n_epochs=1000,
    batch_size=512,
    lr=1e-3,
    clip_norm=10.0,
    shuffle=True,
):
    """
    Pretrain each marginal Zuko flow by maximum likelihood on its corresponding data block.

    Parameters
    ----------
    flows:
        list or nn.ModuleList of Zuko flow modules.

    X:
        Training data, shape [N, D] or [N, M, D].

    block_sizes:
        List of event dimensions per marginal flow.
        For two 2D marginal flows and 4D data: [2, 2].

    n_epochs:
        Number of pretraining epochs.

    batch_size:
        Minibatch size.

    lr:
        Learning rate for marginal-flow pretraining.

    clip_norm:
        Gradient clipping norm.

    Returns
    -------
    history:
        dict with one loss curve per marginal flow and a total loss curve.
    """

    if block_sizes is None:
        block_sizes = [flow().event_shape[0] for flow in flows]

    device = next(flows[0].parameters()).device
    dtype = next(flows[0].parameters()).dtype

    X = X.to(device=device, dtype=dtype)
    X_blocks = split_X_into_blocks(X, block_sizes)

    # Sanity check: each block dimension should match each flow dimension.
    for i, (flow, X_block) in enumerate(zip(flows, X_blocks)):
        flow_dim = flow().event_shape[0]
        if X_block.shape[-1] != flow_dim:
            raise ValueError(
                f"Flow {i} has event dimension {flow_dim}, "
                f"but its data block has dimension {X_block.shape[-1]}"
            )

    params = [
        p
        for flow in flows
        for p in flow.parameters()
        if p.requires_grad
    ]

    optimizer = torch.optim.Adam(params, lr=lr)

    datasets = [
        TensorDataset(X_block)
        for X_block in X_blocks
    ]

    dataloaders = [
        DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=False,
        )
        for dataset in datasets
    ]

    history = {
        "total": [],
    }

    for i in range(len(flows)):
        history[f"flow_{i}"] = []

    for epoch in tqdm(range(n_epochs)):
        epoch_total_loss = 0.0
        epoch_flow_losses = [0.0 for _ in flows]
        n_seen_total = 0
        n_seen_flow = [0 for _ in flows]

        # This assumes all blocks have the same number of rows, which they should.
        for batches in zip(*dataloaders):
            optimizer.zero_grad(set_to_none=True)

            total_loss = 0.0

            for i, (flow, batch_tuple) in enumerate(zip(flows, batches)):
                X_batch = batch_tuple[0]

                flow_dist = flow()
                log_prob = flow_dist.log_prob(X_batch)

                loss_i = -log_prob.mean()
                total_loss = total_loss + loss_i

                batch_n = X_batch.shape[0]
                epoch_flow_losses[i] += loss_i.detach().cpu().item() * batch_n
                n_seen_flow[i] += batch_n

            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                params,
                max_norm=clip_norm,
            )

            optimizer.step()

            batch_n = batches[0][0].shape[0]
            epoch_total_loss += total_loss.detach().cpu().item() * batch_n
            n_seen_total += batch_n

        history["total"].append(epoch_total_loss / n_seen_total)

        for i in range(len(flows)):
            history[f"flow_{i}"].append(epoch_flow_losses[i] / n_seen_flow[i])

    return history

def train(model_to_train,num_epochs,
        X,
        batch_size = 50000, 
        # lr_flows=1e-3,
        # lr_B=5e-2,
        # lr_z=5e-2
    ):
    
    device = model_to_train.B.device
    dtype = model_to_train.B.dtype

    X = X.to(device=device, dtype=dtype)

    # dataset = TensorDataset(X)
    # dataloader = DataLoader(
    #     dataset,
    #     batch_size = batch_size, #2048
    #     shuffle    = False,  #True
    #     drop_last  = False,
    # )
    flow_params = [
        param
        for flow in model_to_train.flows
        for param in flow.parameters()
    ]
    optimizer_args =    [
            {
                "params": flow_params,
                "name": "flows",
            },
            {
                "params": [model_to_train.B],
                "name": "B",
            },
            {
                "params": [model_to_train.z],
                "name": "z",
            },
        ]
    optimizer = torch.optim.Adam(optimizer_args) 
    # optimizer = optim.LBFGS(flow_params + [model_to_train.B, model_to_train.z], lr=1.0,history_size=2000, max_iter=20, line_search_fn="strong_wolfe")
    # gamma = 0.2 ** (1.0 / num_epochs)

    # scheduler = torch.optim.lr_scheduler.ExponentialLR(
    #     optimizer,
    #     gamma=gamma,
    # )
    
    # batch_losses = []
    losses = []
    
    #decompose logProbe
    component_history = {
        "q_total": [],
        "q_marginal_total": [],
        "q_copula_total": [],
        "q_log_det": [],
        "q_copula_quad": [],
    }
    diagnostic_every = 1
    
    for epoch in tqdm(range(num_epochs)):
        # make learning rates change periodically
        update_learning_rates_pytorch(
            optimizer=optimizer,
            get_lr_for_group=get_lr_for_param,
            step=epoch,
        )
        # epoch_loss = 0.0
        # n_seen = 0

        # for (X_batch,) in dataloader:
        optimizer.zero_grad()


        log_prob = model_to_train.log_prob(X)                  # scalar
        loss = -log_prob.mean()
        loss.backward()
        optimizer.step()
        # def closure():
        #     optimizer.zero_grad()

        #     log_prob = model_to_train.log_prob(X_batch)
        #     loss = -log_prob.mean()

        #     loss.backward()

        #     return loss

        # loss = optimizer.step(closure)
        loss_value = loss.detach().cpu().item()

            # Store loss for this specific minibatch
            # batch_losses.append(loss_value)
            # batch_n = X_batch.shape[0]
            # epoch_loss += loss_value* batch_n
            # n_seen += batch_n
            
        # scheduler.step()
        mean_epoch_loss = loss_value# epoch_loss / n_seen
        losses.append(mean_epoch_loss)
        if epoch % diagnostic_every == 0:
            diagnostics = compute_log_prob_diagnostics_from_model(
                model=model_to_train,
                value=X,
                use_samples=False,
            )

            for key, value_diag in diagnostics.items():
                if key not in component_history:
                    component_history[key] = []
                component_history[key].append(value_diag)
    return losses,component_history#, batch_losses, dataloader

def clone_tensor(x):
    return x.detach().cpu().clone()

def get_model_tensors(model, prefix=""):
    tensors = {}

    # Direct tensor/parameter attributes
    if hasattr(model, "B"):
        tensors[f"{prefix}B"] = clone_tensor(model.B)

    if hasattr(model, "z"):
        tensors[f"{prefix}z"] = clone_tensor(model.z)

    # New structure: list / ModuleList of flows
    if hasattr(model, "flows"):
        for i, flow in enumerate(model.flows):
            for name, param in flow.named_parameters():
                tensors[f"{prefix}flows.{i}.{name}"] = clone_tensor(param)
    return tensors

def compare_models_manual(model_a, model_b, name_a="model_a", name_b="model_b", print_tensors=False):
    tensors_a = get_model_tensors(model_a)
    tensors_b = get_model_tensors(model_b)

    keys_a = set(tensors_a.keys())
    keys_b = set(tensors_b.keys())

    only_a = sorted(keys_a - keys_b)
    only_b = sorted(keys_b - keys_a)
    shared = sorted(keys_a & keys_b)

    if only_a:
        print(f"\nParameters only in {name_a}:")
        for name in only_a:
            print(f"  {name}")

    if only_b:
        print(f"\nParameters only in {name_b}:")
        for name in only_b:
            print(f"  {name}")

    for name in shared:
        tensor_a = tensors_a[name]
        tensor_b = tensors_b[name]

        print(f"\n{name}")

        if tensor_a.shape != tensor_b.shape:
            print(f"  shape mismatch:")
            print(f"    {name_a}: {tuple(tensor_a.shape)}")
            print(f"    {name_b}: {tuple(tensor_b.shape)}")
            continue

        diff = tensor_b - tensor_a

        print(f"  max abs diff : {diff.abs().max().item():.6e}")
        print(f"  mean abs diff: {diff.abs().mean().item():.6e}")
        print(f"  L2 diff      : {torch.norm(diff).item():.6e}")

        norm_a = torch.norm(tensor_a).item()
        if norm_a > 0:
            print(f"  relative L2  : {(torch.norm(diff) / torch.norm(tensor_a)).item():.6e}")
        else:
            print("  relative L2  : undefined, first tensor has zero norm")

        if print_tensors:
            print(f"  {name_a}:")
            print(tensor_a)
            print(f"  {name_b}:")
            print(tensor_b)
            print("  diff:")
            print(diff)

def SampleMLGaussianModel(N):
    covTrue = 1000*torch.tensor([ 
        [1.0, 0.5, 0.1, 0.3],
        [0.5, 1.0, 0.2, 0.05],
        [0.1, 0.2, 1.0, 0.45],
        [0.3, 0.05, 0.45, 1.0],
    ]) #[B,B]
    # loc = torch.tensor([0.,0.,0.,0.])
    
    loc = torch.tensor([1.0,1.0,1.0,1.0])
    # covTrue = torch.tensor([[0.4293, 0.1971, 0.0295, 0.1165],
    #         [0.1971, 0.4317, 0.0766, 0.0094],
    #         [0.0295, 0.0766, 0.4336, 0.1769],
    #         [0.1165, 0.0094, 0.1769, 0.4317]])
    
    # loc = torch.tensor([[-0.0117, -0.0009, -0.0024,  0.0151]])
    # covTrue = torch.tensor([[[1.0000, 0.5000, 0.5000, 0.5000],
    #      [0.5000, 1.0000, 0.5000, 0.5000],
    #      [0.5000, 0.5000, 1.0000, 0.5000],
    #      [0.5000, 0.5000, 0.5000, 1.0000]]])
    dist = torch.distributions.MultivariateNormal(loc=loc
                                                    , covariance_matrix=covTrue)
    
    return dist.sample(N),covTrue,loc

def empirical_covariance(model, n_samples=10_000):
    """
    Samples from model and computes empirical covariance.

    Returns:
        samples: [n_samples, M, D]
        mean:    [M, D]
        cov:     [M, D, D]
    """
    with torch.no_grad():
        X = model.sample((n_samples,))  # [S, M, D]

    S, M, D = X.shape

    mean = X.mean(dim=0)                # [M, D]

    X_centered = X - mean[None, :, :]   # [S, M, D]

    # cov[m] = X_centered[:, m, :].T @ X_centered[:, m, :] / (S - 1)
    cov = torch.einsum(
        "smd,sme->mde",
        X_centered,
        X_centered,
    ) / (S - 1)                         # [M, D, D]

    return X, mean, cov


#plots
def corner_plot_tensor(
    X: torch.Tensor,
    suffix: str
):
    """
    Make a corner plot from a torch.Tensor.

    Parameters
    ----------
    X : torch.Tensor
        Tensor containing samples. Expected shapes include:
            [N, D]
            [N, 1, D]
            [N, M, D]

        If X has shape [N, 1, D], the singleton M dimension is removed.
        If X has shape [N, M, D] with M > 1, it is flattened to [N*M, D].

    labels : list[str], optional
        Axis labels. If None, labels are generated as x_1, ..., x_D.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The corner plot figure.
    """

    X = X.detach().cpu()

    if X.ndim == 1:
        X = X[:, None]  # [N] -> [N, 1]

    elif X.ndim == 2:
        pass            # [N, D]

    elif X.ndim == 3:
        # [N, M, D] -> [N*M, D]
        N, M, D = X.shape
        X = X.reshape(N * M, D)

    else:
        raise ValueError(
            f"Expected tensor with shape [N], [N, D], or [N, M, D], "
            f"but got shape {tuple(X.shape)}"
        )

    X_np = X.numpy()

    if not np.isfinite(X_np).all():
        raise ValueError("Tensor contains NaN or Inf values.")

    N, D = X_np.shape

    labels = [rf"$x_{{{j+1}}}$" for j in range(D)]

    fig = corner.corner(
        X_np,
        labels=labels,
        bins=50,
        show_titles=True,
        title_fmt=".3f",
        plot_datapoints=True,
        plot_density=False,
        plot_contours=True,
        fill_contours=False,
    )

    fig.suptitle("Corner plot", fontsize=16)
    fig.tight_layout()
    filename = f"vector_copula_vi_v2/vector_copula_vi/Plots/corner_plot_individual{suffix}.png"

    fig.savefig(filename, dpi=300, bbox_inches="tight")

    return fig

def plot_losses(epoch_losses,batch_losses,dataloader,suffix = "_1"):
    # Suppose these are returned by train(...)
    # epoch_losses, batch_losses = train(model, num_epochs, X)

    batches_per_epoch = len(dataloader)

    batch_x = np.arange(len(batch_losses))

    # Put each epoch loss at the final minibatch index of that epoch
    epoch_x = (np.arange(1, len(epoch_losses) + 1) * batches_per_epoch) - 1

    plt.figure(figsize=(8, 4))

    plt.plot(batch_x, batch_losses, label="Minibatch loss", alpha=0.6, linewidth=0.5)
    plt.plot(epoch_x, epoch_losses, label="Mean epoch loss", marker="o",ms=1, linewidth=0.5,alpha=0.6)

    plt.xlabel("Optimization step / minibatch")
    plt.ylabel("Negative log likelihood")
    plt.title("Training loss")
    plt.legend()
    plt.grid(True)

    plt.savefig(f"vector_copula_vi_v2/vector_copula_vi/Plots/train_loss{suffix}.png", dpi=300, bbox_inches="tight")
    plt.close()

def plotLossesPyro(losses,suffix):
    plt.figure(figsize=(8, 4))
    plt.plot(losses, linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("Mean negative ELBO")
    plt.title("SVI training loss")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"vector_copula_vi_v2/vector_copula_vi/Plots/train_loss_pyro{suffix}.png", dpi=300, bbox_inches="tight")
    plt.close()

def make_corner_plot_two_clouds(
    X1,
    X2,
    labels=None,
    names=("Cloud 1", "Cloud 2"),
    suffix = "_1"
):
    """
    Make an overlaid corner plot for two 4D torch.Tensor data clouds.

    Parameters
    ----------
    X1, X2 : torch.Tensor
        Tensors of shape [N, 4].
    labels : list[str], optional
        Axis labels for the four dimensions.
    names : tuple[str, str]
        Names for the two data clouds.
    filename : str
        Output filename.
    """

    if labels is None:
        labels = [r"$x_1$", r"$x_2$", r"$x_3$", r"$x_4$"]

    # Move to CPU, detach from computation graph, convert to NumPy
    X1_np = X1.detach().cpu().numpy()
    X2_np = X2.detach().cpu().numpy()
    
    X1_np = X1_np.squeeze(1)
    X2_np = X2_np.squeeze(1)
    
    assert X1_np.ndim == 2 and X1_np.shape[1] == 4, "X1 must have shape [N, 4]"
    assert X2_np.ndim == 2 and X2_np.shape[1] == 4, "X2 must have shape [N, 4]"

    # Use common plotting ranges so the two clouds are comparable
    combined = np.vstack([X1_np, X2_np])
    ranges = [
        (combined[:, i].min(), combined[:, i].max())
        for i in range(4)
    ]

    fig = corner.corner(
        X1_np,
        labels=labels,
        range=ranges,
        bins=40,
        color="C0",
        hist_kwargs={"density": True, "alpha": 0.5},
        plot_datapoints=True,
        plot_density=False,
        plot_contours=True,
        fill_contours=False,
        label_kwargs={"fontsize": 12},
        show_titles=True,
        title_fmt=".3f",
    )

    corner.corner(
        X2_np,
        labels=labels,
        range=ranges,
        bins=40,
        color="C1",
        hist_kwargs={"density": True, "alpha": 0.5},
        plot_datapoints=True,
        plot_density=False,
        plot_contours=True,
        fill_contours=False,
        fig=fig,
        show_titles=False,
    )

    # Add a manual legend
    handles = [
        plt.Line2D([], [], color="C0", label=names[0]),
        plt.Line2D([], [], color="C1", label=names[1]),
    ]
    fig.legend(handles=handles, loc="upper right", fontsize=12)

    fig.suptitle("Corner plot of two 4D data clouds", fontsize=16)
    fig.tight_layout()
    filename = f"vector_copula_vi_v2/vector_copula_vi/Plots/corner_plot{suffix}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

    return fig

def movingAverageLosses(losses,suffix,window = 50):
    def moving_average(x, window=200):
        x = np.asarray(x)
        return np.convolve(x, np.ones(window) / window, mode="valid")
    filename = f"vector_copula_vi_v2/vector_copula_vi/Plots/Loss_moving_avg{suffix}.png"
    plt.figure(figsize=(8, 4))
    plt.plot(losses, alpha=0.65, linewidth=0.5, label="Raw loss")
    plt.plot(
        np.arange(window-1, len(losses)),
        moving_average(losses, window=window),
        linewidth=1.5,
        label="Moving average"
    )
    plt.xlabel("Step")
    plt.ylabel("Negative ELBO per datapoint")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

def plot_q_log_prob_components(component_history, suffix="_pyro"):
    plt.figure(figsize=(10, 5))

    plt.plot(
        component_history["q_total"],
        label="q_total",
        linewidth=1.2,
        linestyle="dashed"
    )

    plt.plot(
        component_history["q_marginal_total"],
        label="Marginal contribution",
        linewidth=1.0,
        alpha=0.8,
        linestyle="dashdot"
    )

    plt.plot(
        component_history["q_copula_total"],
        label="Copula contribution",
        linewidth=1.0,
        alpha=0.8,
        linestyle="dotted"
    )

    plt.plot(
        component_history["q_log_det"],
        label="Copula log-det term",
        linewidth=1.0,
        alpha=0.8,
        linestyle="dashdot"
    )

    plt.plot(
        component_history["q_copula_quad"],
        label="Copula quadratic term",
        linewidth=1.0,
        alpha=0.8,
        linestyle="dashdot"
    )

    for key, values in component_history.items():
        if key.startswith("q_marginal_") and key != "q_marginal_total":
            plt.plot(
                values,
                label=key,
                linewidth=0.8,
                alpha=0.8,
                linestyle="dotted"
            )

    plt.xlabel("SVI step")
    plt.ylabel("Mean log-probability contribution")
    plt.title("Variational log-probability components")
    plt.legend()
    plt.grid(True, alpha=0.3)

    filename = (
        f"vector_copula_vi_v2/vector_copula_vi/Plots/"
        f"q_log_prob_components{suffix}.png"
    )

    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

def plot_each_marginal_flow_output(
    q,
    N=10_000,
    out_dir="vector_copula_vi_v2/vector_copula_vi/Plots/Marginals",
    suffix="",
):
    """
    For each Zuko marginal flow in q.flows:

        z ~ N(0, I_d)
        x = flow.transform(z)

    Then creates a corner plot of x.

    Args:
        q:
            VectorCopulaFlow_V2 object.
        N:
            Number of base samples per marginal flow.
        out_dir:
            Directory where plots are saved.
        suffix:
            Optional suffix for filenames.

    Returns:
        list[str]: saved plot filenames.
    """

    os.makedirs(out_dir, exist_ok=True)

    filenames = []

    device = q.B.device
    dtype = q.B.dtype

    for flow_idx, flow in enumerate(q.flows):
        # Construct the Zuko distribution object.
        flow_dist = flow()

        # Input/event dimension of this marginal flow.
        d = flow_dist.event_shape[0]

        with torch.no_grad():
            # Base samples: z ~ N(0, I_d)
            z = torch.randn(
                N,
                d,
                device=device,
                dtype=dtype,
            )

            # Push through the flow: x = T(z)
            x = flow_dist.transform(z)

        x_np = x.detach().cpu().numpy()

        labels = [rf"$x_{{{j+1}}}$" for j in range(d)]

        filename = os.path.join(
            out_dir,
            f"flow_{flow_idx}_output_corner{suffix}.png",
        )

        if d == 1:
            # corner is overkill for 1D and sometimes awkward.
            plt.figure(figsize=(6, 4))
            plt.hist(x_np[:, 0], bins=80, density=True, alpha=0.75)
            plt.xlabel(labels[0])
            plt.ylabel("Density")
            plt.title(f"Output distribution of flow {flow_idx}")
            plt.grid(True, alpha=0.3)
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            plt.close()

        else:
            fig = corner.corner(
                x_np,
                labels=labels,
                bins=50,
                show_titles=True,
                title_fmt=".3f",
                plot_datapoints=True,
                plot_density=False,
                plot_contours=True,
                fill_contours=False,
            )

            fig.suptitle(
                f"Output distribution of marginal flow {flow_idx}",
                fontsize=16,
            )

            fig.tight_layout()
            fig.savefig(filename, dpi=300, bbox_inches="tight")
            plt.close(fig)

        filenames.append(filename)

    return filenames

def plot_marginal_pretraining_history(history,suffix):
    plt.figure(figsize=(8, 4))

    for key, values in history.items():
        plt.plot(values, label=key, linewidth=1.2)

    plt.xlabel("Epoch")
    plt.ylabel("Negative log likelihood")
    plt.title("Marginal flow pretraining")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    filename = (
        f"vector_copula_vi_v2/vector_copula_vi/Plots/"
        f"pretrained_marginals_loss{suffix}.png"
    )
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

def copula_weight_schedule(step, warmup_steps=1000, max_weight=5.0):
    """
    Starts with high copula emphasis and anneals back to 1. -> so that we force learn the copula
    """
    if step >= warmup_steps:
        return 1.0

    t = step / warmup_steps

    return max_weight * (1.0 - t) + 1.0 * t


#genral functions
def trainMLTruth(n_epochs):
    suffix                           = "_pyro_v7_ML"
    N_samples_plot                   = 10000

    model_to_train                   = buildModel()
    N_samples                        = 2000
    PyroData,_,_                     = SampleMLGaussianModel([N_samples])
    PyroData                         = PyroData.unsqueeze(1)
    # corner_plot_tensor(PyroData,"_pyro_V6_ML_pyroData")
    impl                             = pyroImplementation(PyroData)
    X,_,_                            = impl.samplePosteriorTarget([N_samples])
    
    # corner_plot_tensor(X,"_pyro_V6_ML_X")
    losses,_ = train(model_to_train,n_epochs,X,batch_size=N_samples,lr = 1e-3)
    
    # plot_losses(losses, batch_losses, dataloader,suffix = suffix)
    plotLossesPyro(losses,suffix)

    trainedSample = model_to_train.rsample([N_samples_plot])
    make_corner_plot_two_clouds(X,trainedSample,suffix = suffix)
    return impl,model_to_train,PyroData

if __name__ == "__main__":
    suffix = "_PYRO_V9"
    n_epochs                 = 2000  #10000
    # #pyro
    # impl,init_model,PyroData = trainMLTruth(10000)    
    # statePath                = "/root/phd/GW_seperation_analysis/vector_copula_vi_v2/vector_copula_vi/Data/vector_copula_flow_warm_start.pt"
    # torch.save(init_model.state_dict(), statePath)
    
    # print("stop")
    
    ## ML
    N                    = [5000]
    # model_base           = buildModel()
    # X                    = model_base.sample(N)
    
    ## pyro
    X,covTrue,_          = SampleMLGaussianModel(N)
    X_base               = X.unsqueeze(1)
    PyroData             = X_base
    model_to_train       = buildModel()
    # suffix = "temp_ML_fix"
    # state = torch.load(
    # statePath,
    #     map_location=model_to_train.B.device,
    # )
    # model_to_train.load_state_dict(state)
    
    N_samples_plot       = 10000
    
    #classic ML
    # history_marginals = pretrain_marginal_flows(
    #     flows=model_to_train.flows,
    #     X=X,
    #     block_sizes=[2, 2],
    #     n_epochs=50,
    #     batch_size=512,
    #     lr=1e-3,
    #     clip_norm=10.0,
    # )   
    # plot_marginal_pretraining_history(history_marginals,suffix)
    # losses,component_history = train(model_to_train,
    #                                 n_epochs,
    #                                 X,
    #                                 # lr_flows=2e-3,
    #                                 # lr_B=1e-4,
    #                                 # lr_z=1e-4
    #     )
    # plotLossesPyro(losses,suffix)
    # plot_q_log_prob_components(component_history,suffix)
    # # plot_losses(losses, batch_losses, dataloader, suffix = suffix)
    # make_corner_plot_two_clouds(model_to_train.sample([N_samples_plot]),X_base,suffix = suffix)


    #Bayesian
    impl = pyroImplementation(PyroData,covTrue)
    
    losses, component_history = impl.train(n_epochs)
    
    Trained_model = impl.get_trained_variational_model()
    Sample,cov_base,mean_base = impl.samplePosteriorTarget([N_samples_plot])
    
    make_corner_plot_two_clouds(Trained_model.sample([N_samples_plot]),Sample,suffix = suffix)
    plot_q_log_prob_components(component_history, suffix=suffix)
    movingAverageLosses(losses,suffix)
    _, mean_train, cov_train,  = empirical_covariance(Trained_model, n_samples=N_samples_plot)
    print(f'cov_base: {cov_base}')
    print(f'cov_train: {cov_train}')
    print(f'mean_base: {mean_base}')
    print(f'mean_train: {mean_train}')

    print(f'F-Norm: {torch.norm(cov_base-cov_train)}')