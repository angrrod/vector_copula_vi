
import matplotlib.pyplot as plt
import torch
import numpy as np
import corner
from dingo.core.posterior_models.vector_copula_Model import CopulaNormalizingFlowModel
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from dataclasses import dataclass
import utils
from config import ScenarioConfig
from setUpLoggerScenario import setUpLoggerScenario
from pathlib import Path
import os
import pandas as pd
from matplotlib.lines import Line2D
import torch
class IdentityContextEmbedding(torch.nn.Module):
    def __init__(self, input_dim=4, output_dim=4):
        super().__init__()

        if input_dim != output_dim:
            raise ValueError(
                f"Identity embedding requires input_dim == output_dim, "
                f"got input_dim={input_dim}, output_dim={output_dim}."
            )

        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, *context):
        x = context[0]
        return x

class GaussianMeanSBISimulator:
    """
    Simulator for the compressed-summary SBI version.

    The neural posterior is trained on pairs

        theta ~ prior
        x_bar | theta ~ Normal(theta, known_cov / n_obs)

    where x_bar is the sample mean of n_obs observations.
    """

    def __init__(
        self,
        D=4,
        n_obs=5000,
        prior_var=2.0,
        known_cov=None,
        device="cpu",
        dtype=torch.float32,
    ):
        self.D = D
        self.n_obs = n_obs
        self.prior_var = prior_var
        self.device = device
        self.dtype = dtype

        if known_cov is None:
            raise ValueError("known_cov must be provided.")

        self.known_cov = known_cov.to(device=device, dtype=dtype)

        self.prior = torch.distributions.MultivariateNormal(
            loc=torch.zeros(D, device=device, dtype=dtype),
            covariance_matrix=(prior_var ** 2)
            * torch.eye(D, device=device, dtype=dtype),
        )

    def sample_theta(self, batch_size):
        return self.prior.sample((batch_size,))

    def sample_context(self, theta):
        """
        theta: [B, D]

        returns:
            x_bar: [B, D]

        This directly simulates the sample mean of n_obs observations.
        """
        context_dist = torch.distributions.MultivariateNormal(
            loc=theta,
            covariance_matrix=self.known_cov / self.n_obs,
        )
        return context_dist.sample()

    def sample_joint(self, batch_size):
        theta = self.sample_theta(batch_size)
        context = self.sample_context(theta)
        return theta, context

    def context_from_observed_data(self, X):
        """
        X: [N, D] or [N, 1, D]

        returns:
            x_bar: [1, D]
        """
        if X.ndim == 3:
            X = X.squeeze(1)

        if X.ndim != 2:
            raise ValueError(f"Expected X shape [N, D], got {tuple(X.shape)}")

        return X.mean(dim=0, keepdim=True).to(
            device=self.device,
            dtype=self.dtype,
        )

class GaussianMeanSBIDataset(Dataset):
    def __init__(self, simulator, num_simulations):
        theta, context = simulator.sample_joint(num_simulations)

        self.theta = theta
        self.context = context

    def __len__(self):
        return self.theta.shape[0]

    def __getitem__(self, idx):
        return self.theta[idx], self.context[idx]

def train_copula_sbi(
    model,
    train_loader,
    n_epochs=200,
    lr=2e-3,
    device="cpu",
):
    model.network.train()
    
    theta_all = []
    context_all = []

    for theta, context in train_loader:
        theta_all.append(theta)
        context_all.append(context)

    theta_all = torch.cat(theta_all, dim=0).to(device)
    context_all = torch.cat(context_all, dim=0).to(device)
    
    ### LBFGS
    # optimizer = torch.optim.LBFGS(
    #     model.network.parameters(),
    #     lr=lr,
    #     max_iter=20,
    #     history_size=200,
    #     line_search_fn="strong_wolfe",
    #     tolerance_grad=1e-7,
    #     tolerance_change=1e-9,
    # )
    
    # losses = []

    # for step in tqdm(range(n_epochs)):

    #     def closure():
    #         optimizer.zero_grad(set_to_none=True)

    #         loss = model.loss(theta_all, context_all)

    #         if not torch.isfinite(loss):
    #             raise ValueError(
    #                 f"Non-finite loss at LBFGS step {step}: "
    #                 f"{loss.detach().item()}"
    #             )

    #         loss.backward()

    #         torch.nn.utils.clip_grad_norm_(
    #             model.network.parameters(),
    #             max_norm=10.0,
    #         )

    #         return loss

    #     optimizer.step(closure)

    #     # Recompute after the final LBFGS parameter update.
    #     with torch.no_grad():
    #         final_loss = model.loss(theta_all, context_all)

    #     losses.append(final_loss.detach().cpu().item())

    ### ADAM
    lr_min  = 2e-4
    gamma = (lr_min / lr) ** (1 / n_epochs)
    optimizer = torch.optim.Adam(
        model.network.parameters(),
        lr=lr,
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=gamma,
    )
    losses = []

    for epoch in tqdm(range(n_epochs)):
        epoch_loss = 0.0
        n_seen = 0

        for theta, context in train_loader:
            theta = theta.to(device)
            context = context.to(device)

            optimizer.zero_grad(set_to_none=True)

            loss = model.loss(theta, context)

            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite loss at epoch {epoch}: {loss}")

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.network.parameters(),
                max_norm=10.0,
            )

            optimizer.step()

            batch_size = theta.shape[0]
            epoch_loss += loss.detach().cpu().item() * batch_size
            n_seen += batch_size
        scheduler.step()

        losses.append(epoch_loss / n_seen)

    return losses

def make_copula_sbi_metadata():
    return {
        "dataset_settings": {
            "type": "gaussian_mean_sbi",
            "theta_dim": 4,
            "raw_context_dim": 4,
        },
        "train_settings": {
            "model": {
                "posterior_model_type": "copula_normalizing_flow",
                "posterior_kwargs": {
                    "context_dim": 4,
                    "block_dims": [2, 2],
                    "flows": {
                        "flow_1": {
                            "num_flow_steps": 4,
                            "base_transform_kwargs": {
                                "hidden_dim": 64,
                                "num_transform_blocks": 4,
                                "activation": "elu",
                                "dropout_probability": 0.0,
                                "batch_norm": True,
                                "num_bins": 16,
                                "base_transform_type": "rq-coupling",
                            },
                        },
                        "flow_2": {
                            "num_flow_steps": 4,
                            "base_transform_kwargs": {
                                "hidden_dim": 64,
                                "num_transform_blocks": 4,
                                "activation": "elu",
                                "dropout_probability": 0.0,
                                "batch_norm": True,
                                "num_bins": 16,
                                "base_transform_type": "rq-coupling",
                            },
                        },
                    },
                    "CopulaKwargs": {
                        "P": 3,
                        "hidden_dims": [64],
                        "activation": "elu",
                        "dropout": 0.0,
                        "batch_norm": False,
                    },
                },
                "embedding_kwargs": {
                    # "type": "identity",
                    # "input_dim": 4,
                    # "output_dim": 4,
                },
            }
        },
    }

def SampleMLGaussianModel(N):
    covTrue = 1000*torch.tensor([ 
        [1.0, 0.5, 0.1, 0.3],
        [0.5, 1.0, 0.2, 0.05],
        [0.1, 0.2, 1.0, 0.45],
        [0.3, 0.05, 0.45, 1.0],
    ]) #[B,B]
    loc = torch.tensor([0.,0.,0.,0.])
    
    # loc = torch.tensor([1.0,1.0,1.0,1.0])
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

def samplePosteriorTarget(priorVar,trainData,knownCov):
    if trainData.ndim == 3 and trainData.shape[1] == 1:
        trainData = trainData.squeeze(1)   # (N, 1, d) -> (N, d)
    elif trainData.ndim != 2:
        raise ValueError(f"Expected trainData shape (N, d) or (N, 1, d), got {trainData.shape}")

    N, d = trainData.shape

    Sigma0 = (priorVar ** 2) * torch.eye(d, device=trainData.device, dtype=trainData.dtype)
    Sigma0_inv = torch.linalg.inv(Sigma0)
    cov_inv = torch.linalg.inv(knownCov)

    Sigma_post = torch.linalg.inv(Sigma0_inv + N * cov_inv)
    mu_post = Sigma_post @ cov_inv @ trainData.sum(dim=0)

    dist = torch.distributions.MultivariateNormal(mu_post, covariance_matrix=Sigma_post)
    return dist.sample((N,)), Sigma_post, mu_post

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
        hist_kwargs={"alpha": 0.5},
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
        hist_kwargs={"alpha": 0.5},
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

def patch_dingo_copula_embedding_to_local_mlp():
    """
    Local test-file patch.

    Dingo's CopulaNSFFlowWrapper normally interprets embedding_kwargs as GW/SVD
    embedding kwargs. For this toy SBI test, reinterpret them as a simple MLP.
    """
    import dingo.core.nn.copulaNSF as copula_nsf

    def create_local_mlp_embedding(**embedding_kwargs):
        embedding_kwargs = dict(embedding_kwargs)

        # Remove keys that are meaningful for our toy config,
        # but unknown to Dingo's GW embedding constructor.
        embedding_kwargs.pop("V_rb_list", None)
        embedding_kwargs.pop("type",None)

        activation = embedding_kwargs.pop("activation", "elu")

        input_dim = embedding_kwargs.pop("input_dim")
        output_dim = embedding_kwargs.pop("output_dim")
        hidden_dim = embedding_kwargs.pop("hidden_dim", 64)

        return IdentityContextEmbedding(
            input_dim=input_dim,
            output_dim=output_dim,
        )

    copula_nsf.create_enet_with_projection_layer_and_dense_resnet = (
        create_local_mlp_embedding
    )

def parameter_inventory(model):
    print("\n=== REGISTERED PARAMETERS ===")

    total = 0
    by_prefix = {}

    for name, p in model.network.named_parameters():
        n = p.numel()
        total += n
        prefix = name.split(".")[0]
        by_prefix[prefix] = by_prefix.get(prefix, 0) + n
        print(f"{name:80s} {tuple(p.shape)} requires_grad={p.requires_grad}")

    print("\n=== PARAMETER COUNT BY PREFIX ===")
    for k, v in sorted(by_prefix.items()):
        print(f"{k:30s}: {v}")

    print(f"\nTOTAL REGISTERED PARAMETERS: {total}")

    return total, by_prefix

def check_flow_parameter_registration(model):
    registered_ids = {id(p) for p in model.network.parameters()}

    flow_params = []
    for i, flow in enumerate(model.network.flows):
        for name, p in flow.named_parameters():
            flow_params.append((i, name, p))

    n_flow_params = sum(p.numel() for _, _, p in flow_params)
    n_registered_flow_params = sum(
        p.numel()
        for _, _, p in flow_params
        if id(p) in registered_ids
    )

    print("\n=== FLOW PARAMETER REGISTRATION ===")
    print(f"Number of flow modules: {len(model.network.flows)}")
    print(f"Total flow parameters: {n_flow_params}")
    print(f"Registered flow parameters: {n_registered_flow_params}")

    missing = [
        (i, name, tuple(p.shape))
        for i, name, p in flow_params
        if id(p) not in registered_ids
    ]

    if missing:
        print("\nMISSING FLOW PARAMETERS:")
        for i, name, shape in missing[:30]:
            print(f"flow {i}: {name:70s} {shape}")
        if len(missing) > 30:
            print(f"... and {len(missing) - 30} more")

        raise RuntimeError(
            "The marginal flow parameters are not registered in model.network.parameters(). "
            "The optimizer will not train them."
        )

    print("All flow parameters are registered.")
    
# def test_gaussian_mean_sbi_dataset(dataset, simulator, atol_mean=0.2, rtol_cov=0.2):
#     theta = dataset.theta
#     context = dataset.context

#     print("theta shape:  ", theta.shape)
#     print("context shape:", context.shape)

#     assert theta.ndim == 2
#     assert context.ndim == 2
#     assert theta.shape == context.shape
#     assert theta.shape[1] == simulator.D

#     residual = context - theta

#     theta_mean = theta.mean(dim=0)
#     theta_cov = torch.cov(theta.T)

#     residual_mean = residual.mean(dim=0)
#     residual_cov = torch.cov(residual.T)

#     expected_theta_cov = (simulator.prior_var ** 2) * torch.eye(
#         simulator.D, device=theta.device, dtype=theta.dtype
#     )

#     expected_residual_cov = simulator.known_cov / simulator.n_obs

#     print("\ntheta mean:")
#     print(theta_mean)

#     print("\ntheta cov empirical:")
#     print(theta_cov)

#     print("\ntheta cov expected:")
#     print(expected_theta_cov)

#     print("\nresidual mean = mean(context - theta):")
#     print(residual_mean)

#     print("\nresidual cov empirical:")
#     print(residual_cov)

#     print("\nresidual cov expected = known_cov / n_obs:")
#     print(expected_residual_cov)

#     assert torch.allclose(
#         residual_mean,
#         torch.zeros_like(residual_mean),
#         atol=atol_mean,
#     )

#     assert torch.allclose(
#         residual_cov,
#         expected_residual_cov,
#         rtol=rtol_cov,
#         atol=expected_residual_cov.abs().mean() * rtol_cov,
#     )

#     print("\nDataset distribution test passed.")

def main_sbi():
    suffix         = "___DINGO_V1"
    N_samples_plot = 10000
    priorVar       = 2.0
    d              = 4
    n_epochs       = 500
    n_obs          = 2000
    X,covTrue,_    = SampleMLGaussianModel([n_obs])
    device         = "cpu"
    window          = 20
    patch_dingo_copula_embedding_to_local_mlp()
    
    assert n_obs == X.shape[0], "shape error"
    
    simulator = GaussianMeanSBISimulator(
        D         = 4,
        n_obs     = n_obs,
        prior_var = 2.0,
        device    = device,
        known_cov = covTrue
    )
    simulator

    dataset = GaussianMeanSBIDataset(
        simulator=simulator,
        num_simulations=5000,
    )

    train_loader = DataLoader(
        dataset,
        batch_size=500,
        shuffle=True,
    )

    model = CopulaNormalizingFlowModel(
        metadata=make_copula_sbi_metadata(),
        device=device,
    )
    parameter_inventory(model)
    check_flow_parameter_registration(model)

    # actual training
    losses = train_copula_sbi(
        model        = model,
        train_loader = train_loader,
        n_epochs     = n_epochs,
        lr           = 1e-2,
        device       = device,
    )
    
    #Plot
    # Observed data: X has shape [N_obs, 4] because N = [5000]
    X_base = X.unsqueeze(1)  # [N_obs, 1, D]

    # Analytic posterior target p(theta | observed X)
    Sample, cov_base, mean_base = samplePosteriorTarget(
        priorVar  = priorVar,
        trainData = X_base,
        knownCov  = covTrue,
    )

    context_obs = simulator.context_from_observed_data(X) # [1, D]

    # Your trained Dingo model.
    # Rename this to whatever variable you actually trained.
    Trained_model = model
    Trained_model.network.eval()

    with torch.no_grad():
        # Dingo returns [B, num_samples, D].
        # Here B = 1 because we condition on one observed dataset.
        dingo_samples = Trained_model.sample(
            context_obs,
            num_samples=N_samples_plot,
        )  
        # Convert to the old plotting convention [N_samples_plot, 1, D].

    # Both clouds now have shape [N_samples_plot, 1, 4]
    make_corner_plot_two_clouds(
        dingo_samples,
        Sample,
        names=("DINGO copula SBI", "Analytic posterior"),
        suffix=suffix,
    )

    movingAverageLosses(losses, suffix, window = window)

    # Empirical mean/covariance of Dingo posterior samples
    mean_train = dingo_samples.mean(dim=0)  # [1, D]

    dingo_centered = dingo_samples - mean_train[None, :, :]
    cov_train = torch.einsum(
        "smd,sme->mde",
        dingo_centered,
        dingo_centered,
    ) / (dingo_samples.shape[0] - 1)  # [1, D, D]

    print(f"cov_base: {cov_base}")
    print(f"cov_train: {cov_train}")
    print(f"mean_base: {mean_base}")
    print(f"mean_train: {mean_train}")

    print(f"F-Norm: {torch.norm(cov_base - cov_train.squeeze(0))}")

################################ ML ##############################

def _atomic_torch_save(payload: dict, output_path):
    """
    Write a checkpoint atomically.

    The temporary file is replaced only after torch.save succeeds, reducing
    the chance of leaving a corrupt checkpoint after an interruption.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    torch.save(payload, temporary_path)
    os.replace(temporary_path, output_path)

def save_copula_ml_checkpoint(
    output_path,
    *,
    model,
    metadata,
    parameter_names,
    loss_history,
    completed_epochs,
    best_loss,
    optimizer=None,
    scheduler=None,
    data_source=None,
):
    checkpoint = {
        "checkpoint_format_version": 1,
        "completed_epochs": int(completed_epochs),
        "best_loss": float(best_loss),
        "loss_history": [
            float(loss) for loss in loss_history
        ],

        # The actual fitted model.
        "model_state_dict": model.network.state_dict(),

        # Required to reconstruct exactly the same architecture.
        "metadata": metadata,

        # Required to preserve the HDF5 column ordering.
        "parameter_names": [
            str(name) for name in parameter_names
        ],

        "data_source": (
            str(data_source)
            if data_source is not None
            else None
        ),

        # These are None for a weights-only checkpoint.
        "optimizer_state_dict": (
            optimizer.state_dict()
            if optimizer is not None
            else None
        ),
        "scheduler_state_dict": (
            scheduler.state_dict()
            if scheduler is not None
            else None
        ),

        # Useful for approximately continuing shuffled minibatches.
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        ),
    }

    _atomic_torch_save(
        checkpoint,
        output_path,
    )

    print(f"Saved checkpoint: {output_path}")

def _optimizer_to_device(optimizer, device):
    """
    Move tensors stored inside an optimizer state to the target device.
    """
    device = torch.device(device)

    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)

def load_copula_ml_checkpoint(
    checkpoint_path,
    *,
    device="cpu",
    restore_optimizer=True,
):
    checkpoint_path = Path(checkpoint_path)

    # Only load checkpoints you created yourself.
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    metadata = checkpoint["metadata"]

    model = CopulaNormalizingFlowModel(
        metadata=metadata,
        device=device,
    )

    model.network.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    optimizer = None
    scheduler = None

    optimizer_state = checkpoint.get(
        "optimizer_state_dict"
    )
    scheduler_state = checkpoint.get(
        "scheduler_state_dict"
    )

    if restore_optimizer and optimizer_state is not None:
        saved_lr = optimizer_state["param_groups"][0]["lr"]

        optimizer = torch.optim.Adam(
            model.network.parameters(),
            lr=saved_lr,
        )

        optimizer.load_state_dict(
            optimizer_state
        )

        _optimizer_to_device(
            optimizer,
            device,
        )

        if scheduler_state is not None:
            saved_gamma = scheduler_state.get(
                "gamma",
                1.0,
            )

            scheduler = (
                torch.optim.lr_scheduler.ExponentialLR(
                    optimizer,
                    gamma=saved_gamma,
                )
            )

            scheduler.load_state_dict(
                scheduler_state
            )

    rng_state = checkpoint.get("torch_rng_state")

    if rng_state is not None:
        torch.set_rng_state(rng_state.cpu())

    cuda_rng_state = checkpoint.get(
        "cuda_rng_state_all"
    )

    if (
        cuda_rng_state is not None
        and torch.cuda.is_available()
    ):
        torch.cuda.set_rng_state_all(
            cuda_rng_state
        )

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(
        "Completed epochs:",
        checkpoint.get("completed_epochs", 0),
    )
    print(
        "Best stored loss:",
        checkpoint.get("best_loss"),
    )

    return model, optimizer, scheduler, checkpoint

def plot_ml_losses(
    losses,
    output_path,
    moving_average_window=10,
    skip_initial_epochs=20,
):
    losses = np.asarray(losses, dtype=float)

    if losses.ndim != 1 or losses.size == 0:
        raise ValueError(
            f"Expected a non-empty one-dimensional loss array, "
            f"got shape {losses.shape}."
        )

    if not np.isfinite(losses).all():
        raise ValueError("Loss history contains NaN or Inf.")

    epochs = np.arange(1, losses.size + 1)

    skip = int(skip_initial_epochs)
    if skip < 0 or skip >= losses.size:
        raise ValueError(
            f"skip_initial_epochs must be between 0 and "
            f"{losses.size - 1}, got {skip}."
        )

    # Apply the same trimming to every plotted diagnostic.
    plotted_losses = losses[skip:]
    plotted_epochs = epochs[skip:]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(
        plotted_epochs,
        plotted_losses,
        linewidth=1.0,
        alpha=0.65,
        label="Training NLL",
    )

    window = min(
        int(moving_average_window),
        plotted_losses.size,
    )

    if window >= 2:
        kernel = np.ones(window, dtype=float) / window

        smoothed = np.convolve(
            plotted_losses,
            kernel,
            mode="valid",
        )

        # The first moving-average value corresponds to the final epoch
        # in the first averaging window.
        smoothed_epochs = plotted_epochs[window - 1:]

        ax.plot(
            smoothed_epochs,
            smoothed,
            linewidth=2.0,
            label=f"{window}-epoch moving average",
        )

    # Minimum among the epochs actually displayed.
    best_index = int(np.argmin(plotted_losses))
    best_epoch = int(plotted_epochs[best_index])
    best_loss = float(plotted_losses[best_index])

    ax.scatter(
        best_epoch,
        best_loss,
        marker="o",
        label=f"Minimum: {best_loss:.6g} (epoch {best_epoch})",
        zorder=3,
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Negative log-likelihood per sample")
    ax.set_title(
        "Unconditional copula maximum-likelihood training"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

def print_fitted_copula_model(
    model,
    parameter_names,
    output_dir=None,
):
    """
    Print and optionally save the fitted non-amortized copula parameters.

    B and raw z are the optimized parameterization. Omega is the implied
    normalized copula matrix and is the more interpretable fitted object.
    """
    network = model.network

    if getattr(network, "conditional", True):
        raise ValueError(
            "print_fitted_copula_model() expects an unconditional copula model."
        )

    if not hasattr(network, "B") or not hasattr(network, "z"):
        raise AttributeError(
            "The unconditional network must expose trainable B and z parameters."
        )

    parameter_names = list(parameter_names)
    if len(parameter_names) != network.D:
        raise ValueError(
            f"Received {len(parameter_names)} parameter names, but model D={network.D}."
        )

    network.eval()
    with torch.no_grad():
        fitted_distribution = network.distribution()

        B = network.B.detach().cpu().squeeze(0)
        z_raw = network.z.detach().cpu().reshape(-1)
        zeta = torch.nn.functional.softplus(z_raw)
        omega = fitted_distribution.Omega().detach().cpu().squeeze(0)

    factor_names = [f"factor_{j + 1}" for j in range(B.shape[1])]
    B_df = pd.DataFrame(
        B.numpy(),
        index=parameter_names,
        columns=factor_names,
    )
    omega_df = pd.DataFrame(
        omega.numpy(),
        index=parameter_names,
        columns=parameter_names,
    )

    block_dims = list(network.block_dims)
    if sum(block_dims) != network.D:
        raise ValueError(
            f"block_dims={block_dims} do not sum to D={network.D}."
        )

    if len(block_dims) == 2:
        split = block_dims[0]
        omega_cross_df = omega_df.iloc[:split, split:]
    else:
        omega_cross_df = None

    print("\n=== FINAL FITTED NON-AMORTIZED COPULA MODEL ===")
    print(f"Event dimension D: {network.D}")
    print(f"Low-rank dimension P: {B.shape[1]}")
    print(f"Block dimensions: {block_dims}")
    print(f"Raw z: {z_raw.numpy()}")
    print(f"zeta = softplus(z): {zeta.numpy()}")

    print("\nB factor matrix:")
    print(B_df.to_string(float_format=lambda x: f"{x: .6f}"))

    print("\nImplied normalized copula matrix Omega:")
    print(omega_df.to_string(float_format=lambda x: f"{x: .6f}"))

    if omega_cross_df is not None:
        print("\nCross-block copula matrix Omega_AB:")
        print(
            omega_cross_df.to_string(
                float_format=lambda x: f"{x: .6f}"
            )
        )

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        B_path = output_dir / "fitted_copula_B.csv"
        omega_path = output_dir / "fitted_copula_Omega.csv"
        B_df.to_csv(B_path)
        omega_df.to_csv(omega_path)

        print("\nSaved fitted copula parameters:")
        print(f"B:     {B_path}")
        print(f"Omega: {omega_path}")

        if omega_cross_df is not None:
            omega_cross_path = output_dir / "fitted_copula_Omega_AB.csv"
            omega_cross_df.to_csv(omega_cross_path)
            print(f"Omega_AB: {omega_cross_path}")

    return {
        "B": B,
        "z_raw": z_raw,
        "zeta": zeta,
        "Omega": omega,
        "B_dataframe": B_df,
        "Omega_dataframe": omega_df,
        "Omega_AB_dataframe": omega_cross_df,
    }

def train_copula_ML(
    model,
    train_loader,
    n_epochs=200,
    lr=2e-3,
    device="cpu",
    checkpoint_path="postProcessing/copula_ml/copula_ml_best.pt",
    start_from_best=False,
):
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    previous_epochs = 0
    best_loss = float("inf")

    # Optionally start from the previously stored best model.
    if start_from_best:
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Cannot continue training because no checkpoint exists at:\n"
                f"{checkpoint_path}"
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )

        model.network.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )

        best_loss = float(checkpoint["best_loss"])
        previous_epochs = int(
            checkpoint.get("epoch", 0)
        )

        print("\nLoaded best copula model")
        print(f"Checkpoint:       {checkpoint_path}")
        print(f"Stored epoch:     {previous_epochs}")
        print(f"Stored best NLL:  {best_loss:.6f}")

    else:
        print("\nStarting copula training from current initialization.")
    model.network.train()
        
    ### ADAM
    lr_min  = 1e-7
    gamma = (lr_min / lr) ** (1 / n_epochs)
    optimizer = torch.optim.Adam(
        model.network.parameters(),
        lr=lr,
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=gamma,
    )
    losses = []

    for epoch in tqdm(range(n_epochs)):
        epoch_loss = 0.0
        n_seen = 0

        for theta in train_loader:
            theta = theta.to(device)

            optimizer.zero_grad(set_to_none=True)

            loss = model.loss(theta)

            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite loss at epoch {epoch}: {loss}")

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.network.parameters(),
                max_norm=10.0,
            )

            optimizer.step()

            batch_size = theta.shape[0]

            epoch_loss += (
                loss.detach().cpu().item()
                * batch_size
            )
            n_seen += batch_size

        average_loss = epoch_loss / n_seen
        losses.append(average_loss)

        absolute_epoch = (
            previous_epochs + epoch + 1
        )

        # Save only when the model improves on the stored best model.
        if average_loss < best_loss:
            best_loss = average_loss

            torch.save(
                {
                    "model_state_dict": (
                        model.network.state_dict()
                    ),
                    "best_loss": best_loss,
                    "epoch": absolute_epoch,
                },
                checkpoint_path,
            )

            print(
                f"\nSaved new best model: "
                f"epoch={absolute_epoch}, "
                f"NLL={best_loss:.6f}"
            )

        scheduler.step()

    # Ensure all diagnostics and sampling use the best model,
    # rather than merely the weights from the final epoch.
    best_checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.network.load_state_dict(
        best_checkpoint["model_state_dict"],
        strict=True,
    )

    model.network.eval()

    print("\nRestored best model after training")
    print(
        f"Best epoch: {best_checkpoint['epoch']}"
    )
    print(
        f"Best NLL:   {best_checkpoint['best_loss']:.6f}"
    )

    return losses

def make_copula_ml_metadata():
    return {
        "dataset_settings": {
            "type": "posterior_samples_ml",
            "theta_dim": 30,
        },
        "train_settings": {
            "model": {
                "posterior_model_type": "copula_normalizing_flow",
                "posterior_kwargs": {
                    "conditional": False,
                    "block_dims": [15, 15],
                    "flows": {
                        "flow_1": {
                            "num_flow_steps": 4,
                            "base_transform_kwargs": {
                                "hidden_dim": 64,
                                "num_transform_blocks": 4,
                                "activation": "elu",
                                "dropout_probability": 0.0,
                                "batch_norm": True,
                                "num_bins": 32,
                                "base_transform_type": "rq-coupling",
                            },
                        },
                        "flow_2": {
                            "num_flow_steps": 4,
                            "base_transform_kwargs": {
                                "hidden_dim": 64,
                                "num_transform_blocks": 4,
                                "activation": "elu",
                                "dropout_probability": 0.0,
                                "batch_norm": True,
                                "num_bins": 32,
                                "base_transform_type": "rq-coupling",
                            },
                        },
                    },
                    "CopulaKwargs": {
                        "P": 8,
                    },
                },

                # Explicitly absent.
                "embedding_kwargs": None,
            }
        },
    }

def _samples_to_2d_numpy(samples, name: str) -> np.ndarray:
    """
    Convert training or generated samples to shape [N, D].

    Supported inputs:
        [N, D]
        [N, 1, D]
        [1, N, D]
    """
    if torch.is_tensor(samples):
        samples = samples.detach().cpu().numpy()
    else:
        samples = np.asarray(samples)

    if samples.ndim == 3:
        if samples.shape[1] == 1:
            # Unconditional copula output: [N, 1, D]
            samples = samples[:, 0, :]
        elif samples.shape[0] == 1:
            # Alternative Dingo convention: [1, N, D]
            samples = samples[0, :, :]
        else:
            raise ValueError(
                f"{name} has ambiguous three-dimensional shape "
                f"{samples.shape}. Expected [N, 1, D] or [1, N, D]."
            )

    if samples.ndim != 2:
        raise ValueError(
            f"{name} must reduce to shape [N, D], "
            f"but has shape {samples.shape}."
        )

    if not np.isfinite(samples).all():
        raise ValueError(f"{name} contains NaN or Inf.")

    return samples

def plot_ml_training_vs_samples_corner(
    model,
    training_data,
    parameter_names,
    output_path,
    *,
    selected_parameters=None,
    n_model_samples=5000,
    max_training_samples=5000,
    bins=35,
    quantile_range=(0.005, 0.995),
    random_seed=1234,
):
    """
    Overlay training data and samples from a fitted unconditional ML model.

    Parameters
    ----------
    model
        Trained CopulaNormalizingFlowModel.

    training_data
        Original tensor used for training, usually `theta`, with shape [N, D].

    parameter_names
        Names corresponding to the D columns, for example `result.columns`.

    output_path
        Output PNG path.

    selected_parameters
        Optional list of parameter names to include. If None, all dimensions
        are plotted.

    n_model_samples
        Number of samples drawn from the fitted ML model.

    max_training_samples
        Maximum number of original training samples shown. Subsampling only
        affects the plot, not the trained model.

    quantile_range
        Quantile limits used to determine common plotting ranges. This avoids
        a few extreme samples distorting every panel.

    Returns
    -------
    model_samples : np.ndarray
        All generated model samples with shape [n_model_samples, D].
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    parameter_names = [str(name) for name in parameter_names]

    training_np = _samples_to_2d_numpy(
        training_data,
        name="training_data",
    )

    if training_np.shape[1] != len(parameter_names):
        raise ValueError(
            f"training_data has D={training_np.shape[1]}, but "
            f"{len(parameter_names)} parameter names were supplied."
        )

    # Batch-normalization layers must use their fitted running statistics.
    model.network.eval()

    with torch.no_grad():
        generated = model.sample(
            num_samples=n_model_samples,
        )

    model_np = _samples_to_2d_numpy(
        generated,
        name="model samples",
    )

    if model_np.shape[1] != training_np.shape[1]:
        raise ValueError(
            f"Model samples have D={model_np.shape[1]}, while the "
            f"training data have D={training_np.shape[1]}."
        )

    # Select columns by parameter name.
    if selected_parameters is None:
        selected_indices = np.arange(training_np.shape[1])
        selected_names = parameter_names
    else:
        selected_parameters = [
            str(name) for name in selected_parameters
        ]

        missing = [
            name
            for name in selected_parameters
            if name not in parameter_names
        ]

        if missing:
            raise KeyError(
                f"Requested corner-plot parameters were not found: {missing}"
            )

        selected_indices = np.array(
            [parameter_names.index(name) for name in selected_parameters],
            dtype=int,
        )
        selected_names = selected_parameters

    if len(selected_names) > 12:
        print(
            f"Warning: plotting {len(selected_names)} dimensions creates "
            f"{len(selected_names) ** 2} corner panels."
        )

    training_selected = training_np[:, selected_indices]
    model_selected = model_np[:, selected_indices]

    # Subsample the training cloud to prevent it from dominating the figure.
    rng = np.random.default_rng(random_seed)

    n_training_plot = min(
        int(max_training_samples),
        training_selected.shape[0],
    )

    if n_training_plot < training_selected.shape[0]:
        selected_rows = rng.choice(
            training_selected.shape[0],
            size=n_training_plot,
            replace=False,
        )
        training_plot = training_selected[selected_rows]
    else:
        training_plot = training_selected

    # Use common robust ranges for both data clouds.
    combined = np.vstack(
        [training_plot, model_selected]
    )

    lower_q, upper_q = quantile_range

    if not 0.0 <= lower_q < upper_q <= 1.0:
        raise ValueError(
            "quantile_range must satisfy "
            "0 <= lower < upper <= 1."
        )

    lower = np.quantile(combined, lower_q, axis=0)
    upper = np.quantile(combined, upper_q, axis=0)

    span = upper - lower

    # Add a small margin and handle nearly constant parameters.
    padding = np.where(
        span > 0,
        0.05 * span,
        0.01 * np.maximum(np.abs(lower), 1.0),
    )

    plotting_ranges = [
        (float(lo - pad), float(hi + pad))
        for lo, hi, pad in zip(lower, upper, padding)
    ]

    # Draw the original training distribution first.
    fig = corner.corner(
        training_plot,
        labels=selected_names,
        range=plotting_ranges,
        bins=bins,
        color="C0",
        plot_datapoints=False,
        plot_density=True,
        plot_contours=True,
        fill_contours=False,
        smooth=1.0,
        smooth1d=1.0,
        hist_kwargs={
            "alpha": 0.45,
        },
        contour_kwargs={
            "linewidths": 1.2,
        },
        label_kwargs={
            "fontsize": 9,
        },
        show_titles=True,
        title_fmt=".4g",
        title_kwargs={
            "fontsize": 9,
        },
    )

    # Overlay samples from the fitted ML model.
    corner.corner(
        model_selected,
        labels=selected_names,
        range=plotting_ranges,
        bins=bins,
        color="C1",
        fig=fig,
        plot_datapoints=False,
        plot_density=True,
        plot_contours=True,
        fill_contours=False,
        smooth=1.0,
        smooth1d=1.0,
        hist_kwargs={
            "alpha": 0.45,
        },
        contour_kwargs={
            "linewidths": 1.2,
        },
        show_titles=False,
    )

    legend_handles = [
        Line2D(
            [],
            [],
            color="C0",
            linewidth=2.0,
            label=f"Training data (n={training_plot.shape[0]})",
        ),
        Line2D(
            [],
            [],
            color="C1",
            linewidth=2.0,
            label=f"Fitted ML model (n={model_selected.shape[0]})",
        ),
    ]

    fig.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        frameon=True,
    )

    fig.suptitle(
        "Training posterior samples versus fitted copula model",
        y=1.01,
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"Saved corner plot to: {output_path}")
    print(f"Training samples plotted: {training_plot.shape[0]}")
    print(f"Model samples plotted:    {model_selected.shape[0]}")
    print(f"Parameters plotted:       {selected_names}")

    return model_np

#TODO: add validation + early stopping
#TODO: corner plot + masses
def main_hdf5():
    h5_path = (
        "/root/phd/GW_seperation_analysis/postProcessing/results_joint_final.hdf5"
    )

    #build scenario
    ScenConfig     = ScenarioConfig()
    scenario,logger,plot_dir,data_dir = setUpLoggerScenario(ScenConfig)
    data_dir = data_dir/ "results_joint_final.hdf5"
    results = utils.exctractResults(data_dir,logger)
    result = utils.join_waveform_posteriors(results)
    
    theta = torch.from_numpy(
        result.to_numpy(dtype=np.float32)
    )
    train_loader = DataLoader(
        theta,
        batch_size=1000,
        shuffle=True,
    )
    model = CopulaNormalizingFlowModel(
        metadata=make_copula_ml_metadata(),
        device='cpu',
    )
    parameter_inventory(model)
    losses = train_copula_ML(model
                             ,train_loader
                             ,n_epochs = 200
                             ,lr = 1e-6
                             ,start_from_best = True
                             )
    
    diagnostics_dir = Path(plot_dir) / "copula_ml"
    plot_ml_losses(
        losses,
        diagnostics_dir / "training_losses.png",
        moving_average_window=10,
        skip_initial_epochs=1,

    )

    print_fitted_copula_model(
        model,
        parameter_names=result.columns,
        output_dir=diagnostics_dir,
    )
    
    preferred_parameters = [
    "chirp_mass_A",
    "mass_ratio_A",
    "chirp_mass_B",
    "mass_ratio_B",
    'psi_A', 'phase_A',
    'psi_B', 'phase_B',
    'geocent_time_B','geocent_time_A'
    ]

    # Keep only names that actually occur in the HDF5 result.
    corner_parameters = [
        name
        for name in preferred_parameters
        if name in result.columns
    ]

    # Fallback in case these precise variables are not present.
    if len(corner_parameters) < 2:
        corner_parameters = list(result.columns[:6])

    generated_samples = plot_ml_training_vs_samples_corner(
        model=model,
        training_data=theta,
        parameter_names=result.columns,
        selected_parameters=corner_parameters,
        output_path=diagnostics_dir / "training_vs_ml_corner.png",
        n_model_samples=5000,
        max_training_samples=5000,
        bins=35,
    )
    print("end")

if __name__ == "__main__":
    main_hdf5()