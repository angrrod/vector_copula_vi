
import matplotlib.pyplot as plt
import torch
import numpy as np
import corner
from dingo.core.posterior_models.vector_copula_Model import CopulaNormalizingFlowModel
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

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
                    "type": "identity",
                    "input_dim": 4,
                    "output_dim": 4,
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

if __name__ == "__main__":
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