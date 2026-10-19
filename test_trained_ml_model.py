import torch
import numpy as np
import pandas as pd
from dingo.core.posterior_models.vector_copula_Model import CopulaNormalizingFlowModel
import utils
from scipy.spatial import cKDTree
import numpy as np
import bilby
from utils import get_base_log_dir,get_postprocessing_dir

# ============================================================
# Load trained PyTorch checkpoint
# ============================================================

def make_copula_ml_metadata(theta_dim,block_dims,p,is_independent):
    return {
        "dataset_settings": {
            "type": "posterior_samples_ml",
            "theta_dim": theta_dim,
        },
        "train_settings": {
            "model": {
                "posterior_model_type": "copula_normalizing_flow",
                "posterior_kwargs": {
                    "conditional": False,
                    "block_dims": block_dims,
                    "flows": {
                        "flow_1": {
                            "num_flow_steps": 6,
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
                            "num_flow_steps": 6,
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
                        "P": p,
                        "is_independent" : is_independent
                    },
                },

                # Explicitly absent.
                "embedding_kwargs": None,
            }
        },
    }

def checkModel(checkpoint_path):
    parameter_names = [
        "mass_ratio",
        "chirp_mass",
        "luminosity_distance",
        "dec",
        "ra",
        "theta_jn",
        "psi",
        "phase",
        "a_1",
        "a_2",
        "tilt_1",
        "tilt_2",
        "phi_12",
        "phi_jl",
        "geocent_time",
    ]

    ckpt = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    print("Checkpoint keys:")
    print(ckpt.keys())


    # ============================================================
    # Extract model state
    # ============================================================

    state = ckpt["model_state_dict"]

    print("\nRelevant model parameters:")
    print("B shape:", state["B"].shape)
    print("z shape:", state["z"].shape)


    # B is stored as (1, 30, 16)
    B = state["B"].detach().double()[0]

    # raw scalar parameter
    z_raw = state["z"].detach().double().item()

    # actual positive zeta used by the model
    zeta = torch.nn.functional.softplus(
        torch.tensor(z_raw, dtype=torch.float64)
    ).item() + 1e-6

    print("\nraw z =", z_raw)
    print("zeta  =", zeta)


    # ============================================================
    # Reconstruct the 30 x 30 copula matrix
    # ============================================================

    D = B.shape[0]     # 30
    d = D // 2         # 15

    I = torch.eye(D, dtype=torch.float64)

    # Unnormalised covariance / copula structure
    Omega_tilde = zeta * I + B @ B.T


    # ============================================================
    # Block normalisation
    #
    # We want:
    #
    # Omega = [ I    C  ]
    #         [ C.T  I ]
    #
    # ============================================================

    block_diag = torch.zeros_like(Omega_tilde)

    block_diag[:d, :d] = Omega_tilde[:d, :d]
    block_diag[d:, d:] = Omega_tilde[d:, d:]

    # Cholesky of the two diagonal blocks
    A_inv = torch.linalg.cholesky(block_diag)

    # Whitening transform
    A = torch.linalg.inv(A_inv)

    # Final 30 x 30 Gaussian copula matrix
    Omega = A @ Omega_tilde @ A.T


    # ============================================================
    # Extract the interesting 15 x 15 cross block
    # ============================================================

    C = Omega[:d, d:]

    # ============================================================
    # Convert to Pandas DataFrames
    # ============================================================

    df_Omega = pd.DataFrame(
        Omega.numpy(),
        index=[f"x{i+1}" for i in range(30)],
        columns=[f"x{i+1}" for i in range(30)],
    )

    df_C = pd.DataFrame(
        C.numpy() if hasattr(C, "numpy") else C,
        index=[f"{p}_A" for p in parameter_names],
        columns=[f"{p}_B" for p in parameter_names],
    )


    # ============================================================
    # Print the 15 x 15 matrix
    # ============================================================

    print("\n" + "=" * 80)
    print("15 x 15 cross-copula matrix C")
    print("=" * 80)

    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", 250,
        "display.precision", 6,
    ):
        print(df_C)


    # ============================================================
    # Optional sanity checks
    # ============================================================

    print("\nSanity checks:")

    print(
        "First diagonal block max error from identity:",
        np.max(
            np.abs(
                Omega[:d, :d].numpy() - np.eye(d)
            )
        ),
    )

    print(
        "Second diagonal block max error from identity:",
        np.max(
            np.abs(
                Omega[d:, d:].numpy() - np.eye(d)
            )
        ),
    )

    eigenvalues = torch.linalg.eigvalsh(Omega)

    print("Minimum eigenvalue:", eigenvalues.min().item())
    print("Maximum eigenvalue:", eigenvalues.max().item())


    # ============================================================
    # Optional: save DataFrames
    # ============================================================

    df_C.to_csv("copula_cross_block_15x15.csv")
    df_Omega.to_csv("copula_full_30x30.csv")

    print("\nSaved:")
    print("  copula_cross_block_15x15.csv")
    print("  copula_full_30x30.csv")


    # Number of strongest correlations to display
    top_n = 20

    pairs = (
        df_C
        .stack()
        .rename("correlation")
        .reset_index()
    )

    pairs.columns = [
        "parameter_A",
        "parameter_B",
        "correlation",
    ]

    pairs["abs_correlation"] = pairs["correlation"].abs()

    top_correlations = (
        pairs
        .sort_values("abs_correlation", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    print(top_correlations)

def loadModel(is_independent,checkpoint_path):
    device='cpu'
    metadata = make_copula_ml_metadata( 
        theta_dim=30,
        block_dims=[15, 15],
        p=16,
        is_independent = is_independent
    )

    model = CopulaNormalizingFlowModel(
        metadata=metadata,
        device=device,
    )
    
    model,_ = load_best_model(model,checkpoint_path,device)
    return model 

def load_best_model(model,checkpoint_path,device):
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )

        model.network.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )
        return model,checkpoint

def sample_model(model, n=20_000):
    model.network.eval()

    with torch.no_grad():
        samples = model.sample(num_samples=n)

    samples = samples.detach().cpu().numpy()

    # Dingo unconditional output can be [N, 1, D]
    if samples.ndim == 3 and samples.shape[1] == 1:
        samples = samples[:, 0, :]

    return samples

def get_log_prob_components(theta_standardized, model):

    if isinstance(theta_standardized, pd.DataFrame):
        theta_standardized = theta_standardized.to_numpy()

    theta = torch.tensor(
        theta_standardized,
        dtype=torch.float32,
        device="cpu",
    )

    model.network.eval()

    with torch.no_grad():

        distribution = model.network.distribution()

        components = distribution.log_prob_components(theta)

    return components

def get_log_prob(theta_standardized, model):

    if isinstance(theta_standardized, pd.DataFrame):
        theta_standardized = theta_standardized.to_numpy()

    theta_tensor = torch.tensor(
        theta_standardized,
        dtype=torch.float32,
        device="cpu",
    )

    model.network.eval()

    with torch.no_grad():
        distribution = model.network.distribution()

        components = distribution.log_prob_components(
            theta_tensor
        )

    return components["log_prob_total"]

def estimate_kl_knn(
    p_samples,
    q_samples,
    k=5,
):
    """
    Estimate KL(P || Q) from samples.

    p_samples: samples from the true distribution P
    q_samples: samples from the estimated distribution Q
    """

    p = np.asarray(p_samples, dtype=np.float64)
    q = np.asarray(q_samples, dtype=np.float64)

    n, d = p.shape
    m = q.shape[0]

    tree_p = cKDTree(p)
    tree_q = cKDTree(q)

    # Distance to kth neighbour in P.
    # k+1 because the nearest P point is itself.
    rho = tree_p.query(
        p,
        k=k + 1,
        workers=-1,
    )[0][:, k]

    # Distance to kth neighbour in Q.
    nu = tree_q.query(
        p,
        k=k,
        workers=-1,
    )[0]

    if k > 1:
        nu = nu[:, k - 1]

    eps = np.finfo(float).tiny

    rho = np.maximum(rho, eps)
    nu = np.maximum(nu, eps)

    kl = (
        d * np.mean(np.log(nu / rho))
        + np.log(m / (n - 1))
    )

    return kl

def mainComparison(checkpoint_path_dep,checkpoint_path_indep,dep_1,dep_2):
    depModel   = loadModel(dep_1,checkpoint_path_dep)
    indepModel = loadModel(dep_2,checkpoint_path_indep)
    
    samples_dependent   = sample_model(depModel, n=20_000)
    samples_independent = sample_model(indepModel, n=20_000)
    
    # Build scenario.
    data_dir = get_postprocessing_dir()
    data_dir = (
        data_dir / "results_joint_final.hdf5"
    )
    #Logger
    log_dir = get_base_log_dir()

    bilby.core.utils.setup_logger(
        log_level="INFO", #DEBUG
        label="my_run", 
        outdir=log_dir,
    )
    logger = bilby.core.utils.logger
    results = utils.exctractResults(
        data_dir,
        logger,
    )

    true_samples = utils.join_waveform_posteriors(
        results
    )    
    
    theta_mean = true_samples.mean(axis=0)
    theta_std = true_samples.std(axis=0, ddof=1)
    true_samples_standardized = (
        true_samples - theta_mean
    ) / theta_std
    
    # Use the same number of true samples
    rng = np.random.default_rng(1234)

    idx = rng.choice(
        len(true_samples_standardized),
        size=20_000,
        replace=False,
    )

    true_20k = true_samples_standardized.iloc[idx].to_numpy()


    kl_dependent = estimate_kl_knn(
        true_20k,
        samples_dependent,
        k=5,
    )

    kl_independent = estimate_kl_knn(
        true_20k,
        samples_independent,
        k=5,
    )

    print(
        "KL(true || dependent)   =",
        kl_dependent,
    )

    print(
        "KL(true || independent) =",
        kl_independent,
    )
    log_prob_dep = get_log_prob(
        true_samples_standardized,
        depModel,
    )

    log_prob_indep = get_log_prob(
        true_samples_standardized,
        indepModel,
    )

    print("Mean log q dependent:",
        log_prob_dep.mean().item())

    print("Mean log q independent:",
        log_prob_indep.mean().item())
    
    print(
    "KL difference:",
        (log_prob_indep - log_prob_dep).mean().item()
    )
    
    comp_dep = get_log_prob_components(
        true_samples_standardized,
        depModel,
    )

    comp_indep = get_log_prob_components(
        true_samples_standardized,
        indepModel,
    )
    
    print("\nDEPENDENT")
    print(
        "marginals =",
        comp_dep["logp_marg_total"].mean().item(),
    )
    print(
        "copula    =",
        comp_dep["log_copula_total"].mean().item(),
    )
    print(
        "joint     =",
        comp_dep["log_prob_total"].mean().item(),
    )


    print("\nINDEPENDENT")
    print(
        "marginals =",
        comp_indep["logp_marg_total"].mean().item(),
    )
    print(
        "copula    =",
        comp_indep["log_copula_total"].mean().item(),
    )
    print(
        "joint     =",
        comp_indep["log_prob_total"].mean().item(),
    )

if __name__ == "__main__":
    checkpoint_path_cop_only = "/root/phd/GW_seperation_analysis/postProcessing/copula_ml/copula_ml_best_copula_only.pt"
    checkpoint_path_indep = "/root/phd/GW_seperation_analysis/postProcessing/copula_ml/copula_ml_best_independent.pt"
    checkpoint_path_joint = "/root/phd/GW_seperation_analysis/postProcessing/copula_ml/copula_ml_best_joint.pt"
    
    mainComparison(checkpoint_path_cop_only,checkpoint_path_indep,False,True)
    mainComparison(checkpoint_path_joint,checkpoint_path_indep,False,True)
    mainComparison(checkpoint_path_joint,checkpoint_path_cop_only,False,False)

    print("End")