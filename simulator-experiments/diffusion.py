import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pickle

intervention_success_rate = 0.7

########################################
#   INTERVENTION & REWARD FUNCTIONS
########################################
def temperature_penalty(temp):
    return 0 if temp <= 38 else -math.exp(abs(temp - 38.0)/2)

def pulse_penalty(pulse):
    return 0 if pulse <= 120 else -math.exp(abs(pulse - 120) / 17)

def respiratory_penalty(rr):
    return 0 if rr <= 30 else -math.exp(abs(rr - 30) / 5)

def spo2_penalty(spo2):
    return 0 if spo2 >= 90 else -math.exp(abs(spo2 - 90) / 4)

def reward_function(sign_dict):
    """Compute a reward (negative if abnormal) given raw vital values."""
    reward = 0
    if "COVERED_SKIN_TEMPERATURE" in sign_dict:
        reward += temperature_penalty(sign_dict["COVERED_SKIN_TEMPERATURE"])
    if "PULSE_RATE" in sign_dict:
        reward += pulse_penalty(sign_dict["PULSE_RATE"])
    if "RESPIRATORY_RATE" in sign_dict:
        reward += respiratory_penalty(sign_dict["RESPIRATORY_RATE"])
    if "SPO2" in sign_dict:
        reward += spo2_penalty(sign_dict["SPO2"])
    return reward


def improve_vital_signs3(sign_dict):
    """
    If a random chance succeeds and a sign is abnormal, adjust it towards normal.
    (Adjust the magnitudes as needed for raw-scale values.)
    """
    if random.random() < intervention_success_rate:
        if "COVERED_SKIN_TEMPERATURE" in sign_dict and temperature_penalty(sign_dict["COVERED_SKIN_TEMPERATURE"]) < 0:
            sign_dict["COVERED_SKIN_TEMPERATURE"] -= np.random.normal(0.5, 0.2)
        if "PULSE_RATE" in sign_dict and pulse_penalty(sign_dict["PULSE_RATE"]) < 0:
            sign_dict["PULSE_RATE"] -= np.random.normal(5, 2)
        if "RESPIRATORY_RATE" in sign_dict and respiratory_penalty(sign_dict["RESPIRATORY_RATE"]) < 0:
            sign_dict["RESPIRATORY_RATE"] -= np.random.normal(2, 1)
        if "SPO2" in sign_dict and spo2_penalty(sign_dict["SPO2"]) < 0:
            sign_dict["SPO2"] += np.random.normal(1, 0.5)
    return sign_dict

def apply_interventions_in_raw_space(trajectories, vital_signs):
    """
    For each sample and time step in raw-scale trajectories,
    if reward < 0, apply improve_vital_signs3.
    trajectories: (num_samples, seq_len, num_vitals)
    """
    num_samples, seq_len, _ = trajectories.shape
    for i in range(num_samples):
        for t in range(seq_len):
            sign_dict = dict(zip(vital_signs, trajectories[i, t, :]))
            r = reward_function(sign_dict)
            if r < 0:
                updated = improve_vital_signs3(sign_dict)
                for idx, vs in enumerate(vital_signs):
                    trajectories[i, t, idx] = updated[vs]
    return trajectories


def minmax_transform(value, minv, maxv):
    return (value - minv) / (maxv - minv + 1e-8)

def minmax_inverse(value, minv, maxv):
    return value * (maxv - minv + 1e-8) + minv

########################################
# DDPM TIME SERIES MODEL WITH TIMESTEP EMBEDDING
########################################
class TimestepEmbed(nn.Module):
    def __init__(self, embed_dim=64):
        super().__init__()
        self.lin1 = nn.Linear(1, embed_dim)
        self.lin2 = nn.Linear(embed_dim, embed_dim)
    def forward(self, t):
        # t: (batch,)
        t = t.unsqueeze(1).float()  # (batch, 1)
        x = torch.relu(self.lin1(t))
        x = torch.relu(self.lin2(x))
        return x

class DDPMTimeSeriesModel(nn.Module):
    def __init__(self, num_features=4, hidden_dim=64, embed_dim=64):
        super().__init__()
        self.t_embed = TimestepEmbed(embed_dim=embed_dim)
        self.rnn = nn.LSTM(input_size=num_features + embed_dim,
                           hidden_size=hidden_dim,
                           batch_first=True)
        self.linear = nn.Linear(hidden_dim, num_features)
    def forward(self, x, t):
        # x: (batch, T, num_features)
        # t: (batch,)
        t_emb = self.t_embed(t)            # (batch, embed_dim)
        t_emb = t_emb.unsqueeze(1).expand(-1, x.size(1), -1)  # (batch, T, embed_dim)
        x_in = torch.cat([x, t_emb], dim=-1)  # (batch, T, num_features+embed_dim)
        h, _ = self.rnn(x_in)
        eps_pred = self.linear(h)
        return eps_pred

########################################
#  DDPM Training and Sampling Helpers
########################################
def make_beta_schedule(num_timesteps=50, beta_start=1e-4, beta_end=2e-2):
    return torch.linspace(beta_start, beta_end, num_timesteps)

class DDPMTrainer:
    def __init__(self, model, num_timesteps=50, beta_start=1e-4, beta_end=2e-2):
        self.model = model
        self.num_timesteps = num_timesteps
        self.beta = make_beta_schedule(num_timesteps, beta_start, beta_end)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
    def train(self, data, num_epochs=5, batch_size=32):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar).to(device)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar).to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(data).float())
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for epoch in range(num_epochs):
            total_loss = 0
            for batch_x0, in loader:
                batch_x0 = batch_x0.to(device)
                optimizer.zero_grad()
                bsz = batch_x0.size(0)
                t = torch.randint(0, self.num_timesteps, (bsz,), device=device)
                # Forward diffusion: x_t = sqrt(alpha_bar[t]) * x0 + sqrt(1-alpha_bar[t]) * noise
                x_t, eps = forward_diffusion_sample(
                    batch_x0, t,
                    self.sqrt_alpha_bar, self.sqrt_one_minus_alpha_bar
                )
                eps_pred = self.model(x_t, t)
                loss = nn.MSELoss()(eps_pred, eps)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(loader)
            print(f"Epoch {epoch}/{num_epochs} - Loss: {avg_loss:.4f}")

def forward_diffusion_sample(x0, t, sqrt_alpha_bar, sqrt_one_minus_alpha_bar):
    bsz = x0.size(0)
    sqrt_ab = sqrt_alpha_bar[t].view(-1,1,1)
    sqrt_1m_ab = sqrt_one_minus_alpha_bar[t].view(-1,1,1)
    eps = torch.randn_like(x0)
    x_t = sqrt_ab * x0 + sqrt_1m_ab * eps
    return x_t, eps

def ddpm_sample(model, trainer, shape=(10,50,4), device="cpu"):
    model.eval()
    x_t = torch.randn(shape, device=device)
    with torch.no_grad():
        for t_step in reversed(range(trainer.num_timesteps)):
            t_tensor = torch.tensor([t_step]*shape[0], device=device)
            beta_t = trainer.beta[t_step].to(device)
            alpha_t = trainer.alpha[t_step].to(device)
            alpha_bar_t = trainer.alpha_bar[t_step].to(device)
            eps_pred = model(x_t, t_tensor)
            alpha_t_sqrt = torch.sqrt(alpha_t)
            one_minus_alpha_bar_t = 1.0 - alpha_bar_t
            x0_est = (1.0 / alpha_t_sqrt) * (x_t - (1.0 - alpha_t)/torch.sqrt(one_minus_alpha_bar_t + 1e-8)*eps_pred)
            if t_step > 0:
                sigma_t = torch.sqrt(beta_t)
                z = torch.randn_like(x_t)
                x_t = x0_est + sigma_t * z
            else:
                x_t = x0_est
            # Clamp to [0,1] if training data is normalized
            x_t = torch.clamp(x_t, 0.0, 1.0)
    return x_t.cpu().numpy()

########################################
#  Plotting Functions
########################################
def plot_spaghetti(real_data, synth_data, vital_signs):
    """
    real_data: (N_real, T, num_features)
    synth_data: (N_synth, T, num_features)
    """
    T = real_data.shape[1]
    fig, axes = plt.subplots(1, len(vital_signs), figsize=(6*len(vital_signs), 4))
    for idx, vs in enumerate(vital_signs):
        ax = axes[idx]
        for i in range(real_data.shape[0]):
            ax.plot(range(T), real_data[i, :, idx], alpha=0.3, color='blue')
        for i in range(synth_data.shape[0]):
            ax.plot(range(T), synth_data[i, :, idx], alpha=0.3, color='orange')
        ax.set_title(vs)
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Value")
        ax.grid(True)
    real_line = plt.Line2D([], [], color='blue', alpha=0.3, label='Real')
    synth_line = plt.Line2D([], [], color='orange', alpha=0.3, label='Synthetic')
    fig.legend(handles=[real_line, synth_line], loc='upper right')
    plt.tight_layout()
    plt.savefig('../data/difussion_spaghetti.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

def compute_mean_std_over_time(data_array):
    means = data_array.mean(axis=0)
    stds = data_array.std(axis=0)
    return means, stds

def plot_mean_std(ax, time_axis, mean, std, label="", color="blue", alpha=0.2):
    ax.plot(time_axis, mean, label=label, color=color)
    ax.fill_between(time_axis, mean-std, mean+std, color=color, alpha=alpha)

def plot_mean_std_comparison(real_data, synth_data, vital_signs):
    T = real_data.shape[1]
    time_axis = np.arange(T)
    fig, axes = plt.subplots(1, len(vital_signs), figsize=(6*len(vital_signs), 4))
    for idx, vs in enumerate(vital_signs):
        ax = axes[idx]
        real_means, real_stds = compute_mean_std_over_time(real_data[:, :, idx])
        plot_mean_std(ax, time_axis, real_means, real_stds, label="Real", color='blue', alpha=0.2)
        synth_means, synth_stds = compute_mean_std_over_time(synth_data[:, :, idx])
        plot_mean_std(ax, time_axis, synth_means, synth_stds, label="Synthetic", color='orange', alpha=0.2)
        ax.set_title(vs)
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Value")
        ax.grid(True)
        ax.legend()
    plt.tight_layout()
    plt.savefig('../data/diffusion_mean_std.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

def create_time_series_dataset(pivot_df, vital_signs, sequence_length=50):
    groups = pivot_df.groupby('patient_id')
    all_sequences = []

    for pid, group in groups:
        group = group.sort_values('generatedat')
        # we only keep the vital_signs columns as a NumPy array
        arr = group[vital_signs].values  # shape: (num_time_steps, len(vital_signs))
        # extract overlapping windows of length sequence_length
        for start_idx in range(0, len(arr) - sequence_length + 1):
            snippet = arr[start_idx : start_idx + sequence_length]
            all_sequences.append(snippet)

    all_sequences = np.array(all_sequences)  # shape => (N, T, #features)
    return all_sequences

########################################
#  Main Script: Data, Training, Sampling, Plotting
########################################
if __name__ == "__main__":
    # Assume you have a pivot_df in raw scale
    with open("../data/pivot_df_normal.pkl", "rb") as f:
        pivot_df = pickle.load(f)
    pivot_df['generatedat'] = pd.to_datetime(pivot_df['generatedat'])
    vital_signs = ["PULSE_RATE", "RESPIRATORY_RATE", "SPO2", "COVERED_SKIN_TEMPERATURE"]

    # Create time-series dataset (raw scale)
    all_data_raw = create_time_series_dataset(pivot_df, vital_signs, sequence_length=50)
    print("all_data_raw shape:", all_data_raw.shape)

    # Compute min/max for each vital sign (from raw data)
    num_vitals = len(vital_signs)
    flattened = all_data_raw.reshape(-1, num_vitals)
    min_vals = flattened.min(axis=0)
    max_vals = flattened.max(axis=0)

    # Transform raw data to normalized [0,1]
    all_data_norm = np.zeros_like(all_data_raw)
    for i in range(num_vitals):
        all_data_norm[..., i] = minmax_transform(all_data_raw[..., i], min_vals[i], max_vals[i])

    # Train the diffusion model on normalized data
    model = DDPMTimeSeriesModel(num_features=num_vitals, hidden_dim=64, embed_dim=64)
    trainer = DDPMTrainer(model, num_timesteps=50, beta_start=1e-4, beta_end=2e-2)
    trainer.train(all_data_norm, num_epochs=20, batch_size=32)

    # Sample from the diffusion model (normalized outputs)
    samples_norm = ddpm_sample(model, trainer, shape=(40, 50, num_vitals), device="cuda" if torch.cuda.is_available() else "cpu")
    print("samples_norm shape:", samples_norm.shape)

    # Invert normalization => synthetic data in raw scale
    synthetic_data_raw = np.zeros_like(samples_norm)
    for i in range(num_vitals):
        synthetic_data_raw[..., i] = minmax_inverse(samples_norm[..., i], min_vals[i], max_vals[i])

    # Apply interventions in raw scale
    synthetic_data_raw = apply_interventions_in_raw_space(synthetic_data_raw, vital_signs)
    print("Final synthetic data shape:", synthetic_data_raw.shape)

    # Grab 40 real trajectories of length 50 from pivot_df (raw scale)
    grouped = pivot_df.groupby('patient_id')
    MIN_LENGTH = 50
    long_enough_ids = [pid for pid, grp in grouped if len(grp) >= MIN_LENGTH]
    real_ids = random.sample(long_enough_ids, min(40, len(long_enough_ids)))
    real_data = np.zeros((len(real_ids), 50, num_vitals))
    for i, pid in enumerate(real_ids):
        subdf = grouped.get_group(pid).copy()
        subdf.sort_values('generatedat', inplace=True)
        subdf.reset_index(drop=True, inplace=True)
        subdf = subdf.iloc[:50]
        for v_idx, vs in enumerate(vital_signs):
            real_data[i, :len(subdf), v_idx] = subdf[vs].values

    # Plot spaghetti
    plot_spaghetti(real_data, synthetic_data_raw, vital_signs)

    # Plot mean ± std comparison
    plot_mean_std_comparison(real_data, synthetic_data_raw, vital_signs)