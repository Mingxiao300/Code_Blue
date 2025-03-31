import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import math
import pickle
from random import sample
from typing import Dict, Tuple, Callable, Optional

# Constants
VITAL_SIGNS = ['PULSE_RATE', 'RESPIRATORY_RATE', 'SPO2', 'COVERED_SKIN_TEMPERATURE']
TIME_SIZE = 20

def preprocess_alarm_data(
    csv_path: str,
    vital_signs: list = VITAL_SIGNS,
    time_size: int = TIME_SIZE,
    reward_function: Optional[Callable] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process vital sign data and separate into normal and alarm periods.
    
    Args:
        csv_path: Path to the input CSV file
        vital_signs: List of vital signs to include
        time_size: Minutes between samples when resampling
        reward_function: Function to calculate rewards
    
    Returns:
        Tuple of (normal_data, alarm_data) DataFrames
    """
    # 1) Read and clean data
    df = pd.read_csv(csv_path)
    df = _clean_raw_data(df, vital_signs)
    
    # 2) Split into alarm/normal periods
    df_alarm, df_normal = _split_alarm_periods(df)
    
    # 3) Process both datasets identically
    pivot_df_normal = _process_and_pivot(df_normal, vital_signs, time_size, reward_function)
    pivot_df_alarm = _process_and_pivot(df_alarm, vital_signs, time_size, reward_function)
    
    return pivot_df_normal, pivot_df_alarm

def _clean_raw_data(df: pd.DataFrame, vital_signs: list) -> pd.DataFrame:
    """Clean and filter raw input data."""
    # Convert datetimes
    df['generatedat'] = pd.to_datetime(df['generatedat'], errors='coerce')
    if 'alarm_vital_boundary_start_time' in df.columns:
        df['alarm_vital_boundary_start_time'] = pd.to_datetime(
            df['alarm_vital_boundary_start_time'], errors='coerce'
        )
    
    # Filter data
    df.dropna(subset=['patient_id', 'doublevalue'], inplace=True)
    df = df[df['name'].isin(vital_signs)]
    
    # Remove outliers
    conditions = [
        (df['name'] == 'PULSE_RATE') & (df['doublevalue'] > 300),
        (df['name'] == 'RESPIRATIOUS_RATE') & (df['doublevalue'] > 60),
        (df['name'] == 'SPO2') & ((df['doublevalue'] < 50) | (df['doublevalue'] > 100)),
        (df['name'] == 'COVERED_SKIN_TEMPERATURE') & (df['doublevalue'] > 45)
    ]
    for cond in conditions:
        df = df[~cond]
    
    return df

def _split_alarm_periods(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into alarm and normal periods."""
    alarm_rows = df[~df['alarm_id'].isna()].copy()
    alarm_times_by_patient = (
        alarm_rows
        .dropna(subset=['alarm_vital_boundary_start_time'])
        .groupby('patient_id')['alarm_vital_boundary_start_time']
        .unique()
        .to_dict()
    )

    def is_in_alarm_window(row):
        pid = row['patient_id']
        gen_time = row['generatedat']
        if pid not in alarm_times_by_patient:
            return False
        for alarm_start in alarm_times_by_patient[pid]:
            if alarm_start is pd.NaT:
                continue
            alarm_end = alarm_start + pd.Timedelta(hours=24)
            if alarm_start <= gen_time <= alarm_end:
                return True
        return False

    df['in_alarm_period'] = (~df['alarm_id'].isna()) | df.apply(is_in_alarm_window, axis=1)
    df_alarm = df[df['in_alarm_period']].copy()
    df_normal = df[~df['in_alarm_period']].copy()
    
    df_alarm.drop(columns=['in_alarm_period'], inplace=True)
    df_normal.drop(columns=['in_alarm_period'], inplace=True)
    
    return df_alarm, df_normal

def _process_and_pivot(
    df: pd.DataFrame,
    vital_signs: list,
    time_size: int,
    reward_function: Optional[Callable]
) -> pd.DataFrame:
    """Process and pivot a single DataFrame."""
    df = df.copy()
    df.set_index('generatedat', inplace=True)
    df.sort_index(inplace=True)
    
    median_values = (
        df
        .groupby(['patient_id', 'name'])
        .resample(f"{time_size}T")['doublevalue']
        .median()
        .reset_index()
    )
    
    pivot_df = median_values.pivot_table(
        index=['patient_id', 'generatedat'],
        columns='name',
        values='doublevalue'
    ).dropna().reset_index()
    
    pivot_df['reward'] = 0 if reward_function is None else pivot_df.apply(
        lambda row: reward_function(row[vital_signs].to_dict()), axis=1)
    
    return pivot_df

# Reward calculation functions
def reward_function(sign_dict: Dict[str, float], rev_norm: bool = False, o_values: Optional[Dict] = None) -> float:
    """Calculate reward based on vital signs."""
    if rev_norm:
        sign_dict = clean_data(VITAL_SIGNS, sign_dict, o_values)
    reward = 0
    for sign, value in sign_dict.items():
        if sign == "COVERED_SKIN_TEMPERATURE":
            reward += temperature_penalty(value)
        elif sign == "PULSE_RATE":
            reward += pulse_penalty(value)
        elif sign == "RESPIRATIOUS_RATE":
            reward += respiratory_penalty(value)
        elif sign == "SPO2":
            reward += spo2_penalty(value)
    return reward

def clean_data(vital_signs: list, p_df: pd.DataFrame, min_max: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    """Reverse min-max normalization."""
    for sign in vital_signs:
        p_df[sign] = reverse_min_max_normalize(p_df[sign], min_max[sign][0], min_max[sign][1])
    return p_df

def reverse_min_max_normalize(column: pd.Series, min_val: float, max_val: float) -> pd.Series:
    """Reverse min-max normalization."""
    return column * (max_val - min_val) + min_val

def min_max_normalize(column: pd.Series) -> pd.Series:
    """Apply min-max normalization."""
    return (column - column.min()) / (column.max() - column.min())

def temperature_penalty(temperature: float) -> float:
    """Calculate temperature penalty."""
    return 0 if temperature <= 38 else -math.exp(abs(temperature - 38.0)/2)

def pulse_penalty(pulse: float) -> float:
    """Calculate pulse rate penalty."""
    return 0 if pulse <= 120 else -math.exp(abs(pulse - 120) / 17)

def respiratory_penalty(respiratory_rate: float) -> float:
    """Calculate respiratory rate penalty."""
    return 0 if respiratory_rate <= 30 else -math.exp(abs(respiratory_rate - 30) / 5)

def spo2_penalty(spo2: float) -> float:
    """Calculate SpO2 penalty."""
    return 0 if spo2 >= 90 else -math.exp(abs(spo2 - 90) / 4)

def blood_penalty(blood_pressure: float) -> float:
    """Calculate blood pressure penalty."""
    return 0 if blood_pressure <= 127 else -math.exp(abs(blood_pressure - 127) / 5)

# Visualization functions
def prepare_trajectories(df: pd.DataFrame, vital_signs: list, n_samples: int = 20, T: int = 24) -> np.ndarray:
    """Prepare trajectories for visualization."""
    unique_patients = df['patient_id'].unique()
    sampled_patients = sample(list(unique_patients), min(n_samples, len(unique_patients)))
    
    trajectories = np.full((len(sampled_patients), T, len(vital_signs)), np.nan)
    
    for i, patient in enumerate(sampled_patients):
        patient_df = df[df['patient_id'] == patient].sort_values('generatedat')
        for j, vital in enumerate(vital_signs):
            if vital in patient_df.columns:
                values = patient_df[vital].values[:T]
                trajectories[i, :len(values), j] = values
                
    return trajectories

def plot_vital_comparisons(normal_data: pd.DataFrame, alarm_data: pd.DataFrame, T: int = 50):
    """Generate and save comparison plots for vital signs including both spaghetti and mean±std plots."""
    
    # Prepare trajectory data
    normal_traj = prepare_trajectories(normal_data, VITAL_SIGNS, T=T)
    alarm_traj = prepare_trajectories(alarm_data, VITAL_SIGNS, T=T)
    time_axis = np.arange(T)

    # 1. Spaghetti Plots
    fig1, axes1 = plt.subplots(1, 4, figsize=(25, 5))
    for idx, vital in enumerate(VITAL_SIGNS):
        ax = axes1[idx]
        # Plot normal trajectories
        for i in range(normal_traj.shape[0]):
            ax.plot(time_axis, normal_traj[i, :, idx], alpha=0.3, color='blue', linewidth=1)
        # Plot alarm trajectories
        for i in range(alarm_traj.shape[0]):
            ax.plot(time_axis, alarm_traj[i, :, idx], alpha=0.3, color='red', linewidth=1)
        
        ax.set_title(vital)
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Value")
        ax.grid(True)
    
    fig1.legend(
        handles=[
            mlines.Line2D([], [], color='blue', alpha=0.3, label='Normal Periods'),
            mlines.Line2D([], [], color='red', alpha=0.3, label='Alarm Periods')
        ],
        loc='upper right',
        bbox_to_anchor=(0.95, 0.9)
    )
    plt.tight_layout()
    plt.savefig('../data/vital_signs_spaghetti.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)

    # 2. Mean ± STD Plots
    fig2, axes2 = plt.subplots(1, 4, figsize=(25, 5))
    
    def compute_mean_std_over_time(data):
        """Compute mean and std across patients for each time point"""
        means = np.nanmean(data, axis=0)
        stds = np.nanstd(data, axis=0)
        return means, stds

    def plot_mean_std(ax, x, means, stds, label, color, alpha=0.2):
        """Plot mean with shaded std region"""
        ax.plot(x, means, color=color, label=label)
        ax.fill_between(x, means-stds, means+stds, color=color, alpha=alpha)

    for idx, vital in enumerate(VITAL_SIGNS):
        ax = axes2[idx]
        
        # Normal data
        normal_means, normal_stds = compute_mean_std_over_time(normal_traj[:, :, idx])
        plot_mean_std(ax, time_axis, normal_means, normal_stds, 
                     label="Normal", color='blue', alpha=0.2)
        
        # Alarm data
        alarm_means, alarm_stds = compute_mean_std_over_time(alarm_traj[:, :, idx])
        plot_mean_std(ax, time_axis, alarm_means, alarm_stds, 
                     label="Alarm", color='red', alpha=0.2)
        
        ax.set_title(vital)
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Value")
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.savefig('../data/vital_signs_mean_std.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)

def main():
    """Main execution function."""
    # Process data
    normal_data, alarm_data = preprocess_alarm_data(
        csv_path="../data/vital_alarm_sample_ver2.csv",
        time_size=TIME_SIZE,
        reward_function=reward_function
    )
    
    print('normal_data.shape:', normal_data.shape)
    print('alarm_data.shape:', alarm_data.shape)
    
    # Save processed data
    with open("../data/pivot_df_normal.pkl", "wb") as f:
        pickle.dump(normal_data, f)
    with open("../data/pivot_df_alarm.pkl", "wb") as f:
        pickle.dump(alarm_data, f)
    
    # Generate visualizations
    plot_vital_comparisons(normal_data, alarm_data)

if __name__ == "__main__":
    main()