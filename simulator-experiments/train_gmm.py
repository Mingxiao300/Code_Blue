import logging
import os
from typing import Tuple

import hydra
import numpy as np
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from sklearn.mixture import GaussianMixture

logger = logging.getLogger(__name__)


def filter_outliers(df):
    df = df[~((df["name"] == "PULSE_RATE") & (df["doublevalue"] > 300))]
    df = df[~((df["name"] == "RESPIRATORY_RATE") & (df["doublevalue"] > 60))]
    df = df[
        ~(
            (df["name"] == "SPO2")
            & ((df["doublevalue"] < 50) | (df["doublevalue"] > 100))
        )
    ]
    df = df[~((df["name"] == "COVERED_SKIN_TEMPERATURE") & (df["doublevalue"] > 45))]
    return df


def min_max_normalize(column):
    return (column - column.min()) / (column.max() - column.min())


def reverse_min_max_normalize(column, min_val, max_val):
    return column * (max_val - min_val) + min_val


def prepare_gmm_data(pivot_df, num_timesteps, time_size, vital_signs, min_entries=1):
    input_data = []
    output_data = []
    valid_patient_ids = set()
    # patient_rewards = {}

    for patient_id, group in pivot_df.groupby("patient_id"):
        group = group.sort_values("generatedat")
        valid_sequences = []
        # rewards = []

        for i in range(len(group) - num_timesteps):
            valid_sequence = True
            base_time = group.iloc[i]["generatedat"]
            for j in range(1, num_timesteps + 1):
                expected_time = base_time + pd.Timedelta(minutes=time_size * j)
                if group.iloc[i + j]["generatedat"] != expected_time:
                    valid_sequence = False
                    break
            if valid_sequence:
                input_values = group.iloc[i : i + num_timesteps][
                    vital_signs
                ].values.flatten()
                output_values = group.iloc[i + num_timesteps][vital_signs].values
                valid_sequences.append(
                    (input_values.astype(float), output_values.astype(float))
                )
                # rewards.append(group.iloc[i + num_timesteps]["reward"])

        # patient_rewards[patient_id] = np.mean(rewards)
        if len(valid_sequences) > min_entries:
            valid_patient_ids.add(patient_id)
            for input_values, output_values in valid_sequences:
                input_data.append(input_values)
                output_data.append(output_values)

    return np.hstack((np.array(input_data), np.array(output_data)))

def split_alarm_periods(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into alarm and normal periods."""
    # Ensure datetime conversion
    if 'alarm_vital_boundary_start_time' in df.columns:
        df['alarm_vital_boundary_start_time'] = pd.to_datetime(
            df['alarm_vital_boundary_start_time'], 
            errors='coerce'
        )
    
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
            if pd.isna(alarm_start):  # Handle NaT
                continue
            alarm_end = alarm_start + pd.Timedelta(hours=24)
            if alarm_start <= gen_time <= alarm_end:
                return True
        return False

    df['in_alarm_period'] = (~df['alarm_id'].isna()) | df.apply(is_in_alarm_window, axis=1)
    df_alarm = df[df['in_alarm_period']].copy()
    df_normal = df[~df['in_alarm_period']].copy()
    
    return df_alarm, df_normal

def process_data(df, time_size, vital_signs):
    """Process and pivot dataframe with proper datetime handling."""
    # Ensure datetime index exists
    df = df[df['name'].isin(vital_signs)]
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index('generatedat')
    
    # Convert index to DatetimeIndex if not already
    df.index = pd.to_datetime(df.index)
    
    # Group and resample
    resampled_df = (
        df.reset_index()  # Temporarily bring generatedat back as column
        .groupby(["patient_id", "name"])
        .resample(f"{time_size}min", on='generatedat')["doublevalue"]
        .median()
        .dropna()
        .reset_index()
    )
    
    # Pivot the results
    pivot_df = (
        resampled_df.pivot_table(
            index=["patient_id", "generatedat"], 
            columns="name", 
            values="doublevalue"
        )
        .dropna()
        .reset_index()
    )
    
    return pivot_df

def train_gmm_model(data, num_components, vital_signs):
    """Train and return a GMM model."""
    gmm = GaussianMixture(
        n_components=num_components, 
        covariance_type="full", 
        random_state=42
    )
    gmm.fit(data)
    return gmm

def save_gmm_model(gmm, min_max, vital_signs, model_type):
    """Save GMM model parameters."""
    os.makedirs("models/", exist_ok=True)
    model_save_path = f"models/gmm_{model_type}.npz"
    scaler_min = np.array([min_max[k][0] for k in vital_signs])
    scaler_max = np.array([min_max[k][1] for k in vital_signs])
    
    np.savez(
        model_save_path,
        means=gmm.means_,
        covariances=gmm.covariances_,
        weights=gmm.weights_,
        scaler_min=scaler_min,
        scaler_max=scaler_max,
        names=np.array(vital_signs, dtype=str),
    )
    logger.info(f"GMM {model_type} parameters saved to %s", model_save_path)

@hydra.main(config_path="conf", config_name="train_gmm", version_base=None)
def main(cfg: DictConfig):
    """Main training function."""
    logger.info("Loading and preprocessing the data")
    df = (
        pd.read_csv(cfg.dataset.path)
        .assign(
            generatedat=lambda x: pd.to_datetime(x["generatedat"], errors="coerce"),
            doublevalue=lambda x: pd.to_numeric(x["doublevalue"], errors="coerce"),
        )
        .dropna(subset=["doublevalue", "patient_id"])
        .pipe(filter_outliers)
    )

    # Split into normal and alarm data
    logger.info("Splitting data into normal and alarm periods")
    df_alarm, df_normal = split_alarm_periods(df)

    # Process both datasets
    logger.info("Processing normal data")
    pivot_normal = process_data(df_normal, cfg.dataset.time_size, cfg.dataset.vital_signs)
    
    logger.info("Processing alarm data")
    pivot_alarm = process_data(df_alarm, cfg.dataset.time_size, cfg.dataset.vital_signs)

    # Normalize data and store min/max values
    min_max_normal = {}
    min_max_alarm = {}
    for sign in cfg.dataset.vital_signs:
        # Use global min/max from combined data for consistent normalization
        min_max_normal[sign] = [pivot_normal[sign].min(), pivot_normal[sign].max()]
        min_max_alarm[sign] = [pivot_alarm[sign].min(), pivot_alarm[sign].max()]
        pivot_normal[sign] = (pivot_normal[sign] - pivot_normal[sign].min()) / (pivot_normal[sign].max() - pivot_normal[sign].min())
        pivot_alarm[sign] = (pivot_alarm[sign] - pivot_alarm[sign].min()) / (pivot_alarm[sign].max() - pivot_alarm[sign].min())
        
    # Prepare GMM training data
    logger.info("Preparing GMM training data")
    gmm_normal_data = prepare_gmm_data(
        pivot_normal,
        cfg.dataset.num_timesteps,
        cfg.dataset.time_size,
        cfg.dataset.vital_signs,
        cfg.dataset.min_entries,
    )
    
    gmm_alarm_data = prepare_gmm_data(
        pivot_alarm,
        cfg.dataset.num_timesteps,
        cfg.dataset.time_size,
        cfg.dataset.vital_signs,
        cfg.dataset.min_entries,
    )
    
    # Train separate GMMs
    logger.info("Training Normal GMM model")
    gmm_normal = train_gmm_model(gmm_normal_data, cfg.dataset.num_comp, cfg.dataset.vital_signs)
    
    logger.info("Training Alarm GMM model")
    gmm_alarm = train_gmm_model(gmm_alarm_data, cfg.dataset.num_comp, cfg.dataset.vital_signs)

    # Save models
    save_gmm_model(gmm_normal, min_max_normal, cfg.dataset.vital_signs, "normal")
    save_gmm_model(gmm_alarm, min_max_alarm, cfg.dataset.vital_signs, "alarm")

if __name__ == "__main__":
    main()