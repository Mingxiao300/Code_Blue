# Simulator Experiments

This repository extends the [rule-bottleneck-reinforcement-learning](https://github.com/mauriciogtec/rule-bottleneck-reinforcement-learning) framework, specifically tailored to experiments with the new simulator. All experiment scripts are stored under the "simulator-experiments" folder.

## Dataset Setup
The dataset used for these experiments is `vital_alarm_sample_ver2.csv`. Before running anything:
1. Create a `data` folder on the same level as the `simulator-experiments` folder
2. Save the dataset file into that folder

## Experiment Scripts

### GMM Training
- `train_gmm.py`
- Replicates the GMM model fitting from the old simulator
- Adapts it to train two GMMs for normal and alarm data separately
- Models can later be triggered with certain heuristics

**Run with:**
```bash
python train_gmm.py
```
**Outputs**
- Generates two model files (`gmm_normal.npz` and `gmm_alarm.npz`) under the `models` folder

### Diffusion Model
- `diffusion.py`
- Contains experimental code to fit a diffusion model with the new dataset. (This code is not as cleaned)
- First run the `data_preprocess.py` file first to generate processed data for the alarm and normal periods
