from torch.utils.data import DataLoader
import torch
import numpy as np
from torch import nn
from model2 import MyModule
from Dataset1 import MyDataset
import sklearn.metrics as m
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
import pandas as pd
import os
import warnings

warnings.filterwarnings('ignore')


def RMSE(y_true, y_pred):
    return np.sqrt(m.mean_squared_error(y_true, y_pred))


def MAE(y_true, y_pred):
    return m.mean_absolute_error(y_true, y_pred)


def MSE(y_true, y_pred):
    return m.mean_squared_error(y_true, y_pred)


def CORR(y_true, y_pred):
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan
    return pearsonr(y_true, y_pred)[0]


def SD(y_true, y_pred):
    """Standard deviation after linear calibration, consistent with training reports."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) < 2:
        return np.nan
    lr = LinearRegression().fit(y_pred.reshape(-1, 1), y_true)
    calibrated = lr.predict(y_pred.reshape(-1, 1))
    return np.sqrt(np.square(y_true - calibrated).sum() / (len(y_true) - 1))


def c_index(y_true, y_pred):
    """Concordance index."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    concordant = 0.0
    comparable = 0
    for i in range(1, len(y_true)):
        for j in range(i):
            if y_true[i] == y_true[j]:
                continue
            comparable += 1
            true_order = y_true[i] > y_true[j]
            pred_order = y_pred[i] > y_pred[j]
            if true_order == pred_order:
                concordant += 1.0
            elif y_pred[i] == y_pred[j]:
                concordant += 0.5
    return concordant / comparable if comparable else np.nan


def calculate_position_metrics(pred_starts, pred_ends, true_starts, true_ends, max_seq_len):
    """Position metrics using the same interval definitions as the training code."""
    pred_starts_abs = np.asarray(pred_starts) * max_seq_len
    pred_ends_abs = np.asarray(pred_ends) * max_seq_len
    true_starts_abs = np.asarray(true_starts) * max_seq_len
    true_ends_abs = np.asarray(true_ends) * max_seq_len

    start_mae = np.mean(np.abs(pred_starts_abs - true_starts_abs))
    end_mae = np.mean(np.abs(pred_ends_abs - true_ends_abs))

    pred_lengths = pred_ends_abs - pred_starts_abs
    true_lengths = true_ends_abs - true_starts_abs
    length_mae = np.mean(np.abs(pred_lengths - true_lengths))

    overlap_starts = np.maximum(pred_starts_abs, true_starts_abs)
    overlap_ends = np.minimum(pred_ends_abs, true_ends_abs)
    overlap_lengths = np.maximum(overlap_ends - overlap_starts, 0)
    union_lengths = pred_lengths + true_lengths - overlap_lengths

    iou = np.mean(overlap_lengths / (union_lengths + 1e-7))
    dice = np.mean(2 * overlap_lengths / (pred_lengths + true_lengths + 1e-7))
    coverage = np.mean(overlap_lengths / (true_lengths + 1e-7))

    pred_centers = (pred_starts_abs + pred_ends_abs) / 2
    true_centers = (true_starts_abs + true_ends_abs) / 2
    center_mae = np.mean(np.abs(pred_centers - true_centers))

    return {
        'start_mae': start_mae,
        'end_mae': end_mae,
        'length_mae': length_mae,
        'center_mae': center_mae,
        'iou': iou,
        'dice': dice,
        'coverage': coverage,
    }


def evaluate_model(model, dataloader, device, max_seq_len):
    model.eval()
    affinity_predictions, affinity_targets = [], []
    start_predictions, end_predictions = [], []
    start_targets, end_targets = [], []
    sample_ids = []

    with torch.no_grad():
        for id_name, smi_encode, seq_encode, pocket_encode, affinity, pocket_start, pocket_end in dataloader:
            affinity_pred, start_pred, end_pred = model(
                seq_encode.to(device), smi_encode.to(device)
            )

            affinity_predictions.extend(affinity_pred.detach().cpu().numpy().reshape(-1))
            start_predictions.extend(start_pred.detach().cpu().numpy().reshape(-1))
            end_predictions.extend(end_pred.detach().cpu().numpy().reshape(-1))
            affinity_targets.extend(affinity.numpy().reshape(-1))
            start_targets.extend(pocket_start.numpy().reshape(-1))
            end_targets.extend(pocket_end.numpy().reshape(-1))
            sample_ids.extend(list(id_name))

    affinity_predictions = np.asarray(affinity_predictions)
    affinity_targets = np.asarray(affinity_targets)
    start_predictions = np.asarray(start_predictions)
    end_predictions = np.asarray(end_predictions)
    start_targets = np.asarray(start_targets)
    end_targets = np.asarray(end_targets)

    affinity_metrics = {
        'rmse': RMSE(affinity_targets, affinity_predictions),
        'mae': MAE(affinity_targets, affinity_predictions),
        'mse': MSE(affinity_targets, affinity_predictions),
        'sd': SD(affinity_targets, affinity_predictions),
        'r': CORR(affinity_targets, affinity_predictions),
        'ci': c_index(affinity_targets, affinity_predictions),
    }
    position_metrics = calculate_position_metrics(
        start_predictions, end_predictions,
        start_targets, end_targets, max_seq_len
    )

    return {
        'ids': sample_ids,
        'affinity_predictions': affinity_predictions,
        'start_predictions': start_predictions,
        'end_predictions': end_predictions,
        'affinity_targets': affinity_targets,
        'start_targets': start_targets,
        'end_targets': end_targets,
        'affinity_metrics': affinity_metrics,
        'position_metrics': position_metrics,
    }


# The following values must match the training script.
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
batch_size = 8
max_seq_len = 1024
max_smi_len = 256
SEEDS = [3407, 12345, 56789, 98765, 43210]
TEST_SPLIT = 'core2016'

path = os.path.abspath(os.path.dirname(os.getcwd()))
data_path = path + r'/data'
model_dir = './model'
result_dir = './test_results'
os.makedirs(result_dir, exist_ok=True)


if __name__ == '__main__':
    test_dataset = MyDataset(TEST_SPLIT, data_path, max_seq_len, max_smi_len)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(f'Test split: {TEST_SPLIT}; samples: {len(test_dataset)}')

    seed_results = []

    for seed in SEEDS:
        model_path = os.path.join(model_dir, f'seed_{seed}_best_model.ckpt')
        if not os.path.exists(model_path):
            print(f'Warning: checkpoint not found, skipping seed {seed}: {model_path}')
            continue

        model = MyModule(max_seq_len=max_seq_len).to(device)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)

        result = evaluate_model(model, test_dataloader, device, max_seq_len)
        result['seed'] = seed
        seed_results.append(result)

        metrics = result['affinity_metrics']
        pos = result['position_metrics']
        print(f'\nSeed {seed}:')
        print(f"  Affinity: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, "
              f"SD={metrics['sd']:.4f}, R={metrics['r']:.4f}, CI={metrics['ci']:.4f}")
        print(f"  Position: IoU={pos['iou']:.4f}, Dice={pos['dice']:.4f}, "
              f"Coverage={pos['coverage']:.4f}, Start MAE={pos['start_mae']:.2f}, "
              f"End MAE={pos['end_mae']:.2f}")

        pd.DataFrame({
            'id_name': result['ids'],
            'true_affinity': result['affinity_targets'],
            'predict_affinity': result['affinity_predictions'],
            'true_start': result['start_targets'],
            'predict_start': result['start_predictions'],
            'true_end': result['end_targets'],
            'predict_end': result['end_predictions'],
        }).to_csv(os.path.join(result_dir, f'seed_{seed}_test_result.csv'), index=False)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not seed_results:
        raise RuntimeError('No matching checkpoints were found.')

    # Per-seed mean and standard deviation.
    affinity_metric_names = ['rmse', 'mae', 'mse', 'sd', 'r', 'ci']
    position_metric_names = ['start_mae', 'end_mae', 'length_mae', 'center_mae', 'iou', 'dice', 'coverage']
    summary_rows = []
    for result in seed_results:
        row = {'seed': result['seed']}
        row.update({name: result['affinity_metrics'][name] for name in affinity_metric_names})
        row.update({name: result['position_metrics'][name] for name in position_metric_names})
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(result_dir, f'{TEST_SPLIT}_per_seed_summary.csv'), index=False)

    print('\nMean ± standard deviation across available seeds:')
    for name in affinity_metric_names + position_metric_names:
        print(f'{name}: {summary_df[name].mean():.4f} ± {summary_df[name].std(ddof=0):.4f}')

    # Average predictions from the five independently trained seed models.
    ensemble_affinity = np.mean([r['affinity_predictions'] for r in seed_results], axis=0)
    ensemble_start = np.mean([r['start_predictions'] for r in seed_results], axis=0)
    ensemble_end = np.mean([r['end_predictions'] for r in seed_results], axis=0)
    targets = seed_results[0]

    ensemble_affinity_metrics = {
        'rmse': RMSE(targets['affinity_targets'], ensemble_affinity),
        'mae': MAE(targets['affinity_targets'], ensemble_affinity),
        'mse': MSE(targets['affinity_targets'], ensemble_affinity),
        'sd': SD(targets['affinity_targets'], ensemble_affinity),
        'r': CORR(targets['affinity_targets'], ensemble_affinity),
        'ci': c_index(targets['affinity_targets'], ensemble_affinity),
    }
    ensemble_position_metrics = calculate_position_metrics(
        ensemble_start, ensemble_end,
        targets['start_targets'], targets['end_targets'], max_seq_len
    )

    print('\nEnsemble of available seed models:')
    print(ensemble_affinity_metrics)
    print(ensemble_position_metrics)

    pd.DataFrame({
        'id_name': targets['ids'],
        'true_affinity': targets['affinity_targets'],
        'ensemble_predict_affinity': ensemble_affinity,
        'true_start': targets['start_targets'],
        'ensemble_predict_start': ensemble_start,
        'true_end': targets['end_targets'],
        'ensemble_predict_end': ensemble_end,
    }).to_csv(os.path.join(result_dir, f'{TEST_SPLIT}_seed_ensemble_result.csv'), index=False)
