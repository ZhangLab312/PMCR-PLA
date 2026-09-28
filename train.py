from torch.utils.data import DataLoader
import torch
import numpy as np
from torch import nn, optim
from model2 import MyModule, PocketPositionLoss
from Dataset1 import MyDataset
import sklearn.metrics as m
from tqdm import tqdm
import os
import pandas as pd


def RMSE(y_true, y_pred):
    return np.sqrt(m.mean_squared_error(y_true, y_pred))


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def calculate_position_metrics(pred_starts, pred_ends, true_starts, true_ends, max_seq_len):
    """计算位置预测的评估指标"""
    pred_starts_abs = pred_starts * max_seq_len
    pred_ends_abs = pred_ends * max_seq_len
    true_starts_abs = true_starts * max_seq_len
    true_ends_abs = true_ends * max_seq_len

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
        'coverage': coverage
    }


# ==================== 全局配置 ====================
device = "cuda:0" if torch.cuda.is_available() else "cpu"
batch_size = 8
epochs = 50
path = os.path.abspath(os.path.dirname(os.getcwd()))
data_path = path + r'/data'

max_seq_len = 1024
max_smi_len = 256

# 5 个随机种子
SEEDS = [3407, 12345, 56789, 98765, 43210]

# 9:1 划分比例
TRAIN_RATIO = 0.9

# 模型保存目录
SAVE_DIR = './model'
os.makedirs(SAVE_DIR, exist_ok=True)


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)


# ==================== 主流程 ====================
if __name__ == '__main__':
    train_dataset = MyDataset('kfold', data_path, max_seq_len, max_smi_len)
    n_total = len(train_dataset)
    print(f"数据集总样本数: {n_total}")

    seed_results = []

    for seed in SEEDS:
        print(f"\n{'=' * 60}")
        print(f"Seed {seed}  (9:1 随机划分)")
        print(f"{'=' * 60}")
        set_seed(seed)


        rng = np.random.RandomState(seed)
        indices = rng.permutation(n_total)
        n_train = int(TRAIN_RATIO * n_total)
        train_index = indices[:n_train]
        val_index = indices[n_train:]

        print(f"train: {len(train_index)}  val: {len(val_index)}")

        train_subset = torch.utils.data.Subset(train_dataset, train_index)
        val_subset = torch.utils.data.Subset(train_dataset, val_index)

        train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        # 重新初始化模型
        model = MyModule(max_seq_len=max_seq_len).to(device)
        initialize_weights(model)

        optimizer = optim.Adam(model.parameters(), lr=0.00005, weight_decay=1e-5)
        criterion_affinity = nn.MSELoss(reduction="mean")
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=2, verbose=True
        )

        best_val_loss = float('inf')
        best_rmse = float('inf')
        best_iou = 0
        best_dice = 0
        best_coverage = 0

        early_stopping_rounds = 5
        rounds_without_improvement = 0

        best_model_path = f'{SAVE_DIR}/seed_{seed}_best_model.ckpt'

        for epoch in range(epochs):
            model.train()
            total_train_loss = 0
            total_train_affinity_loss = 0
            total_train_pocket_loss = 0

            for id_name, smi_encode, seq_encode, pocket_encode, affinity, pocket_start, pocket_end in tqdm(
                    train_dataloader, desc=f"Seed {seed} Epoch {epoch + 1}"):

                smi_encode = smi_encode.to(device)
                seq_encode = seq_encode.to(device)
                affinity = affinity.to(device)
                pocket_start = pocket_start.to(device)
                pocket_end = pocket_end.to(device)

                affinity_pred, pocket_start_pred, pocket_end_pred = model(seq_encode, smi_encode)

                affinity_pred_flat = affinity_pred.flatten()
                affinity_flat = affinity.flatten()

                loss_affinity = criterion_affinity(affinity_pred_flat, affinity_flat)
                loss_pocket_total, start_loss, end_loss, length_loss, coverage_loss, size_penalty_loss = \
                    model.pocket_loss_fn(
                        pocket_start_pred.flatten(), pocket_end_pred.flatten(),
                        pocket_start.flatten(), pocket_end.flatten()
                    )

                loss = loss_affinity + 0.75 * loss_pocket_total

                total_train_loss += loss.item()
                total_train_affinity_loss += loss_affinity.item()
                total_train_pocket_loss += loss_pocket_total.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_loss = total_train_loss / len(train_dataloader)
            train_affinity_loss = total_train_affinity_loss / len(train_dataloader)
            train_pocket_loss = total_train_pocket_loss / len(train_dataloader)

            model.eval()
            val_loss = 0
            val_affinity_loss = 0
            val_pocket_loss = 0

            affinity_pre = []
            affinity_target = []
            pocket_start_pre = []
            pocket_end_pre = []
            pocket_start_target = []
            pocket_end_target = []

            with torch.no_grad():
                for id_name, smi_encode, seq_encode, pocket_encode, affinity, pocket_start, pocket_end in val_dataloader:
                    smi_encode = smi_encode.to(device)
                    seq_encode = seq_encode.to(device)
                    affinity = affinity.to(device)
                    pocket_start = pocket_start.to(device)
                    pocket_end = pocket_end.to(device)

                    affinity_pred, pocket_start_pred, pocket_end_pred = model(seq_encode, smi_encode)

                    affinity_pred_flat = affinity_pred.flatten()
                    affinity_flat = affinity.flatten()

                    loss_affinity = criterion_affinity(affinity_pred_flat, affinity_flat)
                    loss_pocket_total, start_loss, end_loss, length_loss, coverage_loss, size_penalty_loss = \
                        model.pocket_loss_fn(
                            pocket_start_pred.flatten(), pocket_end_pred.flatten(),
                            pocket_start.flatten(), pocket_end.flatten()
                        )
                    loss = loss_affinity + 0.75 * loss_pocket_total

                    val_loss += loss.item()
                    val_affinity_loss += loss_affinity.item()
                    val_pocket_loss += loss_pocket_total.item()

                    affinity_pre.extend(affinity_pred.cpu().numpy().reshape(-1))
                    affinity_target.extend(affinity.cpu().numpy().reshape(-1))

                    pocket_start_pre.extend(pocket_start_pred.cpu().numpy())
                    pocket_end_pre.extend(pocket_end_pred.cpu().numpy())
                    pocket_start_target.extend(pocket_start.cpu().numpy())
                    pocket_end_target.extend(pocket_end.cpu().numpy())

            val_loss /= len(val_dataloader)
            val_affinity_loss /= len(val_dataloader)
            val_pocket_loss /= len(val_dataloader)

            rmse = RMSE(affinity_pre, affinity_target)

            position_metrics = calculate_position_metrics(
                np.array(pocket_start_pre), np.array(pocket_end_pre),
                np.array(pocket_start_target), np.array(pocket_end_target),
                max_seq_len
            )

            scheduler.step(val_loss)


            print(f'\nSeed {seed} - Epoch {epoch + 1}:')
            print(f'Train Loss: {train_loss:.4f} (Affinity: {train_affinity_loss:.4f}, Pocket: {train_pocket_loss:.4f})')
            print(f'Val Loss: {val_loss:.4f} (Affinity: {val_affinity_loss:.4f}, Pocket: {val_pocket_loss:.4f})')
            print(f'RMSE: {rmse:.4f}')
            print(f'Position Metrics:')
            print(f'  MAE - Start: {position_metrics["start_mae"]:.2f}, End: {position_metrics["end_mae"]:.2f}, '
                  f'Length: {position_metrics["length_mae"]:.2f}, Center: {position_metrics["center_mae"]:.2f}')
            print(f'  Overlap Metrics - IoU: {position_metrics["iou"]:.4f}, '
                  f'Dice: {position_metrics["dice"]:.4f}, Coverage: {position_metrics["coverage"]:.4f}')


        seed_results.append({
            'seed': seed,
            'n_train': len(train_index),
            'n_val': len(val_index),
            'best_val_loss': best_val_loss,
            'best_rmse': best_rmse,
            'best_iou': best_iou,
            'best_dice': best_dice,
            'best_coverage': best_coverage,
            'model_path': best_model_path,
        })

        print(f'\nSeed {seed} completed:')
        print(f'Best Val Loss: {best_val_loss:.4f}')
        print(f'Best RMSE: {best_rmse:.4f}')
        print(f'Best Overlap Metrics - IoU: {best_iou:.4f}, Dice: {best_dice:.4f}, Coverage: {best_coverage:.4f}')

        del model
        torch.cuda.empty_cache()

    # ==================== 汇总 ====================
    print("\n" + "=" * 60)
    print("SUMMARY OF ALL SEEDS")
    print("=" * 60)
    for r in seed_results:
        print(f"Seed {r['seed']}: Val Loss = {r['best_val_loss']:.4f}, "
              f"RMSE = {r['best_rmse']:.4f}, IoU = {r['best_iou']:.4f}, "
              f"Dice = {r['best_dice']:.4f}, Coverage = {r['best_coverage']:.4f}")

    avg_val_loss = np.mean([r['best_val_loss'] for r in seed_results])
    avg_rmse = np.mean([r['best_rmse'] for r in seed_results])
    avg_iou = np.mean([r['best_iou'] for r in seed_results])
    avg_dice = np.mean([r['best_dice'] for r in seed_results])
    avg_coverage = np.mean([r['best_coverage'] for r in seed_results])

    std_val_loss = np.std([r['best_val_loss'] for r in seed_results])
    std_rmse = np.std([r['best_rmse'] for r in seed_results])
    std_iou = np.std([r['best_iou'] for r in seed_results])
    std_dice = np.std([r['best_dice'] for r in seed_results])
    std_coverage = np.std([r['best_coverage'] for r in seed_results])

    print(f"\nAverage Performance (Mean ± Std over 5 seeds):")
    print(f"Val Loss: {avg_val_loss:.4f} ± {std_val_loss:.4f}")
    print(f"RMSE:     {avg_rmse:.4f} ± {std_rmse:.4f}")
    print(f"IoU:      {avg_iou:.4f} ± {std_iou:.4f}")
    print(f"Dice:     {avg_dice:.4f} ± {std_dice:.4f}")
    print(f"Coverage: {avg_coverage:.4f} ± {std_coverage:.4f}")

    # 保存汇总 CSV
    summary_df = pd.DataFrame(seed_results)
    csv_path = f'{SAVE_DIR}/summary_5seeds_9to1.csv'
    summary_df.to_csv(csv_path, index=False)
    print(f"\n汇总已保存: {csv_path}")