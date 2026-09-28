from torch.utils.data import Dataset
import pandas as pd
import torch
from pathlib import Path
import numpy as np

device = "cuda:0" if torch.cuda.is_available() else "cpu"

smi_char = {'<MASK>': 0, 'C': 1, ')': 2, '(': 3, 'c': 4, 'O': 5, ']': 6, '[': 7,
            '@': 8, '1': 9, '=': 10, 'H': 11, 'N': 12, '2': 13, 'n': 14,
            '3': 15, 'o': 16, '+': 17, '-': 18, 'S': 19, 'F': 20, 'p': 21,
            'l': 22, '/': 23, '4': 24, '#': 25, 'B': 26, '\\': 27, '5': 28,
            'r': 29, 's': 30, '6': 31, 'I': 32, '7': 33, '%': 34, '8': 35,
            'e': 36, 'P': 37, '9': 38, 'R': 39, 'u': 40, '0': 41, 'i': 42,
            '.': 43, 'A': 44, 't': 45, 'h': 46, 'V': 47, 'g': 48, 'b': 49,
            'Z': 50, 'T': 51, 'M': 52}
portein_char = {'<MASK>': 0, 'A': 1, 'C': 2, 'D': 3, 'E': 4,
                'F': 5, 'G': 6, 'H': 7, 'K': 8,
                'I': 9, 'L': 10, 'M': 11, 'N': 12,
                'P': 13, 'Q': 14, 'R': 15, 'S': 16,
                'T': 17, 'V': 18, 'Y': 19, 'W': 20, 'X': 0}


def position_seq(seq, position):
    """生成 pocket 序列（非口袋位置填 <MASK>）"""
    res = ['<MASK>'] * len(seq)
    for i in position:
        res[i - 1] = seq[i - 1]
    return res


def label_smiles(line, max_smi_len):
    label = np.zeros(max_smi_len)
    for i, lab in enumerate(line[:max_smi_len]):
        label[i] = smi_char[lab]
    return label


def label_seq(line, max_seq_len):
    label = np.zeros(max_seq_len)
    for i, lab in enumerate(line[:max_seq_len]):
        label[i] = portein_char[lab]
    return label


def extract_pocket_boundaries(position_list, max_seq_len):

    if not position_list:
        return 0.0, 0.0

    positions = np.array(position_list)
    start_pos = positions.min()
    end_pos = positions.max()
    start_pos_0based = start_pos - 1
    end_pos_0based = end_pos - 1
    start_norm = start_pos_0based / max_seq_len
    end_norm = end_pos_0based / max_seq_len
    return start_norm, end_norm


class MyDataset(Dataset):
    def __init__(self, type, data_path, max_seq_len, max_smi_len):
        super().__init__()
        data_path = Path(data_path)
        self.data_path = data_path
        self.max_seq_len = max_seq_len
        self.max_smi_len = max_smi_len
        self.type = type


        affinity_path = data_path / 'affinity.csv'
        affinity_data = pd.read_csv(affinity_path)


        if affinity_data.columns[0].startswith('Unnamed'):
            affinity_data = affinity_data.rename(columns={affinity_data.columns[0]: 'row_id'})


        cols = list(affinity_data.columns)
        name_col = None
        aff_col = None
        for c in cols:
            cl = str(c).strip().lower()
            if cl in ['pdbname', 'pdbid', 'pdb', 'id']:
                name_col = c
            if cl in ['affinity', '-logkd/ki', 'logkd']:
                aff_col = c


        if name_col is None:
            name_col = cols[1] if len(cols) >= 3 else cols[0]
        if aff_col is None:
            aff_col = cols[2] if len(cols) >= 3 else cols[1]

        affinity = {}
        for _, row in affinity_data.iterrows():
            key = str(row[name_col]).strip().lower()
            affinity[key] = float(row[aff_col])
        self.affinity = affinity


        seq_path = data_path / f'seq_data_{type}.csv'
        seq_data = pd.read_csv(seq_path)


        if seq_data.columns[0].startswith('Unnamed'):
            seq_data = seq_data.rename(columns={seq_data.columns[0]: 'row_id'})

        smile = {}
        sequence = {}
        position = {}
        pocket_start = {}
        pocket_end = {}
        idx = {}
        i = 0
        for _, row in seq_data.iterrows():

            cid = str(row['PDBname']).strip().lower()
            idx[i] = cid
            smile[cid] = row['Smile']
            sequence[cid] = row['Sequence']
            position_list = eval(row['Position']) if isinstance(row['Position'], str) else row['Position']
            position[cid] = position_list

            start_norm, end_norm = extract_pocket_boundaries(position_list, max_seq_len)
            pocket_start[cid] = start_norm
            pocket_end[cid] = end_norm
            i += 1

        self.position = position
        self.smile = smile
        self.sequence = sequence
        self.pocket_start = pocket_start
        self.pocket_end = pocket_end
        self.idx = idx
        assert len(sequence) == len(smile)
        self.len = len(self.sequence)

    def __getitem__(self, index):
        global device
        id_name = self.idx[index]
        pos = self.position[id_name]

        smi = self.smile[id_name]
        seq = self.sequence[id_name]
        pocket = position_seq(seq, pos)

        smi_encode = torch.tensor(label_smiles(smi, self.max_smi_len), device=device).long()
        seq_encode = torch.tensor(label_seq(seq, self.max_seq_len), device=device).long()
        pocket_encode = torch.tensor(label_seq(pocket, self.max_seq_len), device=device).long()


        aff_key = id_name.lower()
        aff_val = self.affinity.get(aff_key, self.affinity.get(id_name, 0.0))
        affinity = torch.tensor(np.array(aff_val, dtype=np.float32), device=device)

        pocket_start = torch.tensor(np.array(self.pocket_start[id_name], dtype=np.float32), device=device)
        pocket_end = torch.tensor(np.array(self.pocket_end[id_name], dtype=np.float32), device=device)

        return id_name, smi_encode, seq_encode, pocket_encode, affinity, pocket_start, pocket_end

    def __len__(self):
        return self.len