import csv
import numpy as np
from src.tt_op import *
from tqdm import tqdm


def data_to_tt(file: str, column_idx):
    tt_train = [np.array([0, 0]).reshape(1, 2, 1)]
    with open(file, 'r') as file:
        csvreader = csv.reader(file)
        csvreader.__next__()
        counter = 0
        for row in tqdm(csvreader):
            multi_idx = bin(csvreader.line_num-2)[2:]
            tt_row = [np.array([1-int(i), int(i)]).reshape(1, 2, 1) for i in reversed(multi_idx)]
            tt_row = tt_scale(float(row[column_idx + 1]), tt_row)
            n = len(multi_idx)
            if n > len(tt_train):
                tt_train += [np.array([1, 0]).reshape(1, 2, 1)]
            tt_train = tt_add(tt_train, tt_row)
            if counter % 5 == 0:
                tt_train = tt_rank_reduce(tt_train)
            counter += 1
    tt_train = tt_scale(2, tt_add(tt_train, tt_scale(-0.5, tt_one(len(tt_train)))))
    return tt_walsh_op_inv(tt_rank_reduce(tt_train)), counter
