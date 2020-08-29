# coding: utf-8
import os
import re
import numpy as np
import pandas as pd
import json

patient_ids = []

for filename in os.listdir('./raw'):
    # the patient data in PhysioNet contains 6-digits
    match = re.search('\d{6}', filename)
    if match:
        id_ = match.group()
        patient_ids.append(id_)

out = pd.read_csv('./raw/Outcomes-a.txt').set_index('RecordID')['In-hospital_death']

# we select 32 attributes which contains enough non-values
attributes = ['Count_SB_12_14_F', 'Count_SB_12_14_M', 'Count_SB_15_17_F', 'Count_SB_15_17_M', 'Count_SB_18_21_F',
              'Count_SB_18_21_M', 'Count_SB_22_25_F', 'Count_SB_22_25_M', 'Count_SO_12_14_F', 'Count_SO_12_14_M',
              'Count_SO_15_17_F', 'Count_SO_15_17_M','Count_SO_18_21_F', 'Count_SO_18_21_M', 'Count_SO_22_25_F',
              'Count_SO_22_25_M', 'Count_PB_12_14_F', 'Count_PB_12_14_M', 'Count_PB_15_17_F', 'Count_PB_15_17_M',
              'Count_PB_18_21_F', 'Count_PB_18_21_M', 'Count_PB_22_25_F', 'Count_PB_22_25_M', 'Count_PO_12_14_F',
              'Count_PO_12_14_M', 'Count_PO_15_17_F', 'Count_PO_15_17_M', 'Count_PO_18_21_F', 'Count_PO_18_21_M',
              'Count_PO_22_25_F', 'Count_PO_22_25_M']

# mean and std of 32 attributes
mean = np.array([15.733462, 20.382101, 40.754455, 17.385384, 110.835219, 60.082982, 63.085587,
                 36.448417, 54.677560, 71.199158, 98.524251, 59.913393, 198.919923, 105.699935, 119.354486,
                 60.026871, 21.281217, 28.194668, 36.192066, 29.076904, 77.179518, 40.969107, 44.864202, 24.634217,
                 15.144098, 15.585943, 33.367874, 17.383078, 88.188980, 45.809123, 53.371937, 28.930397])

std = np.array([14.73718198, 16.30789303, 34.97983056, 17.71797247, 97.61718022, 51.31159879, 60.07866337,
                33.37581949, 63.34272858, 84.27032439, 98.35693433,	58.51026789, 205.5317975, 104.6786201,
                133.4268325, 66.3258888, 23.98738471, 32.39421896, 36.94895975, 25.0891656, 76.32853259,
                40.54558237, 48.86878893, 25.23397556, 13.45940474, 14.73307624, 28.13056665, 15.22136631,
                76.20051778, 40.2220299, 47.12498547, 25.83551688])

threshold = np.array([34.00, 41.00, 84.00, 39.00, 235.00, 127.00, 137.00, 78.00, 124.00, 164.10, 229.00,
                      139.20, 460.00, 242.00, 262.00, 133.00, 49.00, 66.00, 84.00, 62.00, 172.00, 93.00, 99.00,
                      54.00, 32.00, 34.00, 69.00, 37.00, 185.00, 98.00, 111.10, 62.00])

fs = open('./json/json', 'w')

def to_time_bin(x):
    h, m = map(int, x.split(':'))
    return h

def parse_data(x):
    x = x.set_index('Parameter').to_dict()['Value']

    values = []

    for attr in attributes:
        if x.has_key(attr):
            values.append(x[attr])
        else:
            values.append(np.nan)

    return values

def parse_delta(masks, dir_):
    if dir_ == 'backward':
        masks = masks[::-1]

    deltas = []

    for h in range(48):
        if h == 0:
            deltas.append(np.ones(32))
        else:
            deltas.append(np.ones(32) + (1 - masks[h]) * deltas[-1])

    return np.array(deltas)


def parse_rec(values, masks, evals, eval_masks, dir_):
    deltas = parse_delta(masks, dir_)

    # only used in GRU-D
    forwards = pd.DataFrame(values).fillna(method='ffill').fillna(0.0).as_matrix()

    rec = {}

    rec['values'] = np.nan_to_num(values).tolist()
    rec['masks'] = masks.astype('int32').tolist()
    # imputation ground-truth
    rec['evals'] = np.nan_to_num(evals).tolist()
    rec['eval_masks'] = eval_masks.astype('int32').tolist()
    rec['forwards'] = forwards.tolist()
    rec['deltas'] = deltas.tolist()

    return rec

def parse_id(id_):
    data = pd.read_csv('./raw/{}.txt'.format(id_))
    # accumulate the records within one hour
    data['Time'] = data['Time'].apply(lambda x: to_time_bin(x))

    evals = []

    # merge all the metrics within one hour
    for h in range(48):

        evals.append(parse_data(data[data['Time'] == h]))

    evals = (np.array(evals) - mean) / std

    shp = evals.shape

    evals = evals.reshape(-1)

    # randomly eliminate 10% values as the imputation ground-truth
    indices = []

    for i in range(len(evals)):

        if not np.isnan([evals[i]]) == True:

            if evals[i] > threshold[i % 32]:

                indices.append(i)

    # indices = np.where(~np.isnan(evals))[0].tolist()
    # indices = np.random.choice(indices, len(indices) // 10)

    values = evals.copy()
    values[indices] = np.nan

    masks = ~np.isnan(values)
    eval_masks = (~np.isnan(values)) ^ (~np.isnan(evals))

    evals = evals.reshape(shp)
    values = values.reshape(shp)

    masks = masks.reshape(shp)
    eval_masks = eval_masks.reshape(shp)

    label = out.loc[int(id_)]

    rec = {'label': int(label)}

    # prepare the model for both directions
    rec['forward'] = parse_rec(values, masks, evals, eval_masks, dir_='forward')
    rec['backward'] = parse_rec(values[::-1], masks[::-1], evals[::-1], eval_masks[::-1], dir_='backward')

    rec = json.dumps(rec)

    fs.write(rec + '\n')


for id_ in patient_ids:
    print('Processing patient {}'.format(id_))
    try:
        parse_id(id_)
    except Exception as e:
        print(e)
        continue

fs.close()