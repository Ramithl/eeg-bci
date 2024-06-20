import pandas as pd
import numpy as np

file = "s1s2_MI_100LP_Resampled.csv"
eeg = pd.read_csv(file).to_numpy()
header = {0: 'Trial', 1: 'Class'}
set_size = 500

reshaped_arr = eeg.reshape(-1, set_size, eeg.shape[1])

np.random.shuffle(reshaped_arr)

train = reshaped_arr[:90]
test = reshaped_arr[90:]

train_ds = train.reshape(-1, 32)
test_ds = test.reshape(-1, 32)

# Reshape back to the original shape
shuffled_arr = reshaped_arr.reshape(eeg.shape)
print(train_ds.shape)

eeg_df_train = pd.DataFrame(train_ds).rename(columns=header)

eeg_df_train.to_csv("s1s2_MI_100LP_Resampled_TRAIN.csv", index=False)

eeg_df_test = pd.DataFrame(test_ds).rename(columns=header)

eeg_df_test.to_csv("s1s2_MI_100LP_Resampled_TEST.csv", index=False)