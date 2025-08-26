import pandas as pd, numpy as np

df = pd.read_parquet("C:/Users/Rahul/CollegeProject/UrbanEyeML/urbeye-ml/outputs/chips_index_s2_test.parquet")
print(len(df), "chips")

if len(df) == 0:
    print("No chips found! Re-run make_chips_s2_pairs.py with relaxed mask or check AOI overlap.")
else:
    r = df.iloc[0]
    a0 = np.load(r.t0_npy); a1 = np.load(r.t1_npy)
    print(a0.shape, a1.shape)
