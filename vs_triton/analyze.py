#!/usr/bin/env python

# %%
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %%
filename = '/tmp/vadd_bench_all/vector-add-performance.csv'
# load into dataframe
df = pd.read_csv(filename)

# %%
fig, ax = plt.subplots(figsize=(8, 6))
cuda_kerns = [col for col in df.columns if col.startswith('cuda')]
for kern in cuda_kerns:
    ax.plot(df['size'], df[kern], color='black', alpha=0.5)
ax.plot(df['size'], df['triton'], color='red', label='triton')
ax.plot(df['size'], df['torch'], color='blue', label='torch')
ax.set_xlabel('Vector Size')
ax.set_ylabel('GB/sec')
# ax.set_yscale('log')
ax.set_xscale('log')
# ax.legend()



# %%
kern_cols = [col for col in df.columns if col != 'size']
for ix, row in df[kern_cols].iterrows():
    # get the max column for this row
    max_col = row.idxmax()
    print(df['size'][ix], max_col, row[max_col])



# %% select down to columns that start with cuda
cuda_columns = [col for col in df.columns if col.startswith('cuda')]
cuda_df = df[cuda_columns]

# %%
for ix, row in cuda_df.iterrows():
    # get the max column for this row
    max_col = row.idxmax()
    print(df['size'][ix], max_col, row[max_col])

# %%
# for each row, get the column with the maximum value in that row
best_kernels = []
best_map = {}
for ix, row in cuda_df.iterrows():
    # get the max value in the row
    max_value = max(row)
    # Get all the columns that are equal to the max value
    cols = [col for col, val in row.items() if val == max_value]
    assert any(c.startswith('cuda') for c in cols)
    best_kernels.extend(cols)
    best_map[int(df['size'][ix])] = sorted(cols)

best_kernels = sorted(set(best_kernels))
len(best_kernels)

# %%
fig, ax = plt.subplots()
# for kern in best_kernels:
#     cuda_line, = ax.plot(df['size'], df[kern], color='k', alpha=0.1)
triton_line, = ax.plot(df['size'], df['triton'], color='y', label='triton', alpha=0.8)
torch_line, = ax.plot(df['size'], df['torch'], color='c', label='torch', alpha=0.8)
best_line, = ax.plot(df['size'], df['cuda_64_1_1'], color='m', label='Best CUDA', alpha=0.8)
# lines = [cuda_line, triton_line, torch_line, best_line]
# labels = ['CUDA', 'Triton', 'Torch', 'Best CUDA']
lines = [triton_line, torch_line, best_line]
labels = ['Triton', 'Torch', 'CUDA']
ax.set_xlabel('Vector Size')
ax.set_ylabel('GB/sec')
# ax.set_yscale('log')
ax.set_xscale('log')
# ax.legend()
ax.legend(lines, labels, loc='upper left')
# title
ax.set_title('Vector Add Kernels: CUDA vs. Triton vs. Torch')

# %%
fig, ax = plt.subplots()
# for kern in best_kernels:
#     cuda_line, = ax.plot(df['size'], df[kern]/df['cuda_64_1_1'], color='k', alpha=0.1)
triton_line, = ax.plot(df['size'], df['triton']/df['cuda_64_1_1'], color='y', label='triton', alpha=0.8)
torch_line, = ax.plot(df['size'], df['torch']/df['cuda_64_1_1'], color='c', label='torch', alpha=0.8)
best_line, = ax.plot(df['size'], df['cuda_64_1_1']/df['cuda_64_1_1'], color='m', label='Best CUDA', alpha=0.8)
lines = [cuda_line, triton_line, torch_line, best_line]
labels = ['CUDA', 'Triton', 'Torch', 'Best CUDA']
lines = [triton_line, torch_line, best_line]
labels = ['Triton', 'Torch', 'CUDA']
ax.set_xlabel('Vector Size')
ax.set_ylabel('Bandwidth as a fraction of CUDA')
# ax.set_yscale('log')
ax.set_xscale('log')
# ax.legend()
ax.legend(lines, labels)
# title
ax.set_title('Vector Add Kernels: CUDA vs. Triton vs. Torch')


# %% plot the results for the kernels to compare
all_kernels = ['torch', 'triton'] + best_kernels

all_df = df[['size'] + all_kernels]
all_df

# %%

fig, ax = plt.subplots(figsize=(8, 6))
for kern in all_kernels:
    ax.plot(df['size'], df[kern], label=kern)
ax.set_xlabel('Vector Size')
ax.set_ylabel('GB/sec')
# ax.set_yscale('log')
ax.set_xscale('log')
ax.legend()

# %% plot the results for the kernels to compare
all_kernels = ['torch', 'triton'] + best_kernels

fig, ax = plt.subplots(figsize=(8, 6))
for kern in all_kernels:
    ax.plot(df['size'], df[kern] / df['triton'], label=kern)
ax.set_xlabel('Vector Size')
ax.set_ylabel('Percent bandwidth vs Triton')
# ax.set_yscale('log')
ax.set_xscale('log')
ax.legend()

# %%
sizes = sorted(best_map.keys())
# %%
set.intersection(*[set(best_map[sz]) for sz in sizes[:9]])
# up to size 1 million, 64_1_1 is competitive with best

# %%
set.intersection(*[set(best_map[sz]) for sz in sizes[9:10]])
# for 2M ,  cuda_512_1_1

# %%
set.intersection(*[set(best_map[sz]) for sz in sizes[10:13]])
# again with 64_1_1 being best

# %%
set.intersection(*[set(best_map[sz]) for sz in sizes[13:14]])
# cuda_512_1_1

# %%
set.intersection(*[set(best_map[sz]) for sz in sizes[14:15]])
# cuda 1024_2_1

# %%
set.intersection(*[set(best_map[sz]) for sz in sizes[15:]])
# cuda_512_1_1

# %%
chosen = ['cuda_64_1_1', 'cuda_512_1_1', 'cuda_1024_2_1']


# %%

fig, ax = plt.subplots(figsize=(8, 6))
for kern in chosen + ['triton', 'torch']:
    ax.plot(df['size'], df[kern], label=kern)
ax.set_xlabel('Vector Size')
ax.set_ylabel('GB/sec')
# ax.set_yscale('log')
ax.set_xscale('log')
ax.legend()

# %% plot the results for the kernels to compare

chosen = ['cuda_64_1_1']

fig, ax = plt.subplots(figsize=(8, 6))
for kern in chosen + ['triton', 'torch']:
    ax.plot(df['size'], df[kern] * 2 / (df['triton'] + df['torch']), label=kern, marker='o', alpha=0.5)
ax.set_xlabel('Vector Size')
ax.set_ylabel('Percent bandwidth vs Avg(Triton, Torch)')
# ax.set_yscale('log')
ax.set_xscale('log')
ax.legend()
