#!/usr/bin/env python

'''
CUDA_PATH ?= /usr/local/cuda

.PHONY: clean

vadd.so: vadd.o
	nvcc -shared $^ -o $@ -lcuda

vadd.o: vadd.cu
	nvcc -I $(CUDA_PATH)/include -I$(CUDA_PATH)/samples/common/inc -arch=sm_70 --compiler-options '-fPIC' $^ -c $@

clean:
	rm -f *.o *.so
'''
# %%
import os
import sys
import glob
import subprocess
from itertools import product
from multiprocessing import Pool
from tqdm.notebook import tqdm

out_dir = '/tmp/vadd/'
os.makedirs(out_dir, exist_ok=True)

this_dir = os.path.dirname(os.path.realpath(__file__))

# Remove all the files in the output directory
for f in glob.glob(os.path.join(out_dir, '*.so')):
	os.remove(f)

def compile(args):
	threads, vectorize, unroll = args
	out = os.path.join(out_dir, f"vadd_{threads}_{vectorize}_{unroll}.so")
	if os.path.exists(out):
		os.remove(out)
	src = os.path.join(this_dir, 'vadd.cu')

	cmd = "/usr/local/cuda/bin/nvcc"
	cmd += f" -DTHREADS={threads}"
	cmd += f" -DVECTORIZE={vectorize}"
	cmd += f" -DUNROLL={unroll}"
	cmd += " -I/usr/local/cuda/include"
	cmd += " -arch=sm_86 --compiler-options '-fPIC'"
	cmd += f" {src} -shared -lcuda -o {out}"
	print(cmd)
	subprocess.check_call(cmd, shell=True)


# thread_range = [32]
# vectorize_range = [8]
# unroll_range = [8]
thread_range = [32, 64, 128, 256, 512, 1024]
vectorize_range = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
unroll_range = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]

# multi-threaded
with Pool(32) as p:
	p.map(compile, product(thread_range, vectorize_range, unroll_range))
print('done')

# %%