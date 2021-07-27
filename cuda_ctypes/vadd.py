#!/usr/bin/env python


# %%
import os
import torch
from ctypes import CDLL, c_void_p, c_int
import time


# %%
# get the current directory
vadd = CDLL('./vadd.so')

N = 10000
A = torch.zeros(N, dtype=torch.float32).cuda()
B = torch.ones(N, dtype=torch.float32).cuda()
C = torch.zeros(N, dtype=torch.float32).cuda()
threads = 1024

Ap = c_void_p(A.data_ptr())
Bp = c_void_p(B.data_ptr())
Cp = c_void_p(C.data_ptr())
vadd.vadd(Ap, Bp, Cp, c_int(N), c_int(threads))


print('A', A[:2])
print('B', B[:2])
print('C', C[:2])
print('A', A[-2:])
print('B', B[-2:])
print('C', C[-2:])
