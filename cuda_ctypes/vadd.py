#!/usr/bin/env python

import torch
from ctypes import CDLL
import time

vadd = CDLL("./vadd.so")

N = 10
A = torch.zeros(N, dtype=torch.float32).cuda(non_blocking=True)
B = torch.ones(N, dtype=torch.float32).cuda(non_blocking=True)
C = torch.empty(N, dtype=torch.float32).cuda(non_blocking=True)
threads = 4

C = A * 99 + B * 99

Ap = A.data_ptr()
Bp = B.data_ptr()
Cp = C.data_ptr()
print('Ap', hex(Ap), 'Bp', hex(Bp), 'Cp', hex(Cp))

print('BEFORE')
print('A', A[:2])
print('B', B[:2])
print('C', C[:2])
print("SYCHRONIZING")
torch.cuda.synchronize()
print("RUNNING")
vadd.vadd(Ap, Bp, Cp, N, threads)
print("SYCHRONIZING")
torch.cuda.synchronize()
print('AFTER')
print('A', A[:2])
print('B', B[:2])
print('C', C[:2])
