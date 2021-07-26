#!/usr/bin/env python

import torch
from ctypes import CDLL, c_void_p

zero = CDLL("./zero.so")

N = 3
A = torch.ones(N, dtype=torch.float32)
Ac = A.cuda()
threads = 4

print('A', hex(A.data_ptr()), 'A.shape', A.shape)

Ap = Ac.data_ptr()
print('Ap', hex(Ap), 'Ac', Ac, 'Ac.shape', Ac.shape)

zero.zero(c_void_p(Ap), N, threads)

print('A', Ac)
