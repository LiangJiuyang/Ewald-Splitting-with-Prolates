# library to read/write mrc files
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import numpy as np
import sys
import math
import argparse
import torch
import pickle
import time

beginning = time.time()
# set device
if(torch.cuda.is_available()):
 device_ = 'cuda'
else:
 device_ = 'cpu'
print(device_)


# initialize parser
parser = argparse.ArgumentParser(prog='python cryo-EM_mergevoxelfile.py', description='Validation of structural models and trajectories against cryo-EM map')

parser.add_argument('--voxelfile', type=str, help='file with selected voxels')
parser.add_argument('--outvox', type=str, help='output file with voxels to keep')
args = parser.parse_args()

voxelfile = vars(args)["voxelfile"]
outvox = vars(args)["outvox"]


# get final list of voxels

voxs = voxelfile.split(',')

tokeep = []
for v in voxs:
   tokeep += pickle.load(open(v,'rb'))

tokeep = list(set(tokeep))
# number of voxels
nvox = len(tokeep)
## PRINT OUT SOME INFO
print("\n\n General input parameters:")
print("%37s %d" % ("# voxels for cc calculation:", nvox))

pickle.dump(tokeep,open(outvox,'wb'))