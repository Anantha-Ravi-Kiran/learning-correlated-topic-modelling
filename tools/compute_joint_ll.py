import sys
import numpy as np

inp_file = sys.argv[1]
with open(inp_file) as f:
    lines = f.read().splitlines()
    
lines = lines[:len(lines)-1]
lines = [float(x) for x in lines]
lines = np.array(lines)
print lines.sum()/lines.shape[0]
