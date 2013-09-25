import sys
import matplotlib.pyplot as plt
import subprocess
import numpy as np


output_folder = "output"
corpus = "nips_abstracts"

X = []
Y = []
Z = []

for K in [2,5,10, 20, 30, 40, 50, 60, 70, 80]:
    K = str(K)
    output = subprocess.check_output(("python ./anchor-word-recovery/learn_topics.py "+output_folder+"/M_"+corpus+".full_docs.mat.trunc.mat ./anchor-word-recovery/settings.example "+output_folder+"/vocab_"+corpus+".txt.trunc "+K+" L2 "+output_folder+"/demo_L2_out."+corpus+"."+K).split())
    lines = output.split('\n')
    for l in lines:
        if "avg objective function during recovery using" in l:
            X.append(int(K))
            Y.append(np.sqrt(float(l.split()[-1])))

infile = file(output_folder+"/log.anchors")
for l in infile:
    if 'max_dist' in l:
        Z.append(float(l.split()[-1]))
    
f, (p1, p2) = plt.subplots(2)
p1.plot(X,Y, lw=3)
#p.title("Num Topics " + corpus.replace('_', '-')) 
p1.set_ylabel("L2 Reconstruction Error")
p1.tick_params(axis='both', which='major', labelsize=10)
p1.set_xlabel("Topics")
p2.set_xlabel("Topics")
p2.set_ylabel("Last anchor distance")
p2.plot(xrange(1,len(Z)+1), Z, lw=3)

plt.show()
