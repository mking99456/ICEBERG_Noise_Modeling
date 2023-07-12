from myprocessor import myprocessor as myp
import numpy as np

#TO USE:
numfiles = 1
numcores = 1

with open('/dune/data/users/mking/ICEBERG_Run5_Decoded/filelist.txt') as f:
    filenames = f.readlines()
with open('/dune/data2/users/mking/ICEBERG_Run5_Decoded/filelist.txt') as g:
    secondfiles = g.readlines()
for i in range(len(secondfiles)):
    filenames.append(secondfiles[i])

asdmodel, totalevents, noskips, psdmodel = myp.multiprocess(filenames, numfiles, numcores)
savefile = '/dune/app/users/poshung/jupyter_scripts/MultiRunData.npz'
np.savez(savefile, asdmodel,psdmodel , totalevents, noskips)