from myprocessor import myprocessor as myp
import numpy as np

#How many files do you want to analyze?
#How many workers do you want to use?
numfiles = 20
numcores = 3

#Read our lists of file paths
with open('/dune/data/users/mking/ICEBERG_Run5_Decoded/filelist.txt') as f:
    filenames = f.readlines()
with open('/dune/data2/users/mking/ICEBERG_Run5_Decoded/filelist.txt') as g:
    secondfiles = g.readlines()
for i in range(len(secondfiles)):
    filenames.append(secondfiles[i])
    
#Process the file list
asdmodel, totalevents, noskips, psdmodel, rcorr,fftmodel = myp.multiprocess(filenames, numfiles, numcores)

#Save the data. You will need to change this every run/delete and re-create the .npz file
savefile = '/dune/app/users/poshung/multirun/MultiRunData.npz'
np.savez(savefile, asdmodel,psdmodel , totalevents, noskips, rcorr, fftmodel)