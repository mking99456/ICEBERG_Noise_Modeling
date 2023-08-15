from myprocessor import myprocessor as myp
import numpy as np
import sys

#Read our lists of file paths
with open('/dune/data/users/mking/ICEBERG_Run5_Decoded/filelist.txt') as f:
    filenames = f.readlines()
with open('/dune/data2/users/mking/ICEBERG_Run5_Decoded/filelist.txt') as g:
    secondfiles = g.readlines()
for i in range(len(secondfiles)):
    filenames.append(secondfiles[i])

currentindex = int(sys.argv[1])
filename = filenames[currentindex]
numcores = 2

#Process the file list
asdmodel, totalevents, noskips, psdmodel, rcorr, fftmodel, megacorr = myp.onefileresilient(filename, numcores)

#Save the data. You will need to change this every run/delete and re-create the .npz file
savefile = '/dune/app/users/poshung/GridResults/unmaskedcorr'+filename[52:60]+'.npz'
np.savez(savefile, ASDs=asdmodel,PSDs=psdmodel, total_events=totalevents, noskips=noskips, corrbyplane=rcorr, FFTs=fftmodel,megacorr=megacorr)