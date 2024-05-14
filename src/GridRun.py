from myprocessor import myprocessor as myp
import numpy as np
import sys

#Arguments are: a text file with a list of input ICEBERG data files
#               the output directory to save the files

filelist = sys.argv[1]
outDir = sys.argv[2]
savefile = outDir+'/'

#Read our lists of file paths
with open(filelist) as f:
    filenames = f.readlines()

#Process the file list
myp.createNoiseModel(filenames,savefile)

#Save the data. You will need to change this every run/delete and re-create the .npz file
savefile = '/dune/app/users/poshung/GridResults/unmaskedcorr'+filename[52:60]+'.npz'
np.savez(savefile, ASDs=asdmodel,PSDs=psdmodel, total_events=totalevents, noskips=noskips, corrbyplane=rcorr, FFTs=fftmodel,megacorr=megacorr)
