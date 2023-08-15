from myprocessor import myprocessor as myp
import numpy as np
#How many files do you want to analyze?
#How many workers do you want to use?
numfiles = 1
numcores = 2

#Read our lists of file paths
with open('/dune/data/users/mking/ICEBERG_Run5_Decoded/filelist.txt') as f:
    filenames = f.readlines()
with open('/dune/data2/users/mking/ICEBERG_Run5_Decoded/filelist.txt') as g:
    secondfiles = g.readlines()
for i in range(len(secondfiles)):
    filenames.append(secondfiles[i])
    
filenames = ['iceberg_r009693_sr01_20210404T192346_1_dl1_decode.root', 'iceberg_r009697_sr01_20210404T194635_1_dl1_decode.root', 'iceberg_r009702_sr01_20210404T204459_1_dl1_decode.root',  'iceberg_r009706_sr01_20210404T210522_1_dl1_decode.root',  'iceberg_r009710_sr01_20210404T212534_1_dl1_decode.root',
'iceberg_r009694_sr01_20210404T192925_1_dl1_decode.root',  'iceberg_r009698_sr01_20210404T195219_1_dl1_decode.root',  'iceberg_r009703_sr01_20210404T205038_1_dl1_decode.root',  'iceberg_r009707_sr01_20210404T211101_1_dl1_decode.root',  'iceberg_r009711_sr01_20210404T213002_1_dl1_decode.root',
'iceberg_r009695_sr01_20210404T193508_1_dl1_decode.root',  'iceberg_r009699_sr01_20210404T195808_1_dl1_decode.root',  'iceberg_r009704_sr01_20210404T205505_1_dl1_decode.root',  'iceberg_r009708_sr01_20210404T211527_1_dl1_decode.root',
'iceberg_r009696_sr01_20210404T194054_1_dl1_decode.root',  'iceberg_r009701_sr01_20210404T200822_1_dl1_decode.root'  ,'iceberg_r009705_sr01_20210404T205939_1_dl1_decode.root' , 'iceberg_r009709_sr01_20210404T212107_1_dl1_decode.root']

for i in range(len(filenames)):
    filenames[i] = '/dune/data/users/mking/ICEBERG_Run5_Decoded/no_pulser_or_cosmics/' + filenames[i]

#Process the file list
asdmodel, totalevents, noskips, psdmodel, rcorr,fftmodel,megacorr,corrband = myp.multiprocess(filenames, numfiles, numcores)

#Save the data. You will need to change this every run/delete and re-create the .npz file
savefile = '/dune/app/users/poshung/nopulserorcosmic/multirunresults.npz'
np.savez(savefile, ASDs=asdmodel,PSDs=psdmodel, total_events=totalevents, noskips=noskips, corrbyplane=rcorr, FFTs=fftmodel,megacorr=megacorr,corrband=corrband)