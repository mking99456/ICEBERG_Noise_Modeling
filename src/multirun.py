from myprocessor import myprocessor as myp
import numpy as np
import sys, time

with open('/dune/data/users/mking/ICEBERG_Run5_Decoded/filelist.txt') as f:
    filenames = f.readlines()
with open('/dune/data2/users/mking/ICEBERG_Run5_Decoded/filelist.txt') as g:
    secondfiles = g.readlines()
for i in range(len(secondfiles)):
    filenames.append(secondfiles[i])
#filenames = ['iceberg_r009055_sr01_20210324T180612_1_dl1_decode.root']
#multiprocess(filenames, numfiles, numprocessors)
try:
    asdmodel, totalevents, noskips, psdmodel = myp.multiprocess(filenames, 24, 8)
    savefile = '/dune/app/users/poshung/jupyter_scripts/ASDDataTest.npz'
    np.savez(savefile, asdmodel,psdmodel , totalevents, noskips)
except Exception as e:
    crash=["Error on line {}".format(sys.exc_info()[-1].tb_lineno),"\n",e]
    print(crash)
    timeX=str(time.time())
    with open('/dune/app/users/poshung/jupyter_scripts/CRASH-'+timeX+'.txt', "w") as crashlog:
        for i in crash:
            i=str(i)
            crashlog.write(i)