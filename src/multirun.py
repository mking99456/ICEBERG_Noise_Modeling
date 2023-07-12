from myprocessor import myprocessor as myp
import numpy as np
import sys, time

#TO USE:
numfiles = 1
numcores = 1

with open('/dune/data/users/mking/ICEBERG_Run5_Decoded/filelist.txt') as f:
    filenames = f.readlines()
with open('/dune/data2/users/mking/ICEBERG_Run5_Decoded/filelist.txt') as g:
    secondfiles = g.readlines()
for i in range(len(secondfiles)):
    filenames.append(secondfiles[i])

starttime = time.time()

try:
    asdmodel, totalevents, noskips, psdmodel = myp.multiprocess(filenames, numfiles, numcores)
    savefile = '/dune/app/users/poshung/jupyter_scripts/ASDDataTest.npz'
    np.savez(savefile, asdmodel,psdmodel , totalevents, noskips)
except Exception as e:
    crash=["Error on line {}".format(sys.exc_info()[-1].tb_lineno),"\n",e]
    print(crash)
    timeX=str(time.time()-starttime)
    with open('/dune/app/users/poshung/jupyter_scripts/CRASH-'+timeX+'.txt', "w") as crashlog:
        mytime = time.gmtime(time.time())
        crashlog.write(str(mytime[1])+"/"+str(mytime[2])+"/"+str(mytime[0])+" at " + str(mytime[3])+":"+str(mytime[4])+":"+str(mytime[5]))
        for i in crash:
            i=str(i)
            crashlog.write(i)