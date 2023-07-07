#using an array of ADC data and an array of channel numbers
#sorts it into an array by tpc/plane/event/channel/waveform
#returns lots of empty data which I ignore later

#this all based off the ICEBERG documentation
class planesorter:
    def sortdata(arrayADC, arrayChannel):
        import numpy as np
        import math
        dataarray = np.zeros((2,3,len(arrayADC),240,2128))
        for nEvent in range(len(arrayADC)):
            for nChannel in range(len(arrayADC[nEvent])):
                currentchannel =  arrayChannel[nEvent][nChannel]
                channelnum = currentchannel
                if(currentchannel > 999):
                    currentchannel -= 40
                    if(currentchannel > 1199):
                        currentchannel -= 40
                planenum = math.floor(currentchannel/200) #num 0-5. 0,2,4 go to TPC 0. 1,3,5 go to TPC 1
                tpcnum = planenum%2 #0 or 1, tpc number
                tpcplane = math.floor(planenum/2) #plane num within tpc, 0 to 2
                channelnum -= planenum*200
                if(planenum == 5):
                    channelnum -= 40
                dataarray[tpcnum][tpcplane][nEvent][channelnum] = arrayADC[nEvent][nChannel]
        return dataarray