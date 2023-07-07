class myprocessor:
    def multiprocess(filenames):
        from twodhist import twodhist as hist
        from planesorter import planesorter as sort
        from fileprocess import fileprocess as file
        from clockmask import clockmask as mask
        import numpy as np
        
        #Everything is essentially hard coded for 2 tpcs, 3 planes per tpc, a maximum of 240 wires per plane, and a minimum waveform length of 2128
        #If you need to adjust these values (except the waveform length) you will have to write your own sortdata() function
        #Also, you will have to write your own getADC() function based on your file structure
        numtpcs = 2
        numplanes = 3
        numchannels = 1024
        maxwires = 240
        minwvfm = 2128
        ASDlength = 1065 #the length of the ASD of the minwvfm
        SampleSpacing = 0.5e-6 #0.5 microseconds per tick

        
        adcarray = np.zeros((numtpcs,numplanes,maxwires,ASDlength), dtype=float)
        noskips = np.zeros((numtpcs,numplanes,maxwires), dtype=int)
        for run in range(len(filenames)):
            print("BEGINNING RUN: " + filenames[run])
            tempadc,tempsample,tempchannel = file.getADC(filenames[run], minwvfm, numchannels) #Read the file
            tempmasked = mask.masksignals(tempadc) #Mask the clock signals
            tempsorted = sort.sortdata(tempmasked,tempchannel) #Sort the waveforms by TPC/Plane
            tempASDs,tempNoSkips = hist.getASDs(tempsorted, numtpcs, numplanes, maxwires, ASDlength, SampleSpacing) #Take the average ASDs of each channel
            adcarray += tempASDs #Add to a running sum
            noskips += tempNoSkips#Lots of empty data, so for our averaging to work out we 
                #keep an array storing the number of events that contributed to each channel sum=
        print("Beginning Final Step")
        for tpc in range(numtpcs): #This is the averaging I was talking about
            for plane in range(numplanes):
                for nChannel in range(maxwires):
                    if(noskips[tpc][plane][nChannel] > 0):
                        adcarray[tpc][plane][nChannel] = adcarray[tpc][plane][nChannel]/noskips[tpc][plane][nChannel]
        hist.HistAvgASDbyPlane(adcarray,numtpcs,numplanes,maxwires,minwvfm,SampleSpacing) #Plot our averaged data
