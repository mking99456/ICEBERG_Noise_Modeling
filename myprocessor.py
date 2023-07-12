class myprocessor:
    def onefile(filename):
        from twodhist import twodhist as hist
        from planesorter import planesorter as sort
        from fileprocess import fileprocess as file
        from clockmask import clockmask as mask
        import numpy as np
        numtpcs = 2
        numplanes = 3
        numchannels = 1024
        maxwires = 240
        minwvfm = 2128
        ASDlength = 1065 #the length of the ASD of the minwvfm
        SampleSpacing = 0.5e-6 #0.5 microseconds per tick
        #print("BEGINNING RUN: " + str(run) + " OUT OF: " + str(len(filenames)))
        #print("File Path: " + filename)
        tempadc,tempsample,tempchannel = file.getADC(filename, minwvfm, numchannels) #Read the file
        TotalEvents = len(tempadc)
        tempmasked = mask.masksignals(tempadc) #Mask the clock signals
        tempsorted = sort.sortdata(tempmasked,tempchannel) #Sort the waveforms by TPC/Plane
        tempASDs,tempNoSkips, tempPSDs = hist.getASDs(tempsorted, numtpcs, numplanes, maxwires, ASDlength, SampleSpacing) #Take the average ASDs of each channel
        return tempASDs, TotalEvents, tempNoSkips, tempPSDs
    def onefilefast(infile_name):
        import uproot
        import awkward as ak
        import numpy as np
        import time
        from clockmask import clockmask as mask
        from planesorter import planesorter as sort
        from twodhist import twodhist as hist
        
        starttime = time.time()
        numtpcs = 2
        numplanes = 3
        numchannels = 1024
        maxwires = 240
        minwvfm = 2128
        ASDlength = 1065 #the length of the ASD of the minwvfm
        SampleSpacing = 0.5e-6 #0.5 microseconds per tick
        infile = uproot.open(infile_name)
        printname = infile_name[44:60]
        events = infile['Events']
        
        fADC = events['raw::RawDigits_tpcrawdecoder_daq_RunIcebergRawDecoder./raw::RawDigits_tpcrawdecoder_daq_RunIcebergRawDecoder.obj/raw::RawDigits_tpcrawdecoder_daq_RunIcebergRawDecoder.obj.fADC']
        fChannel = events['raw::RawDigits_tpcrawdecoder_daq_RunIcebergRawDecoder./raw::RawDigits_tpcrawdecoder_daq_RunIcebergRawDecoder.obj/raw::RawDigits_tpcrawdecoder_daq_RunIcebergRawDecoder.obj.fChannel']

        ADCarr = fADC.array(library = "ak")
        Channelarr = fChannel.array(library = "ak")

        #We end up with lots of empty events, so when we convert from an awk array to a np array we only take the events with data
        #What events have data?
        counter = []
        for index in range(len(ADCarr)):
            if(ak.count(ADCarr[index]) >= numchannels*(minwvfm/2)):
                counter.append(index)
                
        returnASD = np.zeros((numtpcs,numplanes,maxwires,ASDlength),dtype=float)
        returnPSD = np.zeros((numtpcs,numplanes,maxwires,ASDlength),dtype=float)
        returnNoSkips = np.zeros((numtpcs,numplanes,maxwires),dtype=int)
        TotalEvents = len(counter)
        print("BEGAN PROCESSING: " + printname + " CONTAINING "+ str(TotalEvents) + " EVENTS")
        for nEvent in range(len(counter)):
            if(nEvent == int(len(counter)/2)):
                midtime = time.time()
                timeelapsed = midtime-starttime
                print("HALFWAY DONE PROCESSING " + printname + "AT " + str(int(timeelapsed/60))+":"+str(int(timeelapsed%60)))
            for nChannel in range(len(ADCarr[counter[0]])):
                if(abs(np.sum(ADCarr[counter[nEvent]][nChannel])>10)):
                    CurrentWVFM = ak.to_numpy(ADCarr[counter[nEvent]][nChannel])
                    CurrentChannel = Channelarr[counter[index]][nChannel]
                    CurrentWVFM = CurrentWVFM[0:minwvfm]

                    #MASK OUT THE CLOCK SIGNALS OR OTHER TRIGGERS
                    CurrentWVFM = mask.mask_single_wvfm(CurrentWVFM)
                    #Get TPC/Plane/Channel
                    TPCNum, PlaneNum, ChannelNum = sort.sort_single_data(CurrentChannel)
                    #Get ASD/PSD
                    ADCASD, ADCPSD = hist.get_single_ASDPSD(CurrentWVFM, SampleSpacing)
                    
                    returnASD[TPCNum][PlaneNum][ChannelNum] += ADCASD
                    returnPSD[TPCNum][PlaneNum][ChannelNum] += ADCPSD
                    returnNoSkips[TPCNum][PlaneNum][ChannelNum] += 1
        endtime = time.time()
        timeelapsed = endtime-starttime
        print("DONE PROCESSING " + printname + " IN " + str(int(timeelapsed/60))+":"+str(int(timeelapsed%60)))
        return returnASD, TotalEvents, returnNoSkips, returnPSD
    def onefileresilient(filename):
        import sys, time
        import numpy as np
        starttime=time.time()
        try:
            tempASDs, tempEvents, tempNoSkips, tempPSDs = myprocessor.onefilefast(filename)
            return tempASDs, tempEvents, tempNoSkips, tempPSDs
        except Exception as e:
            crash=["Error on line {}".format(sys.exc_info()[-1].tb_lineno),"\n",e]
            print(crash)
            timeX=str(time.time()-starttime)
            with open('/dune/app/users/poshung/jupyter_scripts/CRASH-'+timeX+'.txt', "w") as crashlog:
                mytime = time.gmtime(time.time())
                crashlog.write("Crash Occurred On: " +str(mytime[1])+"/"+str(mytime[2])+"/"+str(mytime[0])+" At " + str(mytime[3])+":"+str(mytime[4])+":"+str(mytime[5]))
                for i in crash:
                    i=str(i)
                    crashlog.write(i)
            numtpcs = 2
            numplanes = 3
            maxwires = 240
            ASDlength = 1065 #the length of the ASD of the minwvfm
            returnASD = np.zeros((numtpcs,numplanes,maxwires,ASDlength),dtype=float)
            returnPSD = np.zeros((numtpcs,numplanes,maxwires,ASDlength),dtype=float)
            returnNoSkips = np.zeros((numtpcs,numplanes,maxwires),dtype=int)
            TotalEvents = 0
            return returnASD, TotalEvents, returnNoSkips, returnPSD 
    def multiprocess(filenames,numfiles,NumProcessors):
        from twodhist import twodhist as hist
        import numpy as np
        from concurrent.futures import ProcessPoolExecutor
        #Everything is essentially hard coded for 2 tpcs, 3 planes per tpc, a maximum of 240 wires per plane, and a minimum waveform length of 2128
        #If you need to adjust these values (except the waveform length) you will have to write your own sortdata() function
        #Also, you will have to write your own getADC() function based on your file structure
        numtpcs = 2
        numplanes = 3
        maxwires = 240
        minwvfm = 2128
        ASDlength = 1065 #the length of the ASD of the minwvfm
        SampleSpacing = 0.5e-6 #0.5 microseconds per tick
        exe = ProcessPoolExecutor(NumProcessors)

        TotalEvents = 0
        ASDarray = np.zeros((numtpcs,numplanes,maxwires,ASDlength), dtype=float)
        PSDarray = np.zeros((numtpcs,numplanes,maxwires,ASDlength), dtype=float)
        noskips = np.zeros((numtpcs,numplanes,maxwires), dtype=int)
        filenames=filenames[0:numfiles]
        
        for tempASDs, tempEvents, tempNoSkips, tempPSDs in exe.map(myprocessor.onefileresilient, filenames):
            TotalEvents += tempEvents
            ASDarray += tempASDs
            PSDarray += tempPSDs
            noskips += tempNoSkips
            
        
        """for run in range(10):
            tempASDs, tempEvents, tempNoSkips = myprocessor.onefile(filenames,run,numtpcs,numchannels,maxwires,minwvfm,ASDlength,SampleSpacing)
            TotalEvents += tempEvents
            ASDarray += tempASDs #Add to a running sum
            noskips += tempNoSkips#Lots of empty data, so for our averaging to work out we keep an array storing the number of events that contributed to each channel sum="""
                
        print("BEGINNING FINAL STEP")
        for tpc in range(numtpcs): #This is the averaging I was talking about
            for plane in range(numplanes):
                for nChannel in range(maxwires):
                    if(noskips[tpc][plane][nChannel] > 0):
                        ASDarray[tpc][plane][nChannel] = ASDarray[tpc][plane][nChannel]/noskips[tpc][plane][nChannel]
                        PSDarray[tpc][plane][nChannel] = PSDarray[tpc][plane][nChannel]/noskips[tpc][plane][nChannel]
        hist.HistAvgASDbyPlane(PSDarray,numtpcs,numplanes,maxwires,minwvfm,SampleSpacing) #Plot our averaged data
        return ASDarray, TotalEvents, noskips, PSDarray
