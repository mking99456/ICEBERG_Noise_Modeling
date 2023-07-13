class myprocessor:
    def onefilefast(infile_name):
        import uproot
        import awkward as ak
        import numpy as np
        import time
        
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
        ak.values_astype(ADCarr, np.uint16)
        Channelarr = fChannel.array(library = "ak")
        ak.values_astype(Channelarr, np.uint16)

        #We end up with lots of empty events, so when we convert from an awk array to a np array we only take the events with data
        #What events have data?
        counter = []
        for index in range(len(ADCarr)):
            if(ak.count(ADCarr[index]) >= numchannels*(minwvfm/2)):
                counter.append(index)
                
        returnASD = np.empty((numtpcs,numplanes,maxwires,ASDlength),dtype=float)
        returnPSD = np.empty((numtpcs,numplanes,maxwires,ASDlength),dtype=float)
        returnFFT = np.empty((numtpcs,numplanes,maxwires,ASDlength),dtype=np.complex64)
        returnNoSkips = np.empty((numtpcs,numplanes,maxwires),dtype=np.uint16)
        returnCorr = np.empty((numtpcs,numplanes,maxwires,maxwires),dtype=float)
        TotalEvents = len(counter)
        print("BEGAN PROCESSING: " + printname + " CONTAINING "+ str(TotalEvents) + " EVENTS")
        for nEvent in range(len(counter)):
            SortedWaveforms = np.empty((numtpcs,numplanes,maxwires,minwvfm), dtype=int)
            if(nEvent == int(len(counter)/2)):
                midtime = time.time()
                timeelapsed = midtime-starttime
                print("HALFWAY DONE PROCESSING " + printname + "AT " + str(int(timeelapsed/60))+":"+str(int(timeelapsed%60)))
            for nChannel in range(len(ADCarr[counter[0]])):
                if(abs(np.sum(ADCarr[counter[nEvent]][nChannel])>10)):
                    CurrentWVFM = ak.to_numpy(ADCarr[counter[nEvent]][nChannel])
                    CurrentChannel = Channelarr[counter[nEvent]][nChannel]
                    CurrentWVFM = CurrentWVFM[0:minwvfm]

                    #MASK OUT THE CLOCK SIGNALS OR OTHER TRIGGERS
                    CurrentWVFM = myprocessor.mask_fast(CurrentWVFM)
                    #Get TPC/Plane/Channel
                    TPCNum, PlaneNum, ChannelNum = myprocessor.sort_single_data(CurrentChannel)
                    SortedWaveforms[TPCNum][PlaneNum][ChannelNum] = CurrentWVFM
                    #Get ASD/PSD
                    ADCASD, ADCPSD, ADCFFT = myprocessor.get_single_ASDPSDFFT(CurrentWVFM, SampleSpacing)
                    
                    returnASD[TPCNum][PlaneNum][ChannelNum] += ADCASD
                    returnPSD[TPCNum][PlaneNum][ChannelNum] += ADCPSD
                    returnFFT[TPCNum][PlaneNum][ChannelNum] += np.complex64(ADCFFT)
                    returnNoSkips[TPCNum][PlaneNum][ChannelNum] += 1
            for tpc in range(numtpcs):
                for plane in range(numplanes):
                    returnCorr[tpc][plane] += np.corrcoef(SortedWaveforms[tpc][plane])
        endtime = time.time()
        timeelapsed = endtime-starttime
        print("DONE PROCESSING " + printname + " IN " + str(int(timeelapsed/60))+":"+str(int(timeelapsed%60)))
        return returnASD, TotalEvents, returnNoSkips, returnPSD, returnCorr, returnFFT
    
    def onefileresilient(filename):
        import sys, time
        import numpy as np
        starttime=time.time()
        try:
            tempASDs, tempEvents, tempNoSkips, tempPSDs, tempCorr, tempFFT = myprocessor.onefilefast(filename)
            return tempASDs, tempEvents, tempNoSkips, tempPSDs, tempCorr, tempFFT
        except Exception as e:
            crash=["Error on line {}".format(sys.exc_info()[-1].tb_lineno),"\n",e]
            print(crash)
            timeX=str(time.time()-starttime)
            with open('/dune/app/users/poshung/multirun/CRASH-'+timeX+'.txt', "w") as crashlog:
                mytime = time.gmtime(time.time())
                crashlog.write("Crash Occurred On: " +str(mytime[1])+"/"+str(mytime[2])+"/"+str(mytime[0])+" At " + str(mytime[3])+":"+str(mytime[4])+":"+str(mytime[5]))
                for i in crash:
                    i=str(i)
                    crashlog.write(i)
            numtpcs = 2
            numplanes = 3
            maxwires = 240
            ASDlength = 1065 #the length of the ASD of the minwvfm
            returnASD = np.empty((numtpcs,numplanes,maxwires,ASDlength),dtype=float)
            returnPSD = np.empty((numtpcs,numplanes,maxwires,ASDlength),dtype=float)
            returnFFT = np.empty((numtpcs,numplanes,maxwires,ASDlength),dtype=np.complex64)
            returnNoSkips = np.empty((numtpcs,numplanes,maxwires),dtype=np.uint16)
            returnCorr = np.empty((numtpcs,numplanes,maxwires,maxwires),dtype=float)
            TotalEvents = 0
            return returnASD, TotalEvents, returnNoSkips, returnPSD, returnCorr
        
    def multiprocess(filenames,numfiles,NumProcessors):
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
        ASDarray = np.empty((numtpcs,numplanes,maxwires,ASDlength), dtype=float)
        PSDarray = np.empty((numtpcs,numplanes,maxwires,ASDlength), dtype=float)
        FFTarray = np.empty((numtpcs,numplanes,maxwires,ASDlength), dtype=np.complex_)
        noskips = np.empty((numtpcs,numplanes,maxwires), dtype=int)
        rCorr = np.empty((numtpcs,numplanes,maxwires,maxwires), dtype=float)
        filenames=filenames[2:2+numfiles]
        
        for tempASDs, tempEvents, tempNoSkips, tempPSDs, tempCorr, tempFFT in exe.map(myprocessor.onefileresilient, filenames):
            TotalEvents += tempEvents
            ASDarray += tempASDs
            PSDarray += tempPSDs
            noskips += tempNoSkips
            rCorr += tempCorr
            FFTarray += tempFFT
            print("Events Processed: "+str(TotalEvents))
            
            
        rCorr = np.square(rCorr)
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
                        FFTarray[tpc][plane][nChannel] = FFTarray[tpc][plane][nChannel]/noskips[tpc][plane][nChannel]
                        rCorr[tpc][plane][nChannel] = rCorr[tpc][plane][nChannel]/noskips[tpc][plane][nChannel]
                        rCorr[tpc][plane][:maxwires][nChannel] = rCorr[tpc][plane][:maxwires][nChannel]/noskips[tpc][plane][nChannel]
        myprocessor.HistAvgPSDbyPlane(PSDarray,numtpcs,numplanes,maxwires,minwvfm,SampleSpacing) #Plot our averaged data
        myprocessor.plotrCorr(rCorr,numtpcs,numplanes)
        return ASDarray, TotalEvents, noskips, PSDarray, rCorr, FFTarray
    
    #this is the recursive method to return the noise-free mean and std of a waveform
    def recursive_stats(data_arr):
        import numpy as np
        #we start by splitting our data in half, first half and second half
        data_len = len(data_arr)
        first_half = data_arr[0:int(data_len/2)+int(data_len/40)]
        second_half = data_arr[int(data_len/2)-int(data_len/40):data_len]
        
        #un-commenting my plot and print statements gives some demonstration of how this method works.

        #then we ask if the std of the full length array is below 10. this is our way of asking if there are any clock signals in the data.
        #if it is, we return the mean and standard deviation. if it's not, then we take the half of the data with less signal and call the function again, except only on that half.
        #this method should try to get away from/avoid any actual signal as soon as possible, since big spikes will skew the standard deviation of that half.
        
        #so essentially we half the length of the data and then half the length again and then repeat until we get a sample of data with no signal.
        
        #if the waveform length drops below 70 data points, we can guarantee there is no clock signal because the time between signals is 280 points and we always
        #pick the half with less noise. if the noiseSTD is still above 10 then it just means we have an especially noisy channel. honestly the code would likely work just fine
        #if only the data_len<50 piece was included
        if((np.std(data_arr)<10)|(data_len<70)):
            return np.std(data_arr), np.mean(data_arr)
        else:
            if(np.std(first_half) < np.std(second_half)):
                return myprocessor.recursive_stats(first_half)
            else:
                return myprocessor.recursive_stats(second_half)
            
    def mask_single_wvfm(waveform):
        import numpy as np
        noiseSTD, noiseMean = myprocessor.recursive_stats(waveform)  
                
        #keeping track of indexes we masked out
        switchedArr = np.zeros((len(waveform)),dtype=bool)
                
        #classic threshold mask. anything varied by >6*noiseSTD get replaced with the mean
        for datapt in range(0,len(waveform)):
            if((waveform[datapt] > noiseMean+6*noiseSTD)|(waveform[datapt] < noiseMean-6*noiseSTD)):
                waveform[datapt] = noiseMean
                switchedArr[datapt] = True
        #then we go back through and wherever we masked out a peak, we trace backwards and keep masking data points until we drop within 2 standard deviations of the mean.
                #we do this again for the forward direction
        for datapt in range(1,len(waveform)-1):
            if(switchedArr[datapt]):
                if(not(switchedArr[datapt-1])):
                    j=1
                    while((waveform[datapt-j]>noiseMean+2*noiseSTD)|(waveform[datapt-j]<noiseMean-2*noiseSTD)):
                        waveform[datapt-j] = noiseMean
                        switchedArr[datapt-j] = True
                        j+=1
                        if(datapt-j == 0):
                            break
                if(not(switchedArr[datapt+1])):
                    j=1
                    while((waveform[datapt+j]>noiseMean+2*noiseSTD)|(waveform[datapt+j]<noiseMean-2*noiseSTD)):
                        waveform[datapt+j] = noiseMean
                        switchedArr[datapt+j] = True
                        j+=1
                        if(datapt+j == len(waveform)):
                            break
                #then as a final check, we go through our data again and say, for every data point, if both the datapoint to the right and to the left have been masked, mask that
                #data point out too. this is just to make sure that section in between bimodal peaks doesn't slip through the cracks.
        for datapt in range(1,len(waveform)-1):
            if(switchedArr[datapt+1]&switchedArr[datapt-1]):
                switchedArr[datapt] = True
                waveform[datapt] = noiseMean
        return waveform
    def mask_fast(CurrentWVFM):
        import numpy as np
        import time
        noisestd, noisemean = myprocessor.recursive_stats(CurrentWVFM)
        peaks = np.where(abs(CurrentWVFM-noisemean)>6*noisestd)[0]
        for wvfmindex in peaks:
            i = wvfmindex
            while((abs(CurrentWVFM[i]-noisemean)>2*noisestd)&(i>=0)):
                CurrentWVFM[i] = noisemean
                i-=1
            i=wvfmindex
            while((abs(CurrentWVFM[i]-noisemean)>2*noisestd)&(i<2128)):
                CurrentWVFM[i] = noisemean
                i+=1
        return CurrentWVFM
    def sort_single_data(currentchannel):
        channelnum = currentchannel
        if(currentchannel > 999):
            currentchannel -= 40
            if(currentchannel > 1199):
                currentchannel -= 40
        planenum = int(currentchannel/200) #num 0-5. 0,2,4 go to TPC 0. 1,3,5 go to TPC 1
        tpcnum = planenum%2 #0 or 1, tpc number
        tpcplane = int(planenum/2) #plane num within tpc, 0 to 2
        channelnum -= planenum*200
        if(planenum == 5):
            channelnum -= 40
        return tpcnum, tpcplane, channelnum
    
    #Credit: Angela
    #This code takes two 1D arrays, our FFT and Frequency, and the number of bins
    #in a 2D histogram, then returns a plot tracing out the maximum bin of the heatmap  
    def hist_maxes(adc_pdbfs, adc_freq, binnum):
        import numpy as np
        hist, *edges = np.histogram2d(adc_freq/1e6, adc_pdbfs, bins=(binnum,binnum))
        ycenters = (edges[1][:-1] + edges[1][1:]) / 2
        xcenters = (edges[0][:-1] + edges[0][1:]) / 2

        maxes = []
        for array in hist:
            maxes.append(np.where(array == max(array))[0][0])
        max_bins = [ycenters[i] for i in maxes]        
        return xcenters, max_bins
   
    def get_single_ASDPSDFFT(waveform, SampleSpacing):
        import numpy as np
        ADCFFT = np.fft.rfft(waveform)
        N = len(waveform)
        T = SampleSpacing*N
        ADCFFT[0] = 0
        ADCPSD = 2*T/N**2 * np.abs(ADCFFT)**2
        ADCASD = np.sqrt(ADCPSD)
        return ADCASD, ADCPSD, ADCFFT
    
    def HistAvgASDbyPlane(data_arr, numtpcs, numplanes, maxwires, minwvfm, SampleSpacing):
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        from mpl_toolkits.axes_grid1 import make_axes_locatable   
             
        #For plot of ASDs
        fig,ax = plt.subplots(numtpcs,numplanes,num=1)
        fig.set_figheight(10)
        fig.set_figwidth(15)
        
        colorvalues = mpl.cm.viridis(range(maxwires))
        divider = make_axes_locatable(plt.gca())
        ax_cb = divider.new_horizontal(size="5%", pad=0.05)    
        cb1 = mpl.colorbar.ColorbarBase(ax_cb, cmap=mpl.cm.viridis, orientation='vertical',boundaries = np.arange(maxwires+1),values = colorvalues)
        plt.gcf().add_axes(ax_cb)
        
        #For plot of freq v. channel # v. color
        fig2,ax2 = plt.subplots(numtpcs,numplanes,num=2)
        fig2.set_figheight(numtpcs*5)
        fig2.set_figwidth(numplanes*5)
        
        planenames = ["u", "v", "z"]
        
        AvgASD = data_arr
        
        for tpc in range(len(data_arr)):
            for plane in range(len(data_arr[tpc])):
                N = minwvfm
                freq = np.fft.rfftfreq(N,SampleSpacing)
                #The 2d histogram takes only 1d arrays
                #so we have to flatten out our AvgASD array into one long array
                #also we have to do the same for the frequencies
                longASD = np.empty(0,dtype=float)
                longFreq = np.empty(0,dtype=float)
                for channel in range(len(AvgASD[tpc][plane])):
                    longASD = np.concatenate((longASD, abs(AvgASD[tpc][plane][channel])))
                    longFreq = np.concatenate((longFreq, freq))
                
                #For 2D plot
                cmap = mpl.cm.get_cmap('viridis')
                cmap.set_under('white')
                ax2[tpc][plane].pcolormesh(freq,range(maxwires),np.log(AvgASD[tpc][plane]),cmap = cmap,shading='gouraud', vmin=np.log(.0001),vmax=np.log(.1))
                
                #For ASD Plot
                for nChannel in range(len(AvgASD[tpc][plane])):
                    ax[tpc][plane].scatter(freq,AvgASD[tpc][plane][nChannel],s=0.05,color = colorvalues[nChannel])
                    
                ax[tpc][plane].set_ylim(0,0.10)
                ax[tpc][plane].set_xlabel("Freq [1e6 Hz]")
                ax[tpc][plane].set_ylabel("ADC strain noise [1/Hz^0.5]")
                ax[tpc][plane].set_title("Averaged Channel ASD for TPC: " + str(tpc) + " Wire Plane: " + planenames[plane])
                
                ax2[tpc][plane].set_xlabel("Freq Hz")
                ax2[tpc][plane].set_ylabel("Channel Number [1/Hz^0.5]")
                ax2[tpc][plane].set_title("Averaged Channel ASD for TPC: " + str(tpc) + " Wire Plane: " + planenames[plane])
        plt.show()
        
    def HistAvgPSDbyPlane(data_arr, numtpcs, numplanes, maxwires, minwvfm, SampleSpacing):
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        from mpl_toolkits.axes_grid1 import make_axes_locatable   
             
        #For plot of ASDs
        fig,ax = plt.subplots(numtpcs,numplanes,num=1)
        fig.set_figheight(5*numplanes)
        fig.set_figwidth(5*numtpcs)
        
        colorvalues = mpl.cm.viridis(range(maxwires))
        divider = make_axes_locatable(plt.gca())
        ax_cb = divider.new_horizontal(size="5%", pad=0.05)    
        cb1 = mpl.colorbar.ColorbarBase(ax_cb, cmap=mpl.cm.viridis, orientation='vertical',boundaries = np.arange(maxwires+1),values = colorvalues)
        plt.gcf().add_axes(ax_cb)
        
        #For plot of freq v. channel # v. color
        fig2,ax2 = plt.subplots(numtpcs,numplanes,num=2)
        fig2.set_figheight(numtpcs*5)
        fig2.set_figwidth(numplanes*5)
        
        planenames = ["u", "v", "z"]
        
        AvgASD = data_arr
        
        for tpc in range(len(data_arr)):
            for plane in range(len(data_arr[tpc])):
                N = minwvfm
                freq = np.fft.rfftfreq(N,SampleSpacing)
                #The 2d histogram takes only 1d arrays
                #so we have to flatten out our AvgASD array into one long array
                #also we have to do the same for the frequencies
                longASD = np.empty(0,dtype=float)
                longFreq = np.empty(0,dtype=float)
                for channel in range(len(AvgASD[tpc][plane])):
                    longASD = np.concatenate((longASD, abs(AvgASD[tpc][plane][channel])))
                    longFreq = np.concatenate((longFreq, freq))
                
                #For 2D plot
                cmap = mpl.cm.get_cmap('viridis')
                cmap.set_under('white')
                ax2[tpc][plane].pcolormesh(freq,range(maxwires),np.log(AvgASD[tpc][plane]),cmap = cmap,shading='gouraud')
                
                #For ASD Plot
                for nChannel in range(len(AvgASD[tpc][plane])):
                    ax[tpc][plane].scatter(freq,AvgASD[tpc][plane][nChannel],s=0.05,color = colorvalues[nChannel])
                    
                ax[tpc][plane].set_yscale("log")
                ax[tpc][plane].set_xlabel("Freq [1e6 Hz]")
                ax[tpc][plane].set_ylabel("ADC strain noise [1/Hz^0.5]")
                ax[tpc][plane].set_title("Averaged Channel PSD for TPC: " + str(tpc) + " Wire Plane: " + planenames[plane])
                
                ax2[tpc][plane].set_xscale("log")
                ax2[tpc][plane].set_xlabel("Freq Hz")
                ax2[tpc][plane].set_ylabel("Channel Number [1/Hz^0.5]")
                ax2[tpc][plane].set_title("Averaged Channel PSD for TPC: " + str(tpc) + " Wire Plane: " + planenames[plane])
        plt.show()
    def plotrCorr(rCorr,numtpcs,numplanes):
        import matplotlib.pyplot as plt
        import numpy as np
        fig,ax = plt.subplots(numtpcs,numplanes,num=1)
        fig.set_figheight(10*numplanes)
        fig.set_figwidth(10*numtpcs)
        fig.set_size_inches(18, 12)
        planenames = ["u", "v", "z"]

        for TPCnum in range(numtpcs):
            for PlaneNum in range(numplanes):
                rCoeff = rCorr[TPCnum][PlaneNum]
                ax[TPCnum][PlaneNum].set_xlabel("Channels 0-240")
                ax[TPCnum][PlaneNum].set_ylabel("Channels 0-240")
                ax[TPCnum][PlaneNum].set_title("Correlations Between Channels for TPC #" + str(TPCnum) +" Wire Plane " + planenames[PlaneNum])
                fig.colorbar(ax[TPCnum][PlaneNum].pcolor(rCoeff))
                ax[TPCnum][PlaneNum].pcolormesh(np.arange(240),np.arange(240),rCoeff,shading='gouraud')