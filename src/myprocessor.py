class myprocessor:
    def processevent(ADCarr,Channelarr,numtpcs=2,numplanes=3,maxwires=240,minwvfm=2128,ASDlength=1065,SampleSpacing=0.5e-6):
        import numpy as np
        import awkward as ak
        #from scipy.stats import kurtosis
        import matplotlib.pyplot as plt
        #We sort the waveforms by TPC/Plane so that we can find the correlation coefficients of all of them
        SortedWaveforms = np.zeros((numtpcs,numplanes,maxwires,minwvfm), dtype=int)
        returnASD = np.zeros((numtpcs,numplanes,maxwires,ASDlength),dtype=float)
        returnPSD = np.zeros((numtpcs,numplanes,maxwires,ASDlength),dtype=float)
        returnFFT = np.zeros((numtpcs,numplanes,maxwires,ASDlength),dtype=np.complex_)
        returnNoSkips = np.zeros((numtpcs,numplanes,maxwires),dtype=np.uint64)
        returnCorr = np.zeros((numtpcs,numplanes,maxwires,maxwires),dtype=float)
        #prekurtosis = 0
        #postkurtosis = 0
        #postmaskstd = 0
        #premaskstd = 0
        #cutouts = 0
        for nChannel in range(len(ADCarr)):
            if(abs(np.sum(ADCarr[nChannel])>10)):
                unmaskedWVFM = np.array(ak.to_numpy(ADCarr[nChannel]))
                #plotchan = np.array(ak.to_numpy(ADCarr[nChannel]))
                CurrentChannel = Channelarr[nChannel]
                CurrentWVFM = unmaskedWVFM[0:minwvfm]

                #prekurtosis += kurtosis(CurrentWVFM)
                #premaskstd += np.std(CurrentWVFM)
                #Mask out the clock signals and other triggers
                CurrentWVFM, tempcutouts = myprocessor.mask_fast(CurrentWVFM)
                #cutouts += tempcutouts
                #Get TPC/Plane/Channel
                TPCNum, PlaneNum, ChannelNum = myprocessor.sort_single_data(CurrentChannel)
                #Sort our current waveform
                SortedWaveforms[TPCNum][PlaneNum][ChannelNum] = CurrentWVFM              
                #Get ASD/PSD/FFT
                ADCASD, ADCPSD, ADCFFT = myprocessor.get_single_ASDPSDFFT(CurrentWVFM, SampleSpacing)

                #postkurtosis += kurtosis(CurrentWVFM)
                #postmaskstd += np.std(CurrentWVFM)
                #Input our current ASD/PSD/FFT where it belongs
                returnASD[TPCNum][PlaneNum][ChannelNum] += ADCASD
                returnPSD[TPCNum][PlaneNum][ChannelNum] += ADCPSD
                returnFFT[TPCNum][PlaneNum][ChannelNum] += np.complex64(ADCFFT)
                #We keep track of how many events contributed to each channel sum so that we can average correctly later
                returnNoSkips[TPCNum][PlaneNum][ChannelNum] += 1
                """if(tempcutouts > 300):
                    fig,ax = plt.subplots(2,2)
                    plotchan = plotchan[0:minwvfm]
                    ax[1][0].hist(CurrentWVFM, bins=30)
                    ax[0][0].hist(plotchan, bins=30)
                    ax[1][0].set_title("Post-mask Kurtosis: " + str(kurtosis(CurrentWVFM)))
                    ax[0][0].set_title("Pre-mask Kurtosis: " + str(kurtosis(plotchan)))
                    ax[0][1].plot(plotchan)
                    ax[1][1].plot(CurrentWVFM)
                    ax[1][1].set_title("Masked Waveform")
                    ax[0][1].set_title("Unmasked Waveform")
                    plt.suptitle("Number of Cutouts: "+str(tempcutouts))
                    plt.show()"""
        MegaSortedWaveforms = SortedWaveforms[0][0]
        for tpc in range(numtpcs):
            for plane in range(numplanes):
                #Now we can find the correlation coefficients of each wire plane
                returnCorr[tpc][plane] += np.corrcoef(SortedWaveforms[tpc][plane])
                if(not((tpc==0)&(plane==0))):
                    MegaSortedWaveforms = np.concatenate((MegaSortedWaveforms, SortedWaveforms[tpc][plane]))
        returnMegaCorr = np.corrcoef(MegaSortedWaveforms)
        corrbandpass = myprocessor.getcorrelationsbyfrequency(MegaSortedWaveforms)
        #print("Pre-mask Kurtosis: " + str(prekurtosis/total_channels))
        #print("Post-mask Kurtosis: " + str(postkurtosis/total_channels))
        #print("Pre-mask STD: " + str(premaskstd/total_channels))
        #print("Post-mask STD: " + str(postmaskstd/total_channels))
        #print("Number of points cut: " + str(cutouts/total_channels))
        return returnASD, returnPSD, returnFFT, returnCorr, returnNoSkips, returnMegaCorr, corrbandpass
    def onefilefast(infile_name, numcores):
        import uproot
        import awkward as ak
        import numpy as np
        import time
        from concurrent.futures import ProcessPoolExecutor
        
        starttime = time.time()
        numtpcs = 2
        numplanes = 3
        numchannels = 1024
        maxwires = 240
        minwvfm = 2128
        ASDlength = 1065 #the length of the ASD of the minwvfm
        SampleSpacing = 0.5e-6 #0.5 microseconds per tick
        numfreqbins=15
        exe = ProcessPoolExecutor(numcores)
        
        infile = uproot.open(infile_name)
        printname = infile_name[44:60]
        events = infile['Events']
        
        fADC = events['raw::RawDigits_tpcrawdecoder_daq_RunIcebergRawDecoder./raw::RawDigits_tpcrawdecoder_daq_RunIcebergRawDecoder.obj/raw::RawDigits_tpcrawdecoder_daq_RunIcebergRawDecoder.obj.fADC']
        fChannel = events['raw::RawDigits_tpcrawdecoder_daq_RunIcebergRawDecoder./raw::RawDigits_tpcrawdecoder_daq_RunIcebergRawDecoder.obj/raw::RawDigits_tpcrawdecoder_daq_RunIcebergRawDecoder.obj.fChannel']

        #Switch to int16 ASAP in order to minimize memory usage. Doesn't really work though!
        ADCarr = fADC.array(library = "ak")
        ak.values_astype(ADCarr, np.uint16)
        Channelarr = fChannel.array(library = "ak")
        ak.values_astype(Channelarr, np.uint16)

        #We end up with lots of empty events, so when we convert from an awk array to a np array we only take the events with data
        
        #Find out events have data
        counter = []
        for index in range(len(ADCarr)):
            if(ak.count(ADCarr[index]) >= numchannels*(minwvfm/2)):
                counter.append(index)

        #Data to return
        returnASD = np.zeros((numtpcs,numplanes,maxwires,ASDlength),dtype=np.float128)
        returnPSD = np.zeros((numtpcs,numplanes,maxwires,ASDlength),dtype=np.float128)
        returnFFT = np.zeros((numtpcs,numplanes,maxwires,ASDlength),dtype=np.complex128)
        returnNoSkips = np.zeros((numtpcs,numplanes,maxwires),dtype=np.uint64)
        returnCorr = np.zeros((numtpcs,numplanes,maxwires,maxwires),dtype=np.float128)
        returnMegaCorr = np.zeros((6*maxwires, 6*maxwires), dtype=float)
        returnCorrBand = np.zeros((numfreqbins,6*maxwires,6*maxwires),dtype=float)
        TotalEvents = len(counter)
        
        print("BEGAN PROCESSING: " + printname + " CONTAINING "+ str(TotalEvents) + " EVENTS")
        #We iterate through the events and the channels, inspecting and processing one waveform at a time
        
        for tempASD, tempPSD, tempFFT, tempCorr, tempNoSkips, tempMegaCorr, tempCorrBand in exe.map(myprocessor.processevent, ADCarr, Channelarr):
            returnASD += tempASD
            returnPSD += tempPSD
            returnFFT += tempFFT
            returnCorr += tempCorr
            returnNoSkips += tempNoSkips
            returnMegaCorr += tempMegaCorr
            returnCorrBand += tempCorrBand
        
        endtime = time.time()
        timeelapsed = endtime-starttime
        print("DONE PROCESSING " + printname + " IN " + str(int(timeelapsed/60))+":"+str(int(timeelapsed%60)))
        return returnASD, TotalEvents, returnNoSkips, returnPSD, returnCorr, returnFFT, returnMegaCorr, returnCorrBand
    
    def onefileresilient(filename,numcores):
        import sys, time
        import numpy as np
        starttime=time.time()
        #We "try" so that if the file crashes the rest of the system keeps going
        try:
            tempASDs, tempEvents, tempNoSkips, tempPSDs, tempCorr, tempFFT, tempMegaCorr, tempCorrBand = myprocessor.onefilefast(filename,numcores)
            return tempASDs, tempEvents, tempNoSkips, tempPSDs, tempCorr, tempFFT, tempMegaCorr, tempCorrBand
        except Exception as e:
            #We log the crash + return empty datasets
            crash=["Error on line {}".format(sys.exc_info()[-1].tb_lineno),"\n",e]
            print(crash)
            timeX=str(time.time()-starttime)
            with open('/dune/app/users/poshung/multirun/crashfiles/CRASH-'+timeX+'.txt', "w") as crashlog:
                mytime = time.gmtime(time.time())
                crashlog.write("Crash Occurred On: " +str(mytime[1])+"/"+str(mytime[2])+"/"+str(mytime[0])+" At " + str(mytime[3])+":"+str(mytime[4])+":"+str(mytime[5]))
                crashlog.write("Crash Occurred While Processing: " + filename[44:60])
                for i in crash:
                    i=str(i)
                    crashlog.write(i)
            numtpcs = 2
            numplanes = 3
            maxwires = 240
            numfreqbins = 15
            ASDlength = 1065 #the length of the ASD of the minwvfm
            returnASD = np.zeros((numtpcs,numplanes,maxwires,ASDlength),dtype=np.float128)
            returnPSD = np.zeros((numtpcs,numplanes,maxwires,ASDlength),dtype=np.float128)
            returnFFT = np.zeros((numtpcs,numplanes,maxwires,ASDlength),dtype=np.complex128)
            returnNoSkips = np.zeros((numtpcs,numplanes,maxwires),dtype=np.uint64)
            returnCorr = np.zeros((numtpcs,numplanes,maxwires,maxwires),dtype=np.float128)
            returnMegaCorr = np.zeros((6*maxwires, 6*maxwires), dtype=float)
            returnCorrBand = np.zeros((numfreqbins, 6*maxwires, 6*maxwires), dtype=float)
            TotalEvents = 0
            return returnASD, TotalEvents, returnNoSkips, returnPSD, returnCorr, returnFFT, returnMegaCorr, returnCorrBand
        
    def multiprocess(filenames,numfiles,NumProcessors):
        import numpy as np
        #Everything is essentially hard coded for 2 tpcs, 3 planes per tpc, a maximum of 240 wires per plane, and a minimum waveform length of 2128
        #If you need to adjust these values (except the waveform length) you will have to write your own sortdata() function
        #Also, you will have to write your own getADC() function based on your file structure
        numtpcs = 2
        numplanes = 3
        maxwires = 240
        minwvfm = 2128
        numfreqbins=15
        ASDlength = 1065 #the length of the ASD of the minwvfm
        SampleSpacing = 0.5e-6 #0.5 microseconds per tick

        TotalEvents = 0
        ASDarray = np.zeros((numtpcs,numplanes,maxwires,ASDlength), dtype=np.float128)
        PSDarray = np.zeros((numtpcs,numplanes,maxwires,ASDlength), dtype=np.float128)
        FFTarray = np.zeros((numtpcs,numplanes,maxwires,ASDlength), dtype=np.complex128)
        noskips = np.zeros((numtpcs,numplanes,maxwires), dtype=np.uint64)
        rCorr = np.zeros((numtpcs,numplanes,maxwires,maxwires), dtype=np.float128)
        MegaCorr = np.zeros((6*maxwires, 6*maxwires), dtype=float)
        CorrBand = np.zeros((numfreqbins, 6*maxwires, 6*maxwires), dtype=float)


        filenames=filenames[1:1+numfiles]
        
        #This is the step that actually runs the processes
        for fileindex in range(len(filenames)):
            tempASDs, tempEvents, tempNoSkips, tempPSDs, tempCorr, tempFFT, tempMegaCorr, tempCorrBand = myprocessor.onefileresilient(filenames[fileindex],NumProcessors)
            TotalEvents += tempEvents
            ASDarray += tempASDs
            PSDarray += tempPSDs
            noskips += tempNoSkips
            rCorr += tempCorr
            FFTarray += tempFFT
            MegaCorr += tempMegaCorr
            CorrBand += tempCorrBand
            print("Events Processed: "+str(TotalEvents))
            
        #We square the correlations because we have to divide by noskips twice because this data is saved along two axes
        rCorr = np.square(rCorr)
        MegaCorr = MegaCorr/TotalEvents
        CorrBand = CorrBand/TotalEvents
        print("BEGINNING FINAL STEP")
        for tpc in range(numtpcs): #This is the averaging out I was talking about
            for plane in range(numplanes):
                for nChannel in range(maxwires):
                    if(noskips[tpc][plane][nChannel] > 0):
                        ASDarray[tpc][plane][nChannel] = ASDarray[tpc][plane][nChannel]/noskips[tpc][plane][nChannel]
                        PSDarray[tpc][plane][nChannel] = PSDarray[tpc][plane][nChannel]/noskips[tpc][plane][nChannel]
                        FFTarray[tpc][plane][nChannel] = FFTarray[tpc][plane][nChannel]/noskips[tpc][plane][nChannel]
                        rCorr[tpc][plane][nChannel] = rCorr[tpc][plane][nChannel]/noskips[tpc][plane][nChannel]
                        rCorr[tpc][plane][:maxwires][nChannel] = rCorr[tpc][plane][:maxwires][nChannel]/noskips[tpc][plane][nChannel] #Have to average along both axes!
        return ASDarray, TotalEvents, noskips, PSDarray, rCorr, FFTarray, MegaCorr, CorrBand
    
    #This needs to apply band pass filters to each waveforms in the 1440x2128 array, then find the correlation matrix of each frequency range
    #This is the finickiest program I've ever tried to work on
    def getcorrelationsbyfrequency(MegaSortedWaveforms, maxfreq=1e6, samplefreq=2e6, numbins=15, order=7):
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.signal import lfilter,butter,bessel,cheby1,filtfilt

        freqstep=maxfreq/numbins
        freqrange = np.arange(0,maxfreq,freqstep)
        returnarray = np.zeros((numbins,len(MegaSortedWaveforms),len(MegaSortedWaveforms)), dtype=float)
        
        for Windex in range(len(MegaSortedWaveforms)):
            subtractmed=np.median(MegaSortedWaveforms[Windex])
            MegaSortedWaveforms[Windex]=MegaSortedWaveforms[Windex]-subtractmed
        
        tempWaveforms = np.zeros((len(MegaSortedWaveforms),len(MegaSortedWaveforms[0])), dtype=int)
        for Windex in range(len(MegaSortedWaveforms)):
            b,a = butter(order,freqrange[1],fs=samplefreq,btype='lowpass')
            tempWaveforms[Windex]=filtfilt(b,a,MegaSortedWaveforms[Windex])
            """if((Windex==0)):
                ADCASD, ADCPSD, ADCFFT = myprocessor.get_single_ASDPSDFFT(tempWaveforms[Windex], samplefreq)
                fig,ax = plt.subplots(2)
                ax[0].plot(tempWaveforms[Windex])
                ax[0].set_title("Filtered Waveform")
                ax[1].plot(ADCPSD)
                ax[1].set_yscale('log')
                ax[1].set_title("Filtered PSD")
                plt.suptitle("Freq Band: " + str(freqstep*0/1e6)+"[1e6 Hz]")
                plt.show()"""
        returnarray[0] = np.corrcoef(tempWaveforms)
        
        for Findex in range(1,len(freqrange)-1):
            tempWaveforms = np.zeros((len(MegaSortedWaveforms),len(MegaSortedWaveforms[0])), dtype=int)
            for Windex in range(len(MegaSortedWaveforms)):
                tempWaveforms[Windex] = myprocessor.butter_bandpass_filter(MegaSortedWaveforms[Windex],freqrange[Findex],freqrange[Findex+1],samplefreq,order)
                """if((Windex==0)):
                    ADCASD, ADCPSD, ADCFFT = myprocessor.get_single_ASDPSDFFT(tempWaveforms[Windex], samplefreq)
                    fig,ax = plt.subplots(2)
                    ax[0].plot(tempWaveforms[Windex])
                    ax[0].set_title("Filtered Waveform")
                    ax[1].plot(ADCPSD)
                    ax[1].set_yscale('log')
                    ax[1].set_title("Filtered PSD")
                    plt.suptitle("Freq Band: " + str(freqstep*Findex/1e6)+"[1e6 Hz]")
                    plt.show()"""
            returnarray[Findex] = np.corrcoef(tempWaveforms)
            
        tempWaveforms = np.zeros((len(MegaSortedWaveforms),len(MegaSortedWaveforms[0])), dtype=int)
        for Windex in range(len(MegaSortedWaveforms)):
            b,a = butter(order,freqrange[len(freqrange)-1],fs=samplefreq,btype='highpass')
            tempWaveforms[Windex]=filtfilt(b,a,MegaSortedWaveforms[Windex])
            """if((Windex==0)):
                ADCASD, ADCPSD, ADCFFT = myprocessor.get_single_ASDPSDFFT(tempWaveforms[Windex], samplefreq)
                fig,ax = plt.subplots(2)
                ax[0].plot(tempWaveforms[Windex])
                ax[0].set_title("Filtered Waveform")
                ax[1].plot(ADCPSD)
                ax[1].set_yscale('log')
                ax[1].set_title("Filtered PSD")
                plt.suptitle("Freq Band: " + str(freqstep*(numbins-1)/1e6)+"[1e6 Hz]")
                plt.show()"""
        returnarray[numbins-1] = np.corrcoef(tempWaveforms)
        return returnarray
    
    def butter_bandpass_filter(data, lowcut, highcut, fs, order):
        from scipy.signal import lfilter,butter,bessel,cheby1,filtfilt
        b, a = butter(order,[lowcut, highcut], fs=fs,btype='bandpass')
        y = filtfilt(b, a, data)
        return y    
        
    #this is the recursive method to return the noise-free mean and std of a waveform
    def recursive_stats(data_arr):
        import numpy as np
        #we start by splitting our data in half, first half and second half
        data_len = len(data_arr)
        first_half = data_arr[0:int(data_len/2)+int(data_len/40)]
        second_half = data_arr[int(data_len/2)-int(data_len/40):data_len]
        
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

    def mask_fast(CurrentWVFM):
        import numpy as np
        #Get the mean/std
        noisestd, noisemean = myprocessor.recursive_stats(CurrentWVFM)
        #Get the peaks
        peaks = np.where(abs(CurrentWVFM-noisemean)>6*noisestd)[0]
        #Then from every peak we trace backward/forward to cut out the whole signal
        cutout = 0
        cutindexes = []
        localmeans = []
        for wvfmindex in peaks:
            i = wvfmindex
            while((abs(CurrentWVFM[i]-noisemean)>1*noisestd)&(i>=0)):
                CurrentWVFM[i] = noisemean
                cutindexes.append(i)
                i-=1
                cutout += 1
            i=wvfmindex+1
            if(i<= 2127):
                while((abs(CurrentWVFM[i]-noisemean)>1*noisestd)):
                    CurrentWVFM[i] = noisemean
                    cutindexes.append(i)
                    i+=1
                    cutout += 1
                    if(i==2128):
                        break
        for index in cutindexes:
            if(index<30):
                localwvfm = CurrentWVFM[0:index+30]
            else: 
                if(index>2097):
                    localwvfm = CurrentWVFM[index-30:2127]
                else:
                    localwvfm = CurrentWVFM[index-30:index+30]
            localmeans.append(np.mean(localwvfm))
        for i in range(len(cutindexes)):
            CurrentWVFM[cutindexes[i]] = localmeans[i]
        return CurrentWVFM, cutout
    def sort_single_data(currentchannel):
        #Gets the tpc/plane/channel from a channel number
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
        #Gets the ASD PSD and FFT
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
        fig.set_figheight(numtpcs*5)
        fig.set_figwidth(numplanes*5)
        
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
        fig.set_figheight(5*numtpcs)
        fig.set_figwidth(5*numplanes)
        
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
                fig2.colorbar(ax2[tpc][plane].pcolormesh(freq,range(maxwires),np.log(AvgASD[tpc][plane]),cmap = cmap,shading='gouraud'))
                if(plane!=2):
                    ax2[tpc][plane].set_ylim(0,200)
                
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
        fig.set_figheight(10*numtpcs)
        fig.set_figwidth(10*numplanes)
        fig.suptitle("Noise Waveform Correlations", y=0.93, size=15)
        fig.set_size_inches(15, 10)
        planenames = ["u", "v", "z"]

        for TPCnum in range(numtpcs):
            for PlaneNum in range(numplanes):
                if(PlaneNum != 2):
                    ax[TPCnum][PlaneNum].set_ylim(0,200)
                    ax[TPCnum][PlaneNum].set_xlim(0,200)
                rCoeff = rCorr[TPCnum][PlaneNum]
                ax[TPCnum][PlaneNum].set_xlabel("Channels 0-240")
                ax[TPCnum][PlaneNum].set_ylabel("Channels 0-240")
                ax[TPCnum][PlaneNum].set_title("TPC #" + str(TPCnum) +" Plane " + planenames[PlaneNum])
                fig.colorbar(ax[TPCnum][PlaneNum].pcolormesh(rCoeff))
                ax[TPCnum][PlaneNum].pcolormesh(np.arange(240),np.arange(240),rCoeff,shading='gouraud')
        plt.show()
    def plotmegacorr(MegaCorr):
        import matplotlib.pyplot as plt
        import numpy as np

        HalfFixedMegaCorr = np.zeros((1280,1440),dtype=float)
        FixedMegaCorr = np.zeros((1280,1280),dtype=float)
        HalfFixedMegaCorr[0:200] = MegaCorr[0:200]
        HalfFixedMegaCorr[200:400] = MegaCorr[240:440]
        HalfFixedMegaCorr[400:640] = MegaCorr[480:720]
        HalfFixedMegaCorr[640:840] = MegaCorr[720:920]
        HalfFixedMegaCorr[840:1040] = MegaCorr[960:1160]
        HalfFixedMegaCorr[1040:1280] = MegaCorr[1200:1440]
        for i in range(1280):
            FixedMegaCorr[i][0:200] = HalfFixedMegaCorr[i][0:200]
            FixedMegaCorr[i][200:400] = HalfFixedMegaCorr[i][240:440]
            FixedMegaCorr[i][400:640] = HalfFixedMegaCorr[i][480:720]
            FixedMegaCorr[i][640:840] = HalfFixedMegaCorr[i][720:920]
            FixedMegaCorr[i][840:1040] = HalfFixedMegaCorr[i][960:1160]
            FixedMegaCorr[i][1040:1280] = HalfFixedMegaCorr[i][1200:1440]
            
        lines = [200,400,640,840,1040,1280]
        linenames = ['200','400','640','840','1040','1280']
        ticks = [40,80,120,160,200,240,280,320,360,400,448,496,544,592,640,680,720,760,800,840,880,920,960,1000,1040,1088,1136,1184,1232,1280]
        planeindexes = [100,300,520,740,940,1160]
        planenames = ['u0','v0','z0','u1','v1','z1']
        plt.figure(figsize=(15,12))
        plt.title('Noise Waveform Correlations',fontsize=18)
        plt.colorbar(plt.pcolormesh(FixedMegaCorr))
        plt.vlines(ticks,0,1280,linestyles='dashed',colors='white')
        plt.hlines(ticks,0,1280,linestyles='dashed',colors='white')
        plt.vlines(lines,0,1280, colors="orange")
        plt.hlines(lines,0,1280, colors="orange")
        plt.xticks(lines,linenames,minor=True,fontsize=8)
        plt.yticks(lines,linenames,minor=True,fontsize=8)
        plt.xticks(planeindexes,planenames,minor=False,fontsize=12)
        plt.yticks(planeindexes,planenames,minor=False,fontsize=12)
        plt.ylabel('Channels 0-1280',fontsize=15)
        plt.xlabel('Channels 0-1280',fontsize=15)
        plt.pcolormesh(np.arange(1280),np.arange(1280),FixedMegaCorr,shading='gouraud')
        plt.show()
        
    def plotmegacorrbyband(corrarray):
        import matplotlib.pyplot as plt
        import numpy as np
        lines = [200,400,640,840,1040,1280]
        linenames = ['200','400','640','840','1040','1280']
        ticks = [40,80,120,160,200,240,280,320,360,400,448,496,544,592,640,680,720,760,800,840,880,920,960,1000,1040,1088,1136,1184,1232,1280]
        planeindexes = [100,300,520,740,940,1160]
        planenames = ['u0','v0','z0','u1','v1','z1']
        numbins = len(corrarray)
        maxfreq=1e6
        freqstep=maxfreq/numbins
        freqrange=np.arange(0,maxfreq,freqstep)
        freqrange=np.append(freqrange,maxfreq)
        fig,ax = plt.subplots(len(corrarray),figsize=(15,96))
        for freqband in range(len(corrarray)):
            MegaCorr = corrarray[freqband]
            HalfFixedMegaCorr = np.zeros((1280,1440),dtype=float)
            FixedMegaCorr = np.zeros((1280,1280),dtype=float)
            HalfFixedMegaCorr[0:200] = MegaCorr[0:200]
            HalfFixedMegaCorr[200:400] = MegaCorr[240:440]
            HalfFixedMegaCorr[400:640] = MegaCorr[480:720]
            HalfFixedMegaCorr[640:840] = MegaCorr[720:920]
            HalfFixedMegaCorr[840:1040] = MegaCorr[960:1160]
            HalfFixedMegaCorr[1040:1280] = MegaCorr[1200:1440]
            for i in range(1280):
                FixedMegaCorr[i][0:200] = HalfFixedMegaCorr[i][0:200]
                FixedMegaCorr[i][200:400] = HalfFixedMegaCorr[i][240:440]
                FixedMegaCorr[i][400:640] = HalfFixedMegaCorr[i][480:720]
                FixedMegaCorr[i][640:840] = HalfFixedMegaCorr[i][720:920]
                FixedMegaCorr[i][840:1040] = HalfFixedMegaCorr[i][960:1160]
                FixedMegaCorr[i][1040:1280] = HalfFixedMegaCorr[i][1200:1440]
                
            
            ax[freqband].set_title(('Noise Waveform Correlations for the ' + str(freqrange[freqband]/1e6)+ '[1e6 Hz] to ' + str(freqrange[freqband+1]/1e6)+'[1e6Hz] Band'),fontsize=18)
            fig.colorbar(ax[freqband].pcolormesh(FixedMegaCorr))
            ax[freqband].vlines(ticks,0,1280,linestyles='dashed',colors='white')
            ax[freqband].hlines(ticks,0,1280,linestyles='dashed',colors='white')
            ax[freqband].vlines(lines,0,1280, colors="orange")
            ax[freqband].hlines(lines,0,1280, colors="orange")
            ax[freqband].set_xticks(lines,linenames,minor=True,fontsize=8)
            ax[freqband].set_yticks(lines,linenames,minor=True,fontsize=8)
            ax[freqband].set_xticks(planeindexes,planenames,minor=False,fontsize=12)
            ax[freqband].set_yticks(planeindexes,planenames,minor=False,fontsize=12)
            ax[freqband].set_ylabel('Channels 0-1280',fontsize=15)
            ax[freqband].set_xlabel('Channels 0-1280',fontsize=15)
            ax[freqband].pcolormesh(np.arange(1280),np.arange(1280),FixedMegaCorr,shading='gouraud')
        plt.show()