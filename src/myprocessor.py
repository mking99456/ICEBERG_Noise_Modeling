#myprocessor class contains all of the fields and methods to take an ICEBERG data file
#and create a noise model in an npz file.

class myprocessor:
    #Contains Methods:
    #   ###make this one### combineNoiseModel
    #   updateNoiseModel
    #   createNoiseModel(filenames,savefile,numcores = 1):
    #   multiprocess(filenames,numcores = 1)
    #   onefileresilient(filename,numcores = 1,errorpath='./'):
    #   onefilefast(infile_name, numcores = 1)
    #   processevent(ADCarr,Channelarr)
    #   getcorrelationsbyfrequency(Waveforms)
    #   butter_bandpass_filter(data, lowcut, highcut, fs, order)
    #   recursive_stats(data_arr)
    #   mask_fast(CurrentWVFM)
    #   get_single_PSD(waveform)

    #Global Field Variables used in every method of this class:
    numchannels = 200*4 + 240*2 #Number of wires in the TPC
    minwvfm = 2128 #Minimum number of time ticks present in ICEBERG data arrays we're analyzing.
    PSDlength = 1065 #the length of the PSD of the minwvfm
    SampleSpacing = 0.5e-6 #0.5 microseconds per tick
    numfreqbins = 10 #Dividing up our correlation plots into different frequency bins.

    #updateNoiseModel
    #I want to build in an option to combine noise models

    def updateNoiseModel(filenames,oldSavefile,newSavefile,numcores = 1):
        import numpy as np
        myarray = np.load(oldSavefile)
        CurrentPSD = myarray['PSD']
        CurrentrCorr = myarray['rCorr']
        CurrentCorrBand = myarray['CorrBand']
        CurrentBothLive = myarray['BothLive']
        CurrentKurtMasking = myarray['KurtMasking']
        CurrentTotalEvents = myarray['TotalEvents']

        NewPSD,NewrCorr,NewCorrBand,NewBothLive,NewKurtMasking,NewTotalEvents = myprocessor.multiprocess(filenames, numcores = numcores)

        #De-normalize:
        CurrentNoSkips = np.diagonal(CurrentBothLive)
        CurrentPSD = np.transpose(np.multiply(np.transpose(CurrentPSD),CurrentNoSkips))
        CurrentrCorr = np.multiply(CurrentrCorr,CurrentBothLive)
        CurrentCorrBand = np.multiply(CurrentCorrBand,CurrentBothLive)

        NewNoSkips = np.diagonal(NewBothLive)
        NewPSD = np.transpose(np.multiply(np.transpose(NewPSD),NewNoSkips))
        NewrCorr = np.multiply(NewrCorr,NewBothLive)
        NewCorrBand = np.multiply(NewCorrBand,NewBothLive)

        #Combine
        TotalPSD = CurrentPSD + NewPSD
        TotalrCorr = CurrentrCorr + NewrCorr
        TotalCorrBand = CurrentCorrBand + NewCorrBand
        TotalBothLive = CurrentBothLive + NewBothLive
        TotalKurtMasking = np.concatenate((CurrentKurtMasking,NewKurtMasking))
        TotalEvents = CurrentTotalEvents + NewTotalEvents

        #Re-normalize:
        TotalNoSkips = np.diagonal(TotalBothLive)
        PSDs = np.transpose(np.divide(np.transpose(TotalPSD),TotalNoSkips,out = np.zeros_like(np.transpose(TotalPSD)), where = TotalNoSkips!=0))
        rCorr = np.divide(TotalrCorr,TotalBothLive,out = np.zeros_like(TotalrCorr), where = TotalBothLive!=0)
        CorrBand = np.divide(TotalCorrBand,TotalBothLive, out = np.zeros_like(TotalCorrBand), where = TotalBothLive!=0)

        np.savez(newSavefile,PSD = PSDs, rCorr = rCorr, CorrBand = CorrBand, BothLive = TotalBothLive, KurtMasking = TotalKurtMasking, TotalEvents = TotalEvents)

    #createNoiseModel
    def createNoiseModel(filenames,savefile,numcores = 1):
        import numpy as np
        #Process the file list
        PSDs,rCorr,CorrBand,BothLive,KurtMasking,TotalEvents = myprocessor.multiprocess(filenames, numcores = numcores)
        np.savez(savefile,PSD = PSDs, rCorr = rCorr, CorrBand = CorrBand, BothLive = BothLive, KurtMasking = KurtMasking, TotalEvents = TotalEvents)

    #multiprocess
    #
    #multiprocess takes in an array of filenames, the number of files to process, and the number
    #of processors used to process the data. It outputs a "noise model"
    #Noise Model averaged over the events in the files
    #
    #called by multirun.py
    #calls myp.onefileresilient
    #
    #INPUTS:
    # filenames is a list of strings
    # numcores is the number of parallel processors to use when processing the data
    #
    #OUTPUTS:
    # PSD[Channel][freq]
    # rCorr
    # CorrBand
    # BothLive
    # KurtMasking
    # TotalEvents
    #
    def multiprocess(filenames,numcores = 1):
        import numpy as np

        PSD = np.zeros((myprocessor.numchannels,myprocessor.PSDlength),dtype=float)
        BothLive = np.zeros((myprocessor.numchannels,myprocessor.numchannels),dtype=np.uint64) #Denominator for averaging corr plots
        rCorr = np.zeros((myprocessor.numchannels,myprocessor.numchannels),dtype=float)
        CorrBand = np.zeros((myprocessor.numfreqbins,myprocessor.numchannels,myprocessor.numchannels),dtype=float)
        KurtMasking = []
        TotalEvents = 0

        #filenames=filenames[0:numfiles]
        
        #This is the step that actually runs the processes
        for fileindex in range(len(filenames)):
            tempPSD, tempCorr,tempCorrBand, tempBothLive, tempKurt, tempEvents = myprocessor.onefileresilient(filenames[fileindex],numcores = numcores)
            PSD += tempPSD
            BothLive += tempBothLive
            rCorr += tempCorr
            CorrBand += tempCorrBand

            if len(KurtMasking) == 0:
                KurtMasking = tempKurt
            else:
                np.concatenate((KurtMasking,tempKurt),axis = 0)

            TotalEvents += tempEvents
            print("Events Processed: "+str(TotalEvents))

        #Now we normalize for the average
        no_skips = np.diagonal(BothLive)

        #If I don't put in the stipulation of BothLive!=0, will it give me a nan to make plots?
        #rCorr = np.divide(rCorr,BothLive)
        rCorr = np.divide(rCorr,BothLive,out = np.zeros_like(rCorr), where = BothLive!=0)

        #The broadcasting works here, verified with small test example
        CorrBand = np.divide(CorrBand,BothLive, out = np.zeros_like(CorrBand), where = BothLive!=0)
        #CorrBand = np.divide(CorrBand,BothLive)

        #Note: ValueError: operands could not be broadcast together with shapes (myprocessor.numchannels,numfreq) (myprocessor.numchannels,)
        # Specifically, cannot broadcast (2,3) (2,)
        # But you can broadcast (2,3) (3,) [this would be like dividing frequency-wise]
        # The below solution works, verified with a small test example.

        PSD = np.transpose(np.divide(np.transpose(PSD),no_skips,out = np.zeros_like(np.transpose(PSD)), where = no_skips!=0))
        #PSD = np.transpose(np.divide(np.transpose(PSD),no_skips))

        return PSD,rCorr,CorrBand,BothLive,KurtMasking,TotalEvents

    #onefileresilient
    #Performs tasks of onefilefast, but builds in a procedure for the case of an error.
    #
    def onefileresilient(filename,numcores = 1,errorpath='./'):
        import sys, time
        import numpy as np
        starttime=time.time()
        #We "try" so that if the file crashes the rest of the system keeps going
        try:
            tempPSD, tempCorr,tempCorrBand, tempBothLive, tempKurt, tempEvents = myprocessor.onefilefast(filename,numcores = numcores)
            return tempPSD, tempCorr,tempCorrBand, tempBothLive, tempKurt, tempEvents
        except Exception as e:
            #We log the crash + return empty datasets
            crash=["Error on line {}".format(sys.exc_info()[-1].tb_lineno),"\n",e]
            print(crash)
            timeX=str(time.time()-starttime)
            with open(errorpath+'/CRASH-'+timeX+'.txt', "w") as crashlog:
                mytime = time.gmtime(time.time())
                crashlog.write("Crash Occurred On: " +str(mytime[1])+"/"+str(mytime[2])+"/"+str(mytime[0])+" At " + str(mytime[3])+":"+str(mytime[4])+":"+str(mytime[5]))
                crashlog.write("Crash Occurred While Processing: " + filename[44:60])
                for i in crash:
                    i=str(i)
                    crashlog.write(i)
            returnPSD = np.zeros((myprocessor.numchannels,myprocessor.PSDlength),dtype=float)
            returnBothLive = np.zeros((myprocessor.numchannels,myprocessor.numchannels),dtype=np.uint64) #Denominator for averaging corr plots
            returnCorr = np.zeros((myprocessor.numchannels,myprocessor.numchannels),dtype=float)
            returnCorrBand = np.zeros((myprocessor.numfreqbins,myprocessor.numchannels,myprocessor.numchannels),dtype=float)
            returnKurt = []
            returnEvents = 0
            return returnPSD, returnCorr, returnCorrBand, returnBothLive, returnKurt, returnEvents
    
    #onefilefast
    #
    #onefilefast takes as an input the path to one data file and returns the
    #"noise model" information from it. Everything is summed, not averaged.
    #
    #called by onefileresilient
    #
    #INPUTS:
    # infile_name - string containing name of input .root ICEBERG data file
    # numcores - Number of cores for parallel processing 
    #
    #OUTPUTS:
    # Summed over events:
    # PSD[GlobalChannel][freq] - numpy array of Power Spectral Densities for each channel
    # Corr[GlobalChannel][GlobalChannel] - numpy array of 2-wire same-time correlations
    # CorrBandPass[freqbin][GlobalChannel][GlobalChannel] - numpy array binning correlations by frequency
    # BothLive[GlobalChannel][GlobalChannel] - Boolean matrix: 1 means both wires contribute data to this event
    #
    # KurtMasking - array of pairs of floats - kurtosis for a waveform before and after masking - quality metric
    #   Stores both kurtosis values before and after masking for every event.
    # TotalEvents - int - number of nonempty events in the data file
    #
    def onefilefast(infile_name, numcores = 1):
        import uproot
        import awkward as ak
        import numpy as np
        import time
        from concurrent.futures import ProcessPoolExecutor
        
        starttime = time.time()
        exe = ProcessPoolExecutor(numcores)

        #Data to return
        #Returns sum of these values over each event
        sumPSD = np.zeros((myprocessor.numchannels,myprocessor.PSDlength),dtype=float)
        sumBothLive = np.zeros((myprocessor.numchannels,myprocessor.numchannels),dtype=np.uint64) #Denominator for averaging corr plots
        sumCorr = np.zeros((myprocessor.numchannels,myprocessor.numchannels),dtype=float)
        sumCorrBand = np.zeros((myprocessor.numfreqbins,myprocessor.numchannels,myprocessor.numchannels),dtype=float)
        TotalEvents = 0
        KurtMasking = []
        
        infile = uproot.open(infile_name)
        printname = infile_name[44:60] #be careful in hardcoding this...
        events = infile['Events']
    
        fADC = events['raw::RawDigits_tpcrawdecoder_daq_RunIcebergRawDecoder./raw::RawDigits_tpcrawdecoder_daq_RunIcebergRawDecoder.obj/raw::RawDigits_tpcrawdecoder_daq_RunIcebergRawDecoder.obj.fADC']
        fChannel = events['raw::RawDigits_tpcrawdecoder_daq_RunIcebergRawDecoder./raw::RawDigits_tpcrawdecoder_daq_RunIcebergRawDecoder.obj/raw::RawDigits_tpcrawdecoder_daq_RunIcebergRawDecoder.obj.fChannel']

        #Switch to int16 ASAP in order to minimize memory usage. Doesn't really work though!
        ADCarr = fADC.array(library = "ak")
        ak.values_astype(ADCarr, np.uint16)
        Channelarr = fChannel.array(library = "ak")
        ak.values_astype(Channelarr, np.uint16)

        #We end up with lots of empty events, so when we convert from an awk array to a np array we only take the events with data
        
        #Find out if events have data
        counter = []
        for index in range(len(ADCarr)):
            if(ak.count(ADCarr[index]) >= myprocessor.minwvfm): #If there is at least one active channel...
                counter.append(index)
        precutlength = len(ADCarr)
        ADCarr = ADCarr[counter] #cut out empty events
        Channelarr = Channelarr[counter]
        
        print("BEGAN PROCESSING: " + printname + " CONTAINING "+ str(TotalEvents) + " EVENTS, CUT FROM: " + str(precutlength))
        #We iterate through the events and the channels, inspecting and processing one waveform at a time
        
        debug = True
        count = 0

        for tempPSD, tempCorr, tempCorrBand, tempBothLive, tempkurt in exe.map(myprocessor.processevent, ADCarr, Channelarr):
            count+=1
            if(debug): print("In onefilefast, processing event "+str(count))
            
            sumPSD += tempPSD
            sumCorr += tempCorr
            sumCorrBand += tempCorrBand
            sumBothLive += tempBothLive
            TotalEvents += 1
            KurtMasking.append(tempkurt)
        
        endtime = time.time()
        timeelapsed = endtime-starttime
        print("DONE PROCESSING " + printname + " IN " + str(int(timeelapsed/60))+":"+str(int(timeelapsed%60)))
        return sumPSD, sumCorr, sumCorrBand, sumBothLive, KurtMasking, TotalEvents

    #processevent
    #
    #This method takes in the awkward arrays for ADC values of the active channels for an event 
    #And returns the "noise model" information for that event - see outputs
    #
    #called by onefilefast
    #
    #INPUTS:
    # ADCarr[nchannel][timetick] - Awkward array of ADC values 
    # Channelarr[nchannel] - Awkward array of Channel numbers 
    #
    # Taken directly from a .root ICEBERG Data File.
    #
    #OUTPUTS:
    # PSD[GlobalChannel][freq] - numpy array of Power Spectral Densities for each channel
    # Corr[GlobalChannel][GlobalChannel] - numpy array of 2-wire same-time correlations
    # CorrBandPass[freqbin][GlobalChannel][GlobalChannel] - numpy array binning correlations by frequency
    # BothLive[GlobalChannel][GlobalChannel] - Boolean matrix: 1 means both wires contribute data to this event
    # KurtMasking - array of 2 floats - kurtosis for a waveform before and after masking - quality metric
    #
    def processevent(ADCarr,Channelarr):
        import numpy as np
        import awkward as ak
        from scipy.stats import kurtosis

        #We sort the waveforms by TPC/Plane so that we can find the correlation coefficients of all of them
        Waveforms = np.zeros((myprocessor.numchannels,myprocessor.minwvfm),dtype=float)
        returnPSD = np.zeros((myprocessor.numchannels,myprocessor.PSDlength),dtype=float)
        returnBothLive = np.zeros((myprocessor.numchannels,myprocessor.numchannels),dtype=np.uint64) #Denominator for averaging corr plots
        returnCorr = np.zeros((myprocessor.numchannels,myprocessor.numchannels),dtype=float)

        returnkurt = np.zeros((myprocessor.numchannels,2),dtype = float)
        #postkurtosis = 0
        #postmaskstd = 0
        #premaskstd = 0
        #cutouts = 0

        for nChannel in range(len(ADCarr)):
            unmaskedWVFM = np.array(ak.to_numpy(ADCarr[nChannel]))
            GlobalChannel = Channelarr[nChannel]
            CurrentWVFM = unmaskedWVFM[0:myprocessor.minwvfm] #We want all of our waveforms to have same length

            prekurtosis = kurtosis(CurrentWVFM) #kurtosis of waveform before masking
            #Mask out the clock signals and other triggers
            CurrentWVFM, tempcutouts = myprocessor.mask_fast(CurrentWVFM)
            #cutouts += tempcutouts
            postkurtosis = kurtosis(CurrentWVFM)
            #postmaskstd += np.std(CurrentWVFM)

            Waveforms[GlobalChannel] = CurrentWVFM

            #Get TPC/Plane/Channel
            #TPCNum, PlaneNum, ChannelNum = myprocessor.sort_single_data(GlobalChannel)
            #Sort our current waveform by tpc and plane
            #SortedWaveforms[TPCNum][PlaneNum][ChannelNum] = CurrentWVFM   

            #Get ASD/PSD/FFT
            ADCPSD = myprocessor.get_single_PSD(CurrentWVFM)

            #Input our current ASD/PSD/FFT where it belongs
            returnPSD[GlobalChannel] += ADCPSD

            #We keep track of how many events contributed to each channel sum so that we can average correctly later
            for mChannel in range(len(ADCarr)):
                mGlobalChannel = Channelarr[mChannel]
                returnBothLive[mGlobalChannel][GlobalChannel] += 1 #Both wires are present

        returnCorr = np.corrcoef(Waveforms)
        corrbandpass = myprocessor.getcorrelationsbyfrequency(Waveforms)

        returnkurt[GlobalChannel] = [prekurtosis,postkurtosis]

        return returnPSD, returnCorr, corrbandpass, returnBothLive, returnkurt
    
    #multiMCprocess
    #
    #Takes in a file list of simulated event files and the BothLive matrix from the noise model used to simulate them.
    #Returns the average PSD, rCorr matrix, and CorrBand
    #
    #INPUTS:
    # filenames is a list of strings - .npz file paths containing simulated events
    # BothLive is the BothLive matrix from the noise model used to create the sim - used in normalization
    def multiMCprocess(filenames,BothLive,numcores = 1):
        import numpy as np
        import time
        from concurrent.futures import ProcessPoolExecutor

        PSD = np.zeros((myprocessor.numchannels,myprocessor.PSDlength),dtype=float)
        rCorr = np.zeros((myprocessor.numchannels,myprocessor.numchannels),dtype=float)
        CorrBand = np.zeros((myprocessor.numfreqbins,myprocessor.numchannels,myprocessor.numchannels),dtype=float)
        TotalEvents = 0

        #This is the step that actually runs the processes
        for fileindex in range(len(filenames)):

            starttime = time.time()
            exe = ProcessPoolExecutor(numcores)

            Simulated_File = np.load(filenames[fileindex])
            WaveformsByEvent = Simulated_File['arr_0']
            #Waveforms = Simulated_File['Waveforms']

            for tempPSD, tempCorr,tempCorrBand in exe.map(myprocessor.processMCevent, WaveformsByEvent):
                PSD += tempPSD
                rCorr += tempCorr
                CorrBand += tempCorrBand
                TotalEvents += 1
            
        #BothLive will be a 1 for good channels, 0 for bad channels, scaled by number of events.
        BothLive = TotalEvents*np.divide(BothLive,BothLive,out = np.zeros_like(BothLive), where = BothLive!=0,casting = 'unsafe')

        rCorr = np.divide(rCorr,BothLive,out = np.zeros_like(rCorr), where = BothLive!=0)
        CorrBand = np.divide(CorrBand,BothLive, out = np.zeros_like(CorrBand), where = BothLive!=0)
        PSD /= TotalEvents

        return PSD,rCorr,CorrBand,TotalEvents

    #processMCevent
    def processMCevent(Waveforms):
        import numpy as np

        #We sort the waveforms by TPC/Plane so that we can find the correlation coefficients of all of them
        returnPSD = np.zeros((myprocessor.numchannels,myprocessor.PSDlength),dtype=float)

        for GlobalChannel in range(len(Waveforms)):
            CurrentWVFM = Waveforms[GlobalChannel]
            returnPSD[GlobalChannel] = myprocessor.get_single_PSD(CurrentWVFM)

        returnCorr = np.corrcoef(Waveforms)
        corrbandpass = myprocessor.getcorrelationsbyfrequency(Waveforms)

        return returnPSD, returnCorr, corrbandpass

    
    #This needs to apply band pass filters to each waveforms in the 1440x2128 array, then find the correlation matrix of each frequency range
    #This is the finickiest program I've ever tried to work on
    #-Poshu

    #getcorrelationsbyfrequency
    #
    # This function takes in waveforms for one event and filters them
    # Using a digital Butterworth bandpass filter
    # pass band is .1 MHz in width, with stop band everything outside another .1 MHz radius
    # Lowpass filter at the lowest bin, highpass filter at the highest bin, bandpass in middle.
    #
    #
    #INPUTS:
    # Waveforms[GlobalChannel][Timetick]
    #
    #OUTPUTS:
    # freqCorr[freqbandindex][GlobalChannel][GlobalChannel]
    #
    def getcorrelationsbyfrequency(Waveforms):
        import numpy as np
        from scipy.signal import butter,filtfilt,buttord
        
        #maxfreq is the Nyquist frequency, samplefreq is the sampling frequency, both in Hz
        maxfreq = 1/(2*myprocessor.SampleSpacing)
        samplefreq = 1/myprocessor.SampleSpacing

        #How did Poshu decide on these values for the filter?
        gpass = 5
        gstop = 80

        #This creates evenly spaced bins at intervals of maxfreq / numfreqbins from [0,maxfreq]
        freqrange = np.linspace(0,maxfreq,num = myprocessor.numfreqbins+1)

        #Initialize the return array
        freqCorr = np.zeros((myprocessor.numfreqbins,len(Waveforms),len(Waveforms)), dtype=float)
        
        #Subtract the median value for every waveform
        for nchannel in range(len(Waveforms)):
            Waveforms[nchannel]=Waveforms[nchannel]-np.median(Waveforms[nchannel])               
        
        #From Butterworth filter documentation:
        #Buttord returns the order of the lowest order digital or analog Butterworth filter
        # that loses no more than gpass dB in the passband
        # and has at least gstop dB attenuation in the stopband.
        #
        # We use a digital Butterworth Filter
        # pass band is .1 MHz in width, with stop band everything outside another .1 MHz radius
        # Lowpass filter at the lowest bin, highpass filter at the highest bin, bandpass in middle.
        #
        tempWaveforms = np.zeros((len(Waveforms),len(Waveforms[0])), dtype=int)
        for nchannel in range(len(Waveforms)):
            order, wn = buttord(wp=freqrange[1],ws=freqrange[2],gpass=gpass,gstop=gstop,fs=samplefreq)
            b,a = butter(order,wn,fs=samplefreq,btype='lowpass')
            tempWaveforms[nchannel]=filtfilt(b,a,Waveforms[nchannel])
        freqCorr[0] = np.corrcoef(tempWaveforms)
        
        for Findex in range(1,myprocessor.numfreqbins-1):
            tempWaveforms = np.zeros((len(Waveforms),len(Waveforms[0])), dtype=int)
            order, wn = buttord(wp=[freqrange[Findex],freqrange[Findex+1]],ws=[freqrange[Findex-1],freqrange[Findex+2]],gpass=gpass,gstop=gstop,fs=samplefreq)
            for nchannel in range(len(Waveforms)):
                tempWaveforms[nchannel] = myprocessor.butter_bandpass_filter(Waveforms[nchannel],wn[0],wn[1],samplefreq,order)
            freqCorr[Findex] = np.corrcoef(tempWaveforms)
            
        tempWaveforms = np.zeros((len(Waveforms),len(Waveforms[0])), dtype=int)
        for nchannel in range(len(Waveforms)):
            order,wn = buttord(wp=freqrange[len(freqrange)-2],ws=freqrange[len(freqrange)-3],gpass=gpass,gstop=gstop,fs=samplefreq)
            b,a = butter(order,wn,fs=samplefreq,btype='highpass')
            tempWaveforms[nchannel]=filtfilt(b,a,Waveforms[nchannel])
        freqCorr[myprocessor.numfreqbins-1] = np.corrcoef(tempWaveforms)

        return freqCorr

    #butter_bandpass_filter
    #Helper function for the bandpass filter above
    def butter_bandpass_filter(data, lowcut, highcut, fs, order):
        from scipy.signal import lfilter,butter,bessel,cheby1,filtfilt
        b, a = butter(order,[lowcut, highcut], fs=fs,btype='bandpass')
        y = filtfilt(b, a, data)
        return y    

    #recursive_stats
    #    
    #this is the recursive method to return the noise-free mean and std of a waveform
    #
    #called by mask_fast
    #
    #INPUTS:
    # data_arr is a single waveform: waveform[timetick]
    #
    #OUTPUTS:
    # Standard Deviation and mean of peakless region
    #
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

    #mask_fast
    #
    #Masks out signals in ICEBERG data
    #
    #INPUTS:
    # CurrentWVFM[timetick]
    #
    #OUTPUTS:
    # CurrentWVFM[timetick] (signals masked)
    # Cutout = number of data points which have been masked as the mean
    #
    def mask_fast(CurrentWVFM):
        import numpy as np
        #Get the mean/std
        noisestd, noisemean = myprocessor.recursive_stats(CurrentWVFM)
        #Get the peaks
        peaks = np.where(abs(CurrentWVFM-noisemean)>12*noisestd)[0]
        #Then from every peak we trace backward/forward to cut out the whole signal
        cutout = 0
        cutindexes = []
        localmeans = []
        for wvfmindex in peaks:
            i = wvfmindex
            continuetrue=False
            while(((abs(CurrentWVFM[i]-noisemean)>1*noisestd)&(i>=0))|(continuetrue)):
                CurrentWVFM[i] = noisemean
                cutindexes.append(i)
                i-=1
                cutout += 1
                continuetrue=False
                if(i>=1):
                    if(abs(CurrentWVFM[i-1]-noisemean)>1*noisestd):
                        continuetrue=True
            i=wvfmindex+1
            if(i<= 2127):
                while((abs(CurrentWVFM[i]-noisemean)>1*noisestd)|(continuetrue)):
                    CurrentWVFM[i] = noisemean
                    cutindexes.append(i)
                    i+=1
                    cutout += 1
                    if(i==2128):
                        break
                    continuetrue=False
                    if(i<=2126):
                        if(abs(CurrentWVFM[i+1]-noisemean)>1*noisestd):
                            continuetrue=True
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

    def get_single_PSD(waveform):
        #Gets the ASD PSD and FFT
        import numpy as np
        ADCFFT = np.fft.rfft(waveform)
        N = len(waveform)
        T = myprocessor.SampleSpacing*N
        ADCFFT[0] = 0 #Zero out the pedestal
        ADCPSD = 2*T/N**2 * np.abs(ADCFFT)**2
        return ADCPSD