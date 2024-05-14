#Read in an npz file containing the power spectra for channels in the ICEBERG TPC
#Return waveforms simulated using an MC method

#To simulate one waveform: getNoiseWaveform(PSD) where PSD is the noise 
#To simulate an event: simulateEvent(npz_file) where npz_file contains the noise model PSD

#The noise simulation code follows M. Carrettoni and O. Cremonesi, "Generation of Noise Time Series with arbitrary Power Spectrum" and works as follows:
#1. Start with our noise model in the form of a power spectrum. (P(0) = 0)
#2. Divide out the normalization (alpha) used for the power spectrum.
#3. Let F(w) = the square root of the unnormalized power spectrum times uniformly generated random phases.
#4. Let f(t) be the inverse real fast fourier transform of F
#5. Let l be a tunable overlap rate of signals. I have it tuned to allow for 50-100 copies of the signal waveform to be added to itself.
#6. Generate time delays according to a Poisson distribution. Stop generating when the total delayed time is greater than the domain of f(t), T (Length in time of a waveform).
#7. Generate random amplitudes a_k from Norm(0,1/(l*T))
#8. Our noise waveform is n(t) = sum_k a_k*f(t-t_k)

class simulate_noise:

    minwvfm = 2128
    PSD_length = 1065
    SampleSpacing = 0.5e-6
    numchannels = 200*4 + 240*2 

    #Input the name of the npz file that contains the PSD information
    #Output an array containing the PSD:
    #PSDs[iTPC][iPlane][iChannel][iFrequency]
    def PSDfromNPZ(npz_file):
        import numpy as np
        myarrays = np.load(npz_file)
        #arr_0 = PSDs #arr_1 = rCorr #arr_2 = CorrBand #arr_3 = BothLive #arr_4=KurtMasking #arr_5=TotalEvents
        return myarrays['PSD']

    #This method takes in a vector of PSD values (same length)
    #It returns a simulated noise waveform with the desired length (usually 2*len(PSD))
    #Formatted for the ICEBERG TPC
    def getNoiseWaveform(PSD,easy_sim = False):
        if (len(PSD)==0): return [0],[0] #avoids processing empty channels

        import numpy as np
        import scipy

        T = simulate_noise.minwvfm * simulate_noise.SampleSpacing #time length of the waveform
        alpha = 2*T/simulate_noise.minwvfm**2 #our choice of normalization for the PSD

        theta = np.random.rand(len(PSD))*2*np.pi #Generate random phases for the FFT
        theta[0]=0 #the first entry needs to be real for irfft

        #When taking the fourier transform, we divide out by the normalization first
        F = np.sqrt(PSD/alpha)*(np.cos(theta) + 1j*np.sin(theta)) #Our fourier transform
        f = np.fft.irfft(F,simulate_noise.minwvfm) #inverse real fourier transform
        time = np.arange(simulate_noise.minwvfm)*simulate_noise.SampleSpacing #time axis

        if(easy_sim): return f,time

        l = 5*10**(-2) #tunable overlap rate of the signals, in Hz
        #l cares about time ticks, not absolute time.

        #now we generate the time delays according to Poisson statistics
        time_delays = [0]
        sum = 0
        while sum < simulate_noise.minwvfm:
            index = 0
            R = np.random.rand()
            sum += int(-np.log(1-R)/l)
            time_delays.append(sum)
        
        #Amplitudes for each term we add of f to create the noise.
        Amplitude_Variance = 1/(l*simulate_noise.minwvfm)
        amplitudes = np.random.normal(0, np.sqrt(Amplitude_Variance), len(time_delays))
        
        #create the waveform
        waveform = np.zeros(simulate_noise.minwvfm)
        for k in range(len(time_delays)):
            waveform += amplitudes[k]*np.roll(f,time_delays[k])

        return waveform, time
    
    #npz_file contains existing noise model PSD (see above method)
    #returns an ICEBERG event's noise waveforms and PSDs of those waveforms.
    #New_PSD[iChannel][frequency]
    #New_Waveform[iChannel][time_tick]
    def simulateEvent(npz_file,easy_sim = False,add_correlations=False):
        import numpy as np
        from myprocessor import myprocessor as myp
        import matplotlib.pyplot as plt

        PSDs = simulate_noise.PSDfromNPZ(npz_file)
        PSDlength = len(PSDs[0]) #should be rectangular and = 1065

        #Convenient naming convention for me when normalizing
        N = simulate_noise.minwvfm
        T = simulate_noise.SampleSpacing * N

        New_Waveform = np.zeros((simulate_noise.numchannels,simulate_noise.minwvfm),dtype=float)
        New_PSD = np.zeros((simulate_noise.numchannels,PSDlength),dtype=float)

        #I would like to figure out how to make this step faster!!!
        for iChannel in range(simulate_noise.numchannels):
            if(len(PSDs[iChannel]) != 0):
                New_Waveform[iChannel], time = simulate_noise.getNoiseWaveform(PSDs[iChannel],easy_sim = easy_sim)
                New_PSD[iChannel] = myp.get_single_PSD(New_Waveform[iChannel])

        #CORRELATIONS
        #This is where we correlate the noise based on our noise model if we choose to.
        if(add_correlations):
            #add correlations
            print()

        return New_Waveform, New_PSD
    
    def simulateMultipleEvents(npz_file, savefile, nEvents, easy_sim = False,add_correlations=False):
        import numpy as np
        Waveform_Array = np.zeros((nEvents,simulate_noise.numchannels,simulate_noise.minwvfm),dtype=float)
        PSD_Array = np.zeros((nEvents,simulate_noise.numchannels,simulate_noise.PSD_length),dtype=float)

        for i in range(nEvents):
            Waveform_Array[i], PSD_Array[i] = simulate_noise.simulateEvent(npz_file,easy_sim = easy_sim,add_correlations=add_correlations)
        
        np.savez(savefile,Waveforms = Waveform_Array,PSDs = PSD_Array)
        return Waveform_Array,PSD_Array