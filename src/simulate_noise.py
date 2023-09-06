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

    #Input the name of the npz file that contains the PSD information
    #Output an array containing the PSD:
    #PSDs[iTPC][iPlane][iChannel][iFrequency]
    def PSDfromNPZ(npz_file):
        myarrays = np.load(npz_file)
        #arr_0 = ASDs #arr_1 = PSDs #arr_2 = total events #arr_3 = noskips #arr_4=rcorr #arr_5=FFTs
        return myarrays['arr_1']

    #This method takes in a vector of PSD values (same length)
    #It returns a simulated noise waveform with the desired length (usually 2*len(PSD))
    #Formatted for the ICEBERG TPC
    def getNoiseWaveform(PSD,time_indices=2128,seconds_per_index=0.5*10**(-6)):
        if (len(PSD)==0): return [0],[0] #avoids processing empty channels

        import numpy as np
        import scipy

        T = time_indices * seconds_per_index #time length of the waveform
        alpha = 2*T/time_indices**2 #our choice of normalization for the PSD

        theta = np.random.rand(len(PSD))*2*np.pi #Generate random phases for the FFT
        theta[0]=0 #the first entry needs to be real for irfft

        #When taking the fourier transform, we divide out by the normalization first
        F = np.sqrt(PSD/alpha)*(np.cos(theta) + 1j*np.sin(theta)) #Our fourier transform
        f = np.fft.irfft(F,time_indices) #inverse real fourier transform
        time = np.arange(time_indices)*seconds_per_index #time axis

        l = 5*10**(-2) #tunable overlap rate of the signals, in Hz
        #l cares about time ticks, not absolute time.

        #now we generate the time delays according to Poisson statistics
        time_delays = [0]
        sum = 0
        while sum < time_indices:
            index = 0
            R = np.random.rand()
            sum += int(-np.log(1-R)/l)
            time_delays.append(sum)
        
        #Amplitudes for each term we add of f to create the noise.
        Amplitude_Variance = 1/(l*time_indices)
        amplitudes = np.random.normal(0, np.sqrt(Amplitude_Variance), len(time_delays))
        
        #create the waveform
        waveform = np.zeros(time_indices)
        for k in range(len(time_delays)):
            waveform += amplitudes[k]*np.roll(f,time_delays[k])

        return waveform, f, time #f is included in order to validate method
    
    #npz_file contains existing noise model PSD (see above method)
    #returns an ICEBERG event's noise waveforms and PSDs of those waveforms.
    #New_PSD[iTPC][iPlane][iChannel][frequency]
    #New_Waveform[iTPC][iPlane][iChannel][time_tick]
    def simulateEvent(npz_file,numtpcs = 2, numplanes = 3, maxwires = 240, time_indices = 2128, SampleSpacing = 0.5*10**(-6)):
        import numpy as np
        from myprocessor import myprocessor as myp
        import matplotlib.pyplot as plt

        PSDs = simulate_noise.PSDfromNPZ(npz_file)
        PSDlength = len(PSDs[0][0][0]) #should be rectangular and = 1065

        #Convenient naming convention for me when normalizing
        N = time_indices
        T = SampleSpacing * N

        New_Waveform = np.zeros((numtpcs,numplanes,maxwires,time_indices),dtype=float)
        New_PSD = np.zeros((numtpcs,numplanes,maxwires,PSDlength),dtype=float)

        for iTPC in range(numtpcs):
            for iPlane in range(numplanes):
                #I would like to figure out how to make this step faster!!!
                for iChannel in range(maxwires):
                    if(len(PSDs[iTPC][iPlane][iChannel]) != 0):
                        New_Waveform[iTPC][iPlane][iChannel], f, time = simulate_noise.getNoiseWaveform(PSDs[iTPC][iPlane][iChannel])
                        
                        ADCASD, ADCPSD, ADCFFT = myp.get_single_ASDPSDFFT(New_Waveform[iTPC][iPlane][iChannel], SampleSpacing)
                        
                        New_PSD[iTPC][iPlane][iChannel] = ADCPSD

        return New_Waveform, New_PSD
    
    def simulateMultipleEvents(npz_file, n, savefile, numtpcs = 2, numplanes = 3, maxwires = 240, time_indices = 2128, SampleSpacing = 0.5*10**(-6)):
        import numpy as np
        Waveform_Array = np.zeros(n)
        PSD_Array = np.zeros(n)

        for i in range(n):
            Waveform_Array[i], PSD_Array[i] = simulateEvent(npz_file,numtpcs = 2, numplanes = 3, maxwires = 240, time_indices = 2128, SampleSpacing = 0.5*10**(-6))
        
        np.savez(savefile,Waveform_Array,PSD_Array)
        return Waveform_Array,PSD_Array