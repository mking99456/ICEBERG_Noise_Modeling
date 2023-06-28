#twodhist contians: hist_maxes(FFT/ASD, freq, binnum)[do not use], histAvgASD(arrayADC, binnum), 
#and histAvgFFT(arrayADC,binnum)

class twodhist:
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
    def histAvgASD(arrayADC, binnum):
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.colors as mpl

        ADCFFT = np.fft.rfft(arrayADC[0][0])
        SumASD = np.zeros((len(arrayADC[0]),len(ADCFFT)),dtype=np.complex128)
        SampleSpacing = 0.5e-6 #0.5 microseconds per tick
        N = len(arrayADC[0][0])
        T = SampleSpacing*N #Period
        freq = np.fft.rfftfreq(N,SampleSpacing)
        #AVERAGING EACH CHANNEL ACROSS ALL EVENTS
        #this code takes the ASDs of each channel of each event, sums, and averages them.
        for nEvent in range(len(arrayADC)):
            for nChannel in range(len(arrayADC[nEvent])):
                ADCFFT = np.fft.rfft(arrayADC[nEvent][nChannel])
                #we cut the first bin since Angela says it's the baseline DC offset
                ADCFFT[0]=0
                #Do we want the PSD (Power Spectral Density)?
                ADCPSD = 2*T/N**2 * np.abs(ADCFFT)**2 #Power Spectral Density
                ADCASD = np.sqrt(ADCPSD) #Amplitude Spectral Density
                SumASD[nChannel] += abs(ADCASD)
        AvgASD = SumASD/len(arrayADC)
        #The 2d histogram takes only 1d arrays
        #so we have to flatten out our AvgASD array into one long array
        #also we have to do the same for the frequencies
        longASD = np.empty(0,dtype=float)
        longFreq = np.empty(0,dtype=float)
        for channel in range(len(AvgASD)):
            longASD = np.concatenate((longASD, abs(AvgASD[channel])))
            longFreq = np.concatenate((longFreq, freq))
        #plt.figure()
        fig,ax = plt.subplots()
        #plot 2d histogram with heatmap and number of bins
        h = ax.hist2d(longFreq/1e6, longASD, bins=(binnum,binnum), \
                            norm=mpl.LogNorm(vmax=1.1e3), cmap=plt.cm.jet)
        #plot the midline of the heatmap
        midline = twodhist.hist_maxes(longASD, longFreq, binnum)
        plt.plot(midline[0], midline[1], 'w',linewidth=0.5)
        plt.yscale("linear")
        plt.xlabel("Freq [1e6 Hz]")
        plt.ylabel("ADC strain noise [1/Hz^0.5]")
        plt.ylim(0,0.10)
        plt.title("Averaged Channel ASD Heatmap")
        fig.colorbar(h[3],ax=ax)
        plt.show()
    
    def histAvgFFT(arrayADC, binnum):
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.colors as mpl
        ADCFFT = np.fft.rfft(arrayADC[0][0])
        SumFFT = np.zeros((len(arrayADC[0]),len(ADCFFT)),dtype=np.complex128)
        SampleSpacing = 0.5e-6 #0.5 microseconds per tick
        N = len(arrayADC[0][0])
        freq = np.fft.rfftfreq(N,SampleSpacing)
        #AVERAGING EACH CHANNEL ACROSS ALL EVENTS
        #this code takes the ASDs of each channel of each event, sums, and averages them.
        for nEvent in range(len(arrayADC)):
            for nChannel in range(len(arrayADC[nEvent])):
                ADCFFT = np.fft.rfft(arrayADC[nEvent][nChannel])
                #we cut the first bin since Angela says it's the baseline DC offset
                ADCFFT[0]=0
                SumFFT[nChannel] += abs(ADCFFT)
        AvgFFT = SumFFT/len(arrayADC)
        #The 2d histogram takes only 1d arrays
        #so we have to flatten out our AvgASD array into one long array
        #also we have to do the same for the frequencies
        longFFT = np.empty(0,dtype=float)
        longFreq = np.empty(0,dtype=float)
        for channel in range(len(AvgFFT)):
            longFFT = np.concatenate((longFFT, abs(AvgFFT[channel])))
            longFreq = np.concatenate((longFreq, freq))
        #plt.figure()
        fig,ax = plt.subplots()
        #plot 2d histogram with heatmap and number of bins
        h = ax.hist2d(longFreq/1e6, longFFT, bins=(binnum,binnum), \
                            norm=mpl.LogNorm(vmax=1.1e3), cmap=plt.cm.jet)
        #plot the midline of the heatmap
        midline = twodhist.hist_maxes(longFFT, longFreq, binnum)
        plt.plot(midline[0], midline[1], 'w',linewidth=0.5)
        plt.yscale("linear")
        plt.xlabel("Freq [1e6 Hz]")
        plt.ylabel("ADC FFT")
        plt.ylim(0,4000)
        plt.title("Averaged Channel FFT Heatmap")
        fig.colorbar(h[3],ax=ax)
        plt.show()   