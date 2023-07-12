#twodhist contians: hist_maxes(FFT/ASD, freq, binnum)[do not use], histAvgASD(arrayADC, binnum), 
#and histAvgFFT(arrayADC,binnum), and HistASDbyPlane(dataarr, binnum)

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
        
    def HistASDbyPlane(data_arr):
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.colors as mpl
        binnum = 300
        fig,ax = plt.subplots(2,3)
        fig.set_figheight(10)
        fig.set_figwidth(15)
        planenames = ["u", "v", "z"]
        midlines = []
        AvgASD = data_arr
        for tpc in range(len(data_arr)):
            for plane in range(len(data_arr[tpc])):
                SampleSpacing = 0.5e-6 #0.5 microseconds per tick
                N = 2128
                freq = np.fft.rfftfreq(N,SampleSpacing)
                #The 2d histogram takes only 1d arrays
                #so we have to flatten out our AvgASD array into one long array
                #also we have to do the same for the frequencies
                longASD = np.empty(0,dtype=float)
                longFreq = np.empty(0,dtype=float)
                for channel in range(len(AvgASD[tpc][plane])):
                    longASD = np.concatenate((longASD, abs(AvgASD[tpc][plane][channel])))
                    longFreq = np.concatenate((longFreq, freq))
                    
                h = ax[tpc][plane].hist2d(longFreq/1e6, longASD, bins=(binnum,binnum), \
                            norm=mpl.LogNorm(vmax=1.1e3), cmap=plt.cm.jet)
                #plot the midline of the heatmap
                midline = twodhist.hist_maxes(longASD, longFreq, binnum)
                midlines.append(midline)
                ax[tpc][plane].plot(midline[0],midline[1],'w',linewidth=0.5)
                ax[tpc][plane].set_ylim(0,0.10)
                ax[tpc][plane].set_xlabel("Freq [1e6 Hz]")
                ax[tpc][plane].set_ylabel("ADC strain noise [1/Hz^0.5]")
                fig.colorbar(h[3], ax=ax[tpc][plane])
                ax[tpc][plane].set_title("Averaged Channel ASD for TPC: " + str(tpc) + " Wire Cell: " + planenames[plane])
        plt.show()
        return midlines
    def getASDs(data_arr, numtpcs, numplanes, maxwires, ASDlength, SampleSpacing):
        #THIS CODE takes in an array of sorted data, shape 2x3x38x240x2128, and returns a sorted array of each waveform's ASD, in the shape 2x3x240x1065. 
        # It also returns an array of the number of entries in each channel, used later for averaging across files
        import numpy as np
        returnASD = np.zeros((numtpcs,numplanes,maxwires,ASDlength),dtype=float)
        returnPSD = np.zeros((numtpcs,numplanes,maxwires,ASDlength),dtype=float)
        returnNoSkips = np.zeros((numtpcs,numplanes,maxwires),dtype=int)
        for tpc in range(len(data_arr)):
            for plane in range(len(data_arr[tpc])):
                arrayADC = data_arr[tpc][plane]
                ADCFFT = np.fft.rfft(arrayADC[0][0])
                SumASD = np.zeros((len(arrayADC[0]),len(ADCFFT)),dtype=float)
                SumPSD = np.zeros((len(arrayADC[0]),len(ADCFFT)),dtype=float)
                N = len(arrayADC[0][0])
                T = SampleSpacing*N #Period
                #AVERAGING EACH CHANNEL ACROSS ALL EVENTS
                #this code takes the ASDs of each channel of each event, sums, and averages them.
                noskiparr = np.zeros(len(arrayADC[0]))
                for nEvent in range(len(arrayADC)):
                    for nChannel in range(len(arrayADC[nEvent])):
                        if(abs(np.sum(arrayADC[nEvent][nChannel])) > 7000000): #this statement ensures that we only take the FFT of data that exists
                            ADCFFT = np.fft.rfft(arrayADC[nEvent][nChannel])
                            #we cut the first bin since Angela says it's the baseline DC offset
                            ADCFFT[0]=0
                            #Do we want the PSD (Power Spectral Density)?
                            ADCPSD = 2*T/N**2 * np.abs(ADCFFT)**2 #Power Spectral Density
                            ADCASD = np.sqrt(ADCPSD) #Amplitude Spectral Density
                            SumASD[nChannel] += abs(ADCASD)
                            SumPSD[nChannel] += abs(ADCPSD)
                            noskiparr[nChannel] += 1
                AvgASD = SumASD
                AvgPSD = SumPSD
                returnASD[tpc][plane] = abs(AvgASD)
                returnPSD[tpc][plane] = abs(AvgPSD)
                returnNoSkips[tpc][plane] = noskiparr
        return returnASD, returnNoSkips, returnPSD
    def get_single_ASDPSD(waveform, SampleSpacing):
        import numpy as np
        ADCFFT = np.fft.rfft(waveform)
        N = len(waveform)
        T = SampleSpacing*N
        ADCFFT[0] = 0
        ADCPSD = 2*T/N**2 * np.abs(ADCFFT)**2
        ADCASD = np.sqrt(ADCPSD)
        return ADCASD, ADCPSD
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
                ax[tpc][plane].set_title("Averaged Channel ASD for TPC: " + str(tpc) + " Pixel Plane: " + planenames[plane])
                
                ax2[tpc][plane].set_xlabel("Freq Hz")
                ax2[tpc][plane].set_ylabel("Channel Number [1/Hz^0.5]")
                ax2[tpc][plane].set_title("Averaged Channel ASD for TPC: " + str(tpc) + " Pixel Plane: " + planenames[plane])
        plt.show() 