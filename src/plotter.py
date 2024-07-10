class plotter:

    #Contains Methods:
    #   convert_to_sorted(MegaArray)
    #   convert_to_mega(SortedArray)
    #   HistAvgPSDbyPlane(PSD,outputFile1 = 'histPSDs.png',outputFile2 = 'PSDs.png')
    #   plotrCorrSorted(MegaCorr,outputFile = 'rCorrSorted.png') ###BROKEN###
    #   plotrCorr(MegaCorr,plotTitle='Noise Waveform Correlations',outputFile = 'rCorr.png')
    #   plotmegacorrbyband(corrarray)

    numtpcs = 2
    numplanes = 3
    maxwires = 240
    numchannels = 200*4 + 240*2 #Number of wires in the TPC
    minwvfm = 2128
    SampleSpacing = 0.5e-6 #Seconds per time tick
    numfreqbins = 10
    
    #convert_index_to_sorted
    #Input a global channel number and get an array with the TPC, Plane, and Channel number
    #Output: TPC, Plane, Channel
    def convert_index_to_sorted(index):
        if(index < 200):
            return 0,0,index
        elif(index < 400):
            return 1,0,index-200
        elif(index < 600):
            return 0,1,index-400
        elif(index < 800):
            return 1,1,index-600
        elif(index < 1040):
            return 0,2,index-800
        elif(index < 1280):
            return 1,2,index-1040
        else:
            return -1,-1,-1


    #convert_to_sorted
    #MegaArray[GlobalChannel][OtherStuff...] is a numpy array with shape ()
    #SortedArray[tpc][plane][channel][OtherStuff...]
    def convert_to_sorted(MegaArray):
        import numpy as np
        RestShape = MegaArray.shape[1:]
        SortedArray = np.zeros(np.concatenate(([plotter.numtpcs,plotter.numplanes,plotter.maxwires],RestShape)))
        
        #Code for plane numbers in sorted array
        #0 = u
        #1 = v
        #2 = z (col)
        
        SortedArray[0][0][0:200] = MegaArray[0:200]     #TPC 0 Plane u
        SortedArray[1][0][0:200] = MegaArray[200:400]   #TPC 1 Plane u
        SortedArray[0][1][0:200] = MegaArray[400:600]   #TPC 0 Plane v
        SortedArray[1][1][0:200] = MegaArray[600:800]   #TPC 1 Plane v
        SortedArray[0][2]        = MegaArray[800:1040]  #TPC 0 Plane z
        SortedArray[1][2]        = MegaArray[1040:1280] #TPC 1 Plane z

        #This complicated piece tells us to concatenate MegaArray with an array of size 40
        #on the first axis and the rest of the shape the same as the input array. 
        #SortedArray[1][0] = np.concatenate((MegaArray[200:400],np.zeros(np.concatenate(([40],RestShape)))))

        return SortedArray
    
    #convert_to_mega
    #MegaArray[GlobalChannel][OtherStuff...] is a numpy array with shape ()
    #SortedArray[tpc][plane][channel][OtherStuff...]
    def convert_to_mega(SortedArray):
        import numpy as np
        RestShape = SortedArray.shape[3:]
        MegaArray = np.zeros(np.concatenate(([plotter.numchannels],RestShape)))

        #Code for plane numbers in sorted array
        #0 = u
        #1 = v
        #2 = z (col)
        
        MegaArray[0:200]     =  SortedArray[0][0][0:200] #TPC 0 Plane u
        MegaArray[200:400]   =  SortedArray[1][0][0:200] #TPC 1 Plane u
        MegaArray[400:600]   =  SortedArray[0][1][0:200] #TPC 0 Plane v
        MegaArray[600:800]   =  SortedArray[1][1][0:200] #TPC 1 Plane v
        MegaArray[800:1040]  =  SortedArray[0][2]        #TPC 0 Plane z
        MegaArray[1040:1280] =  SortedArray[1][2]        #TPC 1 Plane z

        return MegaArray

    #HistAvgPSDbyPlane
    #
    #Plotting function that takes in a PSD[GlobalChan][Freq] and plots it by plane
    #
    #INPUTS: PSD[GlobalChannel][Freq]
    #
    def HistAvgPSDbyPlane(PSD,outputFile1 = 'histPSDs.png',outputFile2 = 'PSDs.png'):
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        from mpl_toolkits.axes_grid1 import make_axes_locatable   
                
        #For plot of averaged PSDs
        fig,ax = plt.subplots(plotter.numtpcs,plotter.numplanes,num=1)
        fig.set_figheight(5*plotter.numtpcs)
        fig.set_figwidth(5*plotter.numplanes)
        
        colorvalues = mpl.cm.viridis(range(plotter.maxwires))
        divider = make_axes_locatable(plt.gca())
        ax_cb = divider.new_horizontal(size="5%", pad=0.05)    
        cb1 = mpl.colorbar.ColorbarBase(ax_cb, cmap=mpl.cm.viridis, orientation='vertical',boundaries = np.arange(plotter.maxwires+1),values = colorvalues)
        plt.gcf().add_axes(ax_cb)
        
        #For plot of freq v. channel # v. color
        fig2,ax2 = plt.subplots(plotter.numtpcs,plotter.numplanes,num=2)
        fig2.set_figheight(plotter.numtpcs*5)
        fig2.set_figwidth(plotter.numplanes*5)

        planenames = ["u", "v", "z"]
        
        Sorted_PSD = plotter.convert_to_sorted(PSD)
        
        for tpc in range(plotter.numtpcs):
            for plane in range(plotter.numplanes):
                freq = np.fft.rfftfreq(plotter.minwvfm,plotter.SampleSpacing)

                #The 2d histogram takes only 1d arrays
                #so we have to flatten out our AvgASD array into one long array
                #also we have to do the same for the frequencies
                longPSD = np.empty(0,dtype=float)
                longFreq = np.empty(0,dtype=float)
                for channel in range(len(Sorted_PSD[tpc][plane])):
                    longPSD = np.concatenate((longPSD, abs(Sorted_PSD[tpc][plane][channel])))
                    longFreq = np.concatenate((longFreq, freq))
                
                #For 2D plot
                cmap = mpl.cm.get_cmap('viridis')
                cmap.set_under('white')
                ax2[tpc][plane].pcolormesh(freq,range(plotter.maxwires),Sorted_PSD[tpc][plane],cmap = cmap,shading='gouraud',vmin = -1, vmax = 1) #default -1.3,3.3
                fig2.colorbar(ax2[tpc][plane].pcolormesh(freq,range(plotter.maxwires),Sorted_PSD[tpc][plane],cmap = cmap,shading='gouraud',vmin = -1, vmax = 1)) #default 1.3,3.3
                if(plane!=2):
                    ax2[tpc][plane].set_ylim(0,200)
                
                #For PSD Plot
                for nChannel in range(len(Sorted_PSD[tpc][plane])):
                    ax[tpc][plane].scatter(freq,Sorted_PSD[tpc][plane][nChannel],s=0.05,color = colorvalues[nChannel])
                    
                ax[tpc][plane].set_yscale("linear") #log
                ax[tpc][plane].set_xlabel("Freq [1e6 Hz]")
                ax[tpc][plane].set_ylabel("Averaged FFT [1/Hz^0.5]")
                ax[tpc][plane].set_title("Averaged Channel FFT for TPC: " + str(tpc) + " Wire Plane: " + planenames[plane])
                
                ax2[tpc][plane].set_xscale("linear")
                ax2[tpc][plane].set_xlabel("Freq [Hz]")
                ax2[tpc][plane].set_ylabel("Channel Number")
                ax2[tpc][plane].set_title("Averaged Channel FFT for TPC: " + str(tpc) + " Wire Plane: " + planenames[plane])
        plt.show()
        fig.savefig(outputFile1, dpi = fig.dpi)
        fig2.savefig(outputFile2, dpi = fig2.dpi)

    #DO NOT USE THIS METHOD - THE CONVERSION FROM MEGACORR TO SORTEDCORR IS NOT CORRECT HERE
    #plotrCorrSorted
    #
    #Function takes in a correlation matrix and plots it
    #
    #INPUT: rMegaCorr[GlobalChannel][GlobalChannel] is the correlation matrix for wires in time
    #           Averaged across all of the events used to create the model
    #
    #
    def plotrCorrSorted(MegaCorr,outputFile = 'rCorrSorted.png'):
        import matplotlib.pyplot as plt
        import numpy as np

        fig,ax = plt.subplots(plotter.numtpcs,plotter.numplanes,num=1)
        fig.set_figheight(10*plotter.numtpcs)
        fig.set_figwidth(10*plotter.numplanes)
        fig.suptitle("Noise Waveform Correlations", y=0.93, size=15)
        fig.set_size_inches(15, 10)
        planenames = ["u", "v", "z"]

        SortedCorr = plotter.convert_to_sorted(MegaCorr)

        for TPCnum in range(plotter.numtpcs):
            for PlaneNum in range(plotter.numplanes):
                if(PlaneNum != 2):
                    ax[TPCnum][PlaneNum].set_ylim(0,200)
                    ax[TPCnum][PlaneNum].set_xlim(0,200)
                rCoeff = SortedCorr[TPCnum][PlaneNum]
                ax[TPCnum][PlaneNum].set_xlabel("Channels 0-240")
                ax[TPCnum][PlaneNum].set_ylabel("Channels 0-240")
                ax[TPCnum][PlaneNum].set_title("TPC #" + str(TPCnum) +" Plane " + planenames[PlaneNum])
                fig.colorbar(ax[TPCnum][PlaneNum].pcolormesh(rCoeff),vmin=-1,vmax=1)
                ax[TPCnum][PlaneNum].pcolormesh(np.arange(240),np.arange(240),rCoeff,shading='gouraud',vmin = -1,vmax = 1)
        plt.show()
        fig.savefig(outputFile, dpi = fig.dpi)
    
    #plotrCorr
    #
    #Plots the correlations between the IECBERG wires
    #Saves an image of the plot
    #
    #INPUTS: MegaCorr[GlobalChannel][GlobalChannel] 
    #    
    def plotrCorr(MegaCorr,BothLive,plotTitle='Noise Waveform Correlations',outputFile = 'rCorr.png'):
        import matplotlib.pyplot as plt
        import numpy as np

        #Note: Where BothLive == 0, change rCorr value to nan. This visually makes more sense.
        MegaCorr[BothLive == 0] = np.NaN
            
        lines = [200,400,600,800,1040,1280]
        linenames = ['200','400','600','800','1040','1280']
        ticks = np.concatenate((40*(np.arange(19)+1),48*np.arange(10)+800))
        planeindexes = [100,300,500,700,920,1160]
        planenames = ['u0','u1','v0','v1','z0','z1']

        fig = plt.figure(figsize=(15,12))
        plt.title(plotTitle,fontsize=18)
        plt.colorbar(plt.pcolormesh(np.arange(1280),np.arange(1280),MegaCorr,shading='gouraud',vmin=-1,vmax=1))
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
        plt.pcolormesh(np.arange(1280),np.arange(1280),MegaCorr,shading='gouraud',vmin=-1,vmax=1)
        plt.show()
        fig.savefig(outputFile, dpi = fig.dpi)

    #plotmegacorrbyband
    #
    #Plots the correlations across channels of the waveforms for different frequencies
    #
    #INPUT: corrarray[freqindex][GlobalChannel][GlobalChannel]
    #
    def plotmegacorrbyband(corrarray,BothLive):
        import matplotlib.pyplot as plt
        import numpy as np

        maxfreq=1/(2*plotter.SampleSpacing)
        freqrange = np.linspace(0,maxfreq,num = plotter.numfreqbins+1)

        for freqindex in range(len(corrarray)):
            PlotTitle = 'Noise Waveform Correlations for the ' + str(freqindex)+ '[1e6 Hz] to ' + str(freqindex+1)+'[1e6Hz] Band'
            outputFile = 'rCorr' +  str(freqindex)+ 'e6to' + str(freqindex+1)+'e6.png'
            plotter.plotrCorr(corrarray[freqindex],BothLive,PlotTitle,outputFile)

    #plotKurtosis
    #
    #Plots the distributions of kurtosis before and after masking to investigate the efficacy of removing large outlier signals
    #
    #INPUT: Kurtosis values sorted by Kurtosis[Event][Channel][Before/After]
    def plotKurtosis(Kurtosis):
        import numpy as np
        import matplotlib.pyplot as plt

        #All Channels Together
        LongKurt = np.reshape(Kurtosis,(-1,2))

        plt.figure()
        plt.hist(LongKurt[:][0],color = "blue") #Before
        plt.hist(LongKurt[:][1],color = "green") #After
        plt.xlabel("Kurtosis Value")
        plt.ylabel("Bin Counts")

        return