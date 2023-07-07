#processfile contains: getADC("filepath/filename.root")

#Keys for 9613:
#ADC = events['raw::RawDigits_tpcrawdecoder_daq_RunIcebergRawDecoder.obj.fADC']
#fSamples = events['raw::RawDigits_tpcrawdecoder_daq_RunIcebergRawDecoder.obj.fSamples']
#fChannel = events['raw::RawDigits_tpcrawdecoder_daq_RunIcebergRawDecoder.obj.fChannel']


class fileprocess:
    def getADC(infile_name,minwvfm,numchannels):
        import uproot
        import numpy as np
        import awkward as awk
        infile = uproot.open(infile_name)
        events = infile['Events']
        
        fADC = events['raw::RawDigits_tpcrawdecoder_daq_RunIcebergRawDecoder./raw::RawDigits_tpcrawdecoder_daq_RunIcebergRawDecoder.obj/raw::RawDigits_tpcrawdecoder_daq_RunIcebergRawDecoder.obj.fADC']
        fSamples = events['raw::RawDigits_tpcrawdecoder_daq_RunIcebergRawDecoder./raw::RawDigits_tpcrawdecoder_daq_RunIcebergRawDecoder.obj/raw::RawDigits_tpcrawdecoder_daq_RunIcebergRawDecoder.obj.fSamples']
        fChannel = events['raw::RawDigits_tpcrawdecoder_daq_RunIcebergRawDecoder./raw::RawDigits_tpcrawdecoder_daq_RunIcebergRawDecoder.obj/raw::RawDigits_tpcrawdecoder_daq_RunIcebergRawDecoder.obj.fChannel']

        ADCarr = fADC.array()
        Samplesarr = fSamples.array()
        Channelarr = fChannel.array()

        #We end up with lots of empty events, so when we convert from an awk array to a np array we only take the events with data
        
        #What events have data?
        counter = []
        for index in range(len(ADCarr)):
            if(awk.count(ADCarr[index]) >= numchannels*(minwvfm-1)):
                counter.append(index)

        #Put those events in a np array
        npADC = np.zeros((len(counter),len(ADCarr[counter[0]]),minwvfm), dtype=int)
        npSamples = np.zeros((len(counter),len(Samplesarr[counter[0]])), dtype=int)
        npChannel = np.zeros((len(counter),len(Channelarr[counter[0]])), dtype=int)
        for index in range(len(counter)):
            for nChannel in range(len(npADC[index])):
                npADC[index][nChannel] = ADCarr[counter[index]][nChannel][0:minwvfm] #standardize waveform lengths
                npSamples[index][nChannel] = Samplesarr[counter[index]][nChannel]
                npChannel[index][nChannel] = Channelarr[counter[index]][nChannel]
        print("Number of events with data: "+str(len(npADC))+" Number of Channels: " + str(len(npADC[0])) + " Waveform Length: " + str(len(npADC[0][0])))
        return npADC, npSamples, npChannel