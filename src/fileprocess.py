#processfile contains: getADC("filepath/filename.root")

#Keys for 9613:
#ADC = events['raw::RawDigits_tpcrawdecoder_daq_RunIcebergRawDecoder.obj.fADC']
#fSamples = events['raw::RawDigits_tpcrawdecoder_daq_RunIcebergRawDecoder.obj.fSamples']
#fChannel = events['raw::RawDigits_tpcrawdecoder_daq_RunIcebergRawDecoder.obj.fChannel']


class fileprocess:
    def getADC(infile_name):
        import uproot
        import numpy as np
        import awkward as awk
        from scipy import stats
        infile = uproot.open(infile_name)
        events = infile['Events']
        
        fADC = events['raw::RawDigits_tpcrawdecoder_daq_RunIcebergRawDecoder./raw::RawDigits_tpcrawdecoder_daq_RunIcebergRawDecoder.obj/raw::RawDigits_tpcrawdecoder_daq_RunIcebergRawDecoder.obj.fADC']
        fSamples = events['raw::RawDigits_tpcrawdecoder_daq_RunIcebergRawDecoder./raw::RawDigits_tpcrawdecoder_daq_RunIcebergRawDecoder.obj/raw::RawDigits_tpcrawdecoder_daq_RunIcebergRawDecoder.obj.fSamples']
        fChannel = events['raw::RawDigits_tpcrawdecoder_daq_RunIcebergRawDecoder./raw::RawDigits_tpcrawdecoder_daq_RunIcebergRawDecoder.obj/raw::RawDigits_tpcrawdecoder_daq_RunIcebergRawDecoder.obj.fChannel']


        ADCarr = fADC.array()
        Samplesarr = fSamples.array()
        Channelarr = fChannel.array()

        counter = []
        for index in range(len(ADCarr)):
            if(awk.count(ADCarr[index]) >= 2000000):
                counter.append(index)
        SampleSpacing = 0.5e-6 #0.5 microseconds per tick

        npADC = np.zeros((len(counter),len(ADCarr[counter[0]]),len(ADCarr[counter[0]][0])), dtype=int)
        npSamples = np.zeros((len(counter),len(Samplesarr[counter[0]])), dtype=int)
        npChannel = np.zeros((len(counter),len(Channelarr[counter[0]])), dtype=int)
        for index in range(len(counter)):
            npADC[index] = ADCarr[counter[index]]
            npSamples[index] = Samplesarr[counter[index]]
            npChannel[index] = Channelarr[counter[index]]
        print("Number of events with data: "+str(len(npADC))+" Number of Channels: " + str(len(npADC[0])) + " Waveform Length: " + str(len(npADC[0][0])))
        return npADC, npSamples, npChannel