#processfile contains: getADC("filepath/filename.root")
#this will likely only work for 9055 or other similar files. I don't know why it doesn't work for 9613

#here are the appropriate keys for 9613. switch them out in the code:
#fADC = events['raw::RawDigits_tpcrawdecoder_daq_RunIcebergRawDecoder.obj.fADC']
#fSamples = events['raw::RawDigits_tpcrawdecoder_daq_RunIcebergRawDecoder.obj.fSamples']
#fChannel = events['raw::RawDigits_tpcrawdecoder_daq_RunIcebergRawDecoder.obj.fChannel']

class fileprocess:
    def getADC(infile_name):
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

        npADC = np.array(awk.to_list(ADCarr), dtype="O")
        npSamples = np.array(awk.to_list(Samplesarr), dtype="O")
        npChannel = np.array(awk.to_list(Channelarr), dtype="O")
        SampleSpacing = 0.5e-6 #0.5 microseconds per tick

        #There are 141 nEvents
        #For some nEvents we have 1024 channels and 2128 data points for each
        #Other nEvents are EMPTY. I remove all the empty ones
        removecount = 0
        for i in range(len(npADC)):
            if(len(npADC[i-removecount])==0):
                npADC = np.delete(npADC, i-removecount)
                npSamples = np.delete(npSamples, i-removecount)
                npChannel = np.delete(npChannel, i-removecount)
                removecount += 1
            if(len(npADC[i-removecount][0]) != len(npADC[0][0])):
                npADC = np.delete(npADC, i-removecount)
                npSamples = np.delete(npSamples, i-removecount)
                npChannel = np.delete(npChannel, i-removecount)
                removecount += 1
        print("Number of events with data: "+str(len(npADC)))
        arrayADC = np.zeros((len(npADC),len(npADC[0]),len(npADC[0][0])), dtype=int)
        arraySamples = np.zeros((len(npSamples),len(npSamples[0])),dtype=int)
        arrayChannel = np.zeros((len(npChannel),len(npChannel[0])),dtype=int)
        for eventnum in range(len(npADC)):
            arraySamples[eventnum] = npSamples[eventnum]
            arrayChannel[eventnum] = npChannel[eventnum]
            arrayADC[eventnum] = npADC[eventnum]
        print("Num Channels: " + str(len(arrayADC[0])) + " Waveform Length: " + str(len(arrayADC[0][0])))
        return arrayADC, arraySamples, arrayChannel
