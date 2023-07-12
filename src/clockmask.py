#mask contains: recursive_stats(pulsed_data)[do not use], masksignals(pulsed_data)

class clockmask: 
    #this is the recursive method to return the noise-free mean and std of a waveform
    def recursive_stats(data_arr):
        import numpy as np
        #we start by splitting our data in half, first half and second half
        data_len = len(data_arr)
        first_half = data_arr[0:int(data_len/2)+int(data_len/40)]
        second_half = data_arr[int(data_len/2)-int(data_len/40):data_len]
        
        #un-commenting my plot and print statements gives some demonstration of how this method works.

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
                return clockmask.recursive_stats(first_half)
            else:
                return clockmask.recursive_stats(second_half)
    def masksignals(arrayADC):
        import numpy as np
        #checking what the data looks like before
        stds = []
        for i in range(len(arrayADC)):
            for j in range(len(arrayADC[i])):
                stds.append(np.std(arrayADC[i][j]))
        #print("pre-mask max std: "+str(np.max(stds)))
        #print("pre-mask mean std: "+ str(np.mean(stds)))


        for nEvent in range(len(arrayADC)):
            #I just use this to see how far along the code is, since it usually takes a few minutes
            #if(nEvent%2==0):
                #print("starting event number: " + str(nEvent))
            for nChannel in range(len(arrayADC[nEvent])):   
                #print(str(nChannel)) 
                #Lets mask out the peaks and replace them with a delta function that can later be taken out of the fourier transform
                #I have a kinda complicated way of doing this but its meant to be rigorous enough to remove all peaks
                
                #first we use the recursive method above to find the mean and STD of a section of the noise with no peaks
                noiseSTD, noiseMean = clockmask.recursive_stats(arrayADC[nEvent][nChannel])  
                
                #keeping track of indexes we masked out
                switchedArr = np.zeros((len(arrayADC[nEvent][nChannel])),dtype=bool)
                
                #classic threshold mask. anything varied by >6*noiseSTD get replaced with the mean
                for datapt in range(0,len(arrayADC[nEvent][nChannel])):
                    if((arrayADC[nEvent][nChannel][datapt] > noiseMean+6*noiseSTD)|(arrayADC[nEvent][nChannel][datapt] < noiseMean-6*noiseSTD)):
                        arrayADC[nEvent][nChannel][datapt] = noiseMean
                        switchedArr[datapt] = True
                #then we go back through and wherever we masked out a peak, we trace backwards and keep masking data points until we drop within 2 standard deviations of the mean.
                #we do this again for the forward direction
                for datapt in range(1,len(arrayADC[nEvent][nChannel])-1):
                    if(switchedArr[datapt]):
                        if(not(switchedArr[datapt-1])):
                            j=1
                            while((arrayADC[nEvent][nChannel][datapt-j]>noiseMean+2*noiseSTD)|(arrayADC[nEvent][nChannel][datapt-j]<noiseMean-2*noiseSTD)):
                                arrayADC[nEvent][nChannel][datapt-j] = noiseMean
                                switchedArr[datapt-j] = True
                                j+=1
                                if(datapt-j == 0):
                                    break
                        if(not(switchedArr[datapt+1])):
                            j=1
                            while((arrayADC[nEvent][nChannel][datapt+j]>noiseMean+2*noiseSTD)|(arrayADC[nEvent][nChannel][datapt+j]<noiseMean-2*noiseSTD)):
                                arrayADC[nEvent][nChannel][datapt+j] = noiseMean
                                switchedArr[datapt+j] = True
                                j+=1
                                if(datapt+j == len(arrayADC[nEvent][nChannel])):
                                    break
                #then as a final check, we go through our data again and say, for every data point, if both the datapoint to the right and to the left have been masked, mask that
                #data point out too. this is just to make sure that section in between bimodal peaks doesn't slip through the cracks.
                for datapt in range(1,len(arrayADC[nEvent][nChannel])-1):
                    if(switchedArr[datapt+1]&switchedArr[datapt-1]):
                        switchedArr[datapt] = True
                        arrayADC[nEvent][nChannel][datapt] = noiseMean
                        
        #comparing these numbers to the ones above in order to check the clock masking performance
        stds = []
        for i in range(len(arrayADC)):
            for j in range(len(arrayADC[i])):
                stds.append(np.std(arrayADC[i][j]))
        print("post-mask max std: "+str(np.max(stds)))
        print("post-mask mean std: " + str(np.mean(stds)))
        return arrayADC