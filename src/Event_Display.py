#############################################################
### Event Displays for the ICEBERG TPC			  ###
### By Matt King mking9@uchicago.edu, 7/3/2023 		  ###
### University of Chicago				  ###
###							  ###
### This program allows us to visualize the 6 ICEBERG	  ###
### wire planes separated out, giving us better control   ###
### and understanding than the LArSOFT event display.	  ###
###							  ###
### It takes as an argument the file to be read in.	  ###
###							  ###
### When operating it, use asdf keys to navigate between  ###
### different events (after closing out the current image)###
###							  ###
### a = previous event					  ###
### s = current event					  ###
### d = next event					  ###
### f = type in your desired event index		  ###
###							  ###
### Currently having trouble getting the plot axes and	  ###
### Colorbar to work properly. In the meantime, this does ###
### successfully show the desired information.		  ###
###							  ###
### Also currently only displays good channels. Next work ###
### also includes filling in blanks for dead channels.	  ###
###							  ###
#############################################################

#import standard math and plotting things
import numpy as np
import matplotlib.pylab as plt 
import math

#import root
import uproot
import awkward as ak

#import system
import sys

#Import keystroke reader
import getch

###########################################
# Getting Relevant Objects from ROOT File #
###########################################

#The decoded ICEBERG run file
infile_name = sys.argv[1]
infile = uproot.open(infile_name)

#Open the TBranch which contains the RawDigits we care about
events = infile['Events']

#Extract the important information from the RawDigits object: ADC count, number of time ticks, and channel ID
Digits = events['raw::RawDigits_tpcrawdecoder_daq_RunIcebergRawDecoder.obj.fADC']
TimeTicks = events['raw::RawDigits_tpcrawdecoder_daq_RunIcebergRawDecoder.obj.fSamples']
ChannelID = events['raw::RawDigits_tpcrawdecoder_daq_RunIcebergRawDecoder.obj.fChannel']

#Turn these into arrays.
#Future: Learn how to use awkward and turn these into awkward arrays
#We can simply loop over them for now until we need to optimize for speed.
ADCArray = Digits.array() #ADC count for every channel and time tick
TimeTickArray = TimeTicks.array() #Number of time ticks recorded for each channel number
ChannelIDArray = ChannelID.array() #Channel ID for each channel number

#The basic structure of these arrays is as follows:

#ADCArray[nEvent][nChannel][nTimeTick]
#ChannelIDArray[nEvent][nChannel]
#TimeTickArray[nEvent][nChannel]

################################################
#  Key inputs to determine which event to show #
################################################

#initialize nEvent
nEvent = 0

#Takes in an index of the current event, prompts the user 
def keyInput(currentEvent):
	keystroke = input("Input: ")
	if (keystroke=="a"):
		return (currentEvent - 1) % len(ADCArray)
	elif (keystroke=="s"):
		return currentEvent
	elif (keystroke=="d"):
		return (currentEvent + 1) % len(ADCArray)
	elif (keystroke=="f"):
		return int(input("Please enter the index of the event you want to look at: ")) % len(ADCArray)
	else:
		keyInput(currentEvent)

###########################
#  Show the desired event #
###########################

def showEvent(currentEvent):
	ADC_ColorMap = np.array(ADCArray[currentEvent])
	Channels = np.array(ChannelIDArray[currentEvent])
	TimeTicks = np.array(TimeTickArray[currentEvent]) #Number of time ticks for a given channel
	
	#Skip empty events
	if (len(TimeTicks) == 0):
		return -1
	
	#This gives us an array counting the timeticks for each channel in that event. Assume rectangular.
	Time_Tick_Axis = np.arange(TimeTicks[0]) #Assume constant for all channels. Or else this won't end well...
	
	#These are the cutoffs between each channel. The structure of the array is row = plane, column = TPC
	Channel_Cutoffs = [[[800,1039],[1040,1279]],[[0,199],[200,399]],[[400,599],[600,799]]] #Bounds are Inclusive

	#Set the parameters for the pretty plot
	fig,axis = plt.subplots(3,2) #3 rows, 2 columns. TPC 0, 1; Col, U, V
	
	#Fix up the title and axis labels!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	#plt.title("Event Display for nEvent = "+str(currentEvent), size = 40) 
	#plt.xlabel("Channel",size=20)
	#plt.ylabel("Time Tick",size=20)
	
	#iPlane: 0 = Col, 1 = U, 2 = V
	for iPlane in range(3):
		for iTPC in range(2):
			This_ColorMap = ADC_ColorMap[Channels <= Channel_Cutoffs[iPlane][iTPC][1]]
			#This_TimeTicks = TimeTicks[Channels <= Channel_Cutoffs[iPlane][iTPC][1]]
			This_Channels = Channels[Channels <= Channel_Cutoffs[iPlane][iTPC][1]]
			
			This_ColorMap = This_ColorMap[This_Channels >= Channel_Cutoffs[iPlane][iTPC][0]]
			#This_TimeTicks = This_TimeTicks[This_Channels >= Channel_Cutoffs[iPlane][iTPC][0]]
			This_Channels = This_Channels[This_Channels >= Channel_Cutoffs[iPlane][iTPC][0]]

			#fig.set_size_inches(7, 5)
			#axis[iPlane][iTPC].set_xlabel("Channel",size=20)
			#axis[iPlane][iTPC].set_ylabel("Time Tick",size=20)
			#axis[iPlane][iTPC].set_xticks(fontsize = 15)
			#axis[iPlane][iTPC].set_yticks(fontsize = 15)
			#fig.colorbar(axis[iPlane][iTPC].pcolor(This_ColorMap)) #Legend for our colormap - see https://stackoverflow.com/questions/2451264/creating-a-colormap-legend-in-matplotlib

			# To assign the colors, I use pcolormesh, with a gouraud shading.
			axis[iPlane][iTPC].pcolormesh(This_Channels,Time_Tick_Axis,This_ColorMap.T,shading='gouraud', vmin=np.min(ADC_ColorMap),vmax=np.max(ADC_ColorMap))
	
	#Manually add in a colorbar!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	#fig.subplots_adjust(right=0.8)
	#cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
	#fig.colorbar(axis[0][0].pcolor(ADC_ColorMap), cax=cbar_ax)
	
	#Show the figure
	plt.show()

###################
# Run the program #
###################

while(True):
	showEvent(nEvent)
	nEvent = keyInput(nEvent)
	plt.close()



