#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include "canvas/Utilities/InputTag.h"
#include "gallery/Event.h"

#include "TFile.h"
#include "TInterpreter.h"
#include "TROOT.h"
#include "TGraph.h"
#include "TH1D.h"
#include "TStyle.h"
#include "TColor.h"
#include "TMath.h"
#include "TVectorT.h"
#include "TCanvas.h"
#include "lardataobj/RawData/RawDigit.h"
#include "TLegend.h"
#include <vector>
#include <set>
#include "Riostream.h"
#include "TString.h"
#include "TH1F.h"
#include "TStyle.h"
#include "TROOT.h"

using namespace art;
using namespace std;


void ICEBERG_Waveform_Creator(
	std::string const& filename="/exp/dune/app/users/mking/dunesw_v09_82_00d00/ICEBERG_Noise_Ar_39/iceberg_noise/AR_39_sim/Ar39_50.root", 
	//std::string const& filename = "/exp/dune/data/users/mking/mking/ICEBERG_Run5_Decoded/no_pulser_or_cosmics/iceberg_r009693_sr01_20210404T192346_1_dl1_decode.root",
    std::string const& inputtag="tpcrawdecoder:daq")
    //std::string const& inputtag="tpcrawdecoder:sig")
{
  //Output TFile:
  std::string const& outputfile="Waveforms_Data_6_Runs.root";

  std::unique_ptr<TFile> myFile(TFile::Open(outputfile.c_str(), "RECREATE"));
  myFile->cd();
  //myFile = ROOT.TFile.Open(outputfile, "RECREATE")

  size_t evcounter=0;

  InputTag rawdigit_tag{ inputtag };
  //vector<string> filenames(1, filename);

cout<<"Creating the list of filenames"<<endl;

  vector<string> filenames = {
"iceberg_r009693_sr01_20210404T192346_1_dl1_decode.root", "iceberg_r009697_sr01_20210404T194635_1_dl1_decode.root", "iceberg_r009702_sr01_20210404T204459_1_dl1_decode.root",  "iceberg_r009706_sr01_20210404T210522_1_dl1_decode.root",  "iceberg_r009710_sr01_20210404T212534_1_dl1_decode.root"};
//,
//"iceberg_r009694_sr01_20210404T192925_1_dl1_decode.root",  "iceberg_r009698_sr01_20210404T195219_1_dl1_decode.root",  "iceberg_r009703_sr01_20210404T205038_1_dl1_decode.root",  "iceberg_r009707_sr01_20210404T211101_1_dl1_decode.root",  "iceberg_r009711_sr01_20210404T213002_1_dl1_decode.root",
//"iceberg_r009695_sr01_20210404T193508_1_dl1_decode.root",  "iceberg_r009699_sr01_20210404T195808_1_dl1_decode.root",  "iceberg_r009704_sr01_20210404T205505_1_dl1_decode.root",  "iceberg_r009708_sr01_20210404T211527_1_dl1_decode.root",
//"iceberg_r009696_sr01_20210404T194054_1_dl1_decode.root",  "iceberg_r009701_sr01_20210404T200822_1_dl1_decode.root"  ,"iceberg_r009705_sr01_20210404T205939_1_dl1_decode.root", "iceberg_r009709_sr01_20210404T212107_1_dl1_decode.root"};

cout<<"Adding the path to the filenames"<<endl;

//add the path to the filenames
for (int i=0; i<5; i++)
{
  filenames[i] = "/exp/dune/data/users/mking/mking/ICEBERG_Run5_Decoded/no_pulser_or_cosmics/" + filenames[i];
}

  //TFile *myFile = new TFile(outputfile.c_str(), "RECREATE"

cout<<"Filelist created"<<endl;


// loop over all the events
  for (gallery::Event ev(filenames); !ev.atEnd(); ev.next()) {

	auto const& evaux = ev.eventAuxiliary();
    int runno = evaux.run();
    int subrunno = evaux.subRun();
    int eventno = evaux.event();

	//for debugging purposes, print the event number to screen
	std::cout<<"Processing Run "<<runno<<" Subrun "<<subrunno<<" Event "<<eventno<<std::endl;

	auto const& rawdigits = *ev.getValidHandle<vector<raw::RawDigit>>(rawdigit_tag);
	if (!rawdigits.empty())
	  {
		int numchannels = 1280;
		int numtimeticks = 2128;

		//channel on x, time on y
		std::string histname = "run_" + std::to_string(runno) + "_sub_" + std::to_string(subrunno) + "_event_" + std::to_string(eventno) + "_daq";
		TH2F *outhist = (TH2F*) new TH2F(histname.c_str(),histname.c_str(),numchannels,0,numchannels,numtimeticks,0,numtimeticks);
		outhist->SetDirectory(0);
		outhist->GetXaxis()->SetTitle("Channel");
		outhist->GetYaxis()->SetTitle("Time Tick");
		outhist->GetZaxis()->SetTitle("ADC");
	
	    const size_t nrawdigits = rawdigits.size();
	    for (size_t ichan=0;ichan<nrawdigits; ++ichan) 
	      { 
	        size_t thigh = rawdigits[ichan].Samples()-1; // assume uncompressed
			size_t ic = rawdigits[ichan].Channel();
            for (size_t itick=0;itick<=thigh;++itick)
	        	{
			outhist->SetBinContent(ic,itick,rawdigits[ichan].ADC(itick));
		      }
	      } // loop over channels
		  //myFile->WriteObject(&outhist, histname.c_str());
		  myFile->cd();
		  outhist->Write();
      }//make sure rawdigits is not empty
    ++evcounter;
  }//loop over events
  std::cout<<"Done"<<std::endl;
}//ICEBERG_Waveform_Creator