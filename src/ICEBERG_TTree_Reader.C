#include <string>
#include <iostream>
#include "TFile.h"
#include "TTree.h"
#include "TF1.h"
#include "TCanvas.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TRandom2.h"
#include "TMath.h"
#include <TMinuit.h>
#include "Math/Minimizer.h"
#include "Math/Factory.h"
#include "Math/Functor.h"
#include "TFitter.h"
#include "TVector3.h"
#include "TStyle.h"
#include "TLine.h"
#include <time.h>
#include <vector>
#include <math.h>
#include <fstream>

#include lardataobj/RawData/RawDigit.h

std::string             fFileName     ="/exp/dune/app/users/mking/dunesw_v09_82_00d00/ICEBERG_Noise_Ar_39/iceberg_noise/AR_39_sim/Ar39.root";
std::string             fTreeName     = "Events";

float daq_fChannel[1280];
float daq_fADC[1280][2128];
float sig_fChannel[1280];
float sig_fADC[1280][2128];

TFile *f = new TFile(fFileName.c_str());
TTree *t = (TTree*)f->Get(fTreeName.c_str());

//std::vector<raw::RawDigit> daq;
//root [5] t->SetBranchAddress("raw::RawDigits_tpcrawdecoder_daq_SinglesGen.obj",&daq)

t->SetBranchAddress("raw::RawDigits_tpcrawdecoder_daq_SinglesGen./raw::RawDigits_tpcrawdecoder_daq_SinglesGen.obj/fChannel",&daq_fChannel);
t->SetBranchAddress("raw::RawDigits_tpcrawdecoder_daq_SinglesGen./raw::RawDigits_tpcrawdecoder_daq_SinglesGen.obj/fADC",&daq_fADC);
t->SetBranchAddress("raw::RawDigits_tpcrawdecoder_sig_SinglesGen./raw::RawDigits_tpcrawdecoder_sig_SinglesGen.obj/fChannel",&sig_fChannel);
t->SetBranchAddress("raw::RawDigits_tpcrawdecoder_sig_SinglesGen./raw::RawDigits_tpcrawdecoder_sig_SinglesGen.obj/fADC",&sig_fADC);

for(int iEvent=0; iEvent<t->GetEntries(); iEvent++){
    t->GetEntry(iEvent);
    for(int i=0; i<5; i++){
        std::cout << "Event: " << iEvent << " Channel: " << daq_fChannel[i] << " ADC: " << daq_fADC[i] << std::endl;
    }
}


