#include <stdio.h>
#include <iostream>
#include <vector>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "t1.h"

#include <TMath.h>
#include <TFile.h>
#include <TTree.h>
#include <TH1F.h>
#include <TCanvas.h> 
#include <TLorentzVector.h>



//------------------------------------------------------------------------------
// Particle Class
//
class Particle{

	public:
	Particle();
	Particle(double, double, double, double);
	double   pt, eta, phi, E, m, p[4];
	void     p4(double, double, double, double);
	void     print();
	void     setMass(double);
	double   sintheta();
};

class Lepton : public Particle {
	public:
		int charge;
		void setCharge(int);
		void print();		
};

class Jet : public Particle {
	public:
		int flavour;
		void setFlavour(int);
		void print();
};

//------------------------------------------------------------------------------

//*****************************************************************************
//                                                                             *
//    MEMBERS functions of the Particle Class                                  *
//                                                                             *
//*****************************************************************************

//
//*** Default constructor ------------------------------------------------------
//
Particle::Particle(){
	pt = eta = phi = E = m = 0.0;
	p[0] = p[1] = p[2] = p[3] = 0.0;
}

//*** Additional constructor ------------------------------------------------------
Particle::Particle(double p0, double p1, double p2, double p3){ 
	p[0] = p0;
	p[1] = p1;
	p[2] = p2;
	p[3] = p3;
	ROOT::Math::PxPyPzEVector vec(p0, p1, p2, p3);
	pt = vec.pt();
	eta = vec.eta();
	phi = vec.phi();
	E = vec.E();
	m = vec.M();
}


//
//*** Members  ------------------------------------------------------
//
double Particle::sintheta(){
	return 1.0 / std::cosh(eta);
}

void Particle::p4(double pT, double eta, double phi, double energy){
	pt = pT;
	this->eta = eta;
	this->phi = phi;
	E = energy;
	
	ROOT::Math::PtEtaPhiEVector vec(pt, eta, phi, E);
	m = vec.M();
	p[0] = vec.px();
	p[1] = vec.py();
	p[2] = vec.pz();
	p[3] = vec.E();
}

void Particle::setMass(double mass)
{
	m = mass;
	E = std::sqrt(p[0]*p[0] + p[1]*p[1] + p[2]*p[2] + m*m);
}

//
//*** Prints 4-vector ----------------------------------------------------------
//
void Particle::print() {
	std::cout << std::endl;
	std::cout << "(" << p[0] <<",\t" << p[1] <<",\t"<< p[2] <<",\t"<< p[3] << ")" << "  " <<  sintheta() << std::endl;
}

void Lepton::setCharge(int charge) {
	this->charge = charge;
}

void Lepton::print() {
	Particle::print();
	std::cout << "Charge: " << charge << std::endl;
}

void Jet::setFlavour(int flavour) {
	this->flavour = flavour;
}

void Jet::print() {
	Particle::print();
	std::cout << "Flavour: " << flavour << std::endl;
}

int main() {
	
	/* ************* */
	/* Input Tree   */
	/* ************* */

	TFile *f      = new TFile("input.root","READ");
	TTree *t1 = (TTree*)(f->Get("t1"));

	// Read the variables from the ROOT tree branches
	t1->SetBranchAddress("nleps",&nleps);
	t1->SetBranchAddress("lepPt",&lepPt);
	t1->SetBranchAddress("lepEta",&lepEta);
	t1->SetBranchAddress("lepPhi",&lepPhi);
	t1->SetBranchAddress("lepE",&lepE);
	t1->SetBranchAddress("lepQ",&lepQ);
	
	t1->SetBranchAddress("njets",&njets);
	t1->SetBranchAddress("jetPt",&jetPt);
	t1->SetBranchAddress("jetEta",&jetEta);
	t1->SetBranchAddress("jetPhi",&jetPhi);
	t1->SetBranchAddress("jetE", &jetE);
	t1->SetBranchAddress("jetHadronFlavour",&jetHadronFlavour);

	// Total number of events in ROOT tree
	Long64_t nentries = t1->GetEntries();
	
	Particle lep(0., 0., 0., 0.);

	for (Long64_t jentry=0; jentry<100;jentry++)
 	{
		t1->GetEntry(jentry);
		std::cout<<" Event "<< jentry <<std::endl;	
		
		for (int ilep = 0; ilep < nleps; ilep++) {
			Lepton lep;
			lep.p4(static_cast<double>(lepPt[ilep]), static_cast<double>(lepEta[ilep]), static_cast<double>(lepPhi[ilep]), static_cast<double>(lepE[ilep]));
			lep.setCharge(static_cast<int>(lepQ[ilep]));
			lep.print();
		}

		for (int ijet = 0; ijet < njets; ijet++) {
			Jet jet;
			jet.p4(static_cast<double>(jetPt[ijet]), static_cast<double>(jetEta[ijet]), static_cast<double>(jetPhi[ijet]), static_cast<double>(jetE[ijet]));
			jet.setFlavour(static_cast<int>(jetHadronFlavour[ijet]));
			jet.print();
		}
		std::cout << std::endl;

	} // Loop over all events

  	return 0;
}
