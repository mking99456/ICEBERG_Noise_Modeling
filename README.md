# ICEBERG_Noise_Modeling
#
## Project Overview
#The ICEBERG Noise Modeling Project is a methodology for characterizing and simulating the electronics noise in the ICEBERG LArTPC at Fermilab. Following this procedure, we will model other sources of background (such as Ar-39 radiological deposits) for low-energy events (such as supernova neutrinos). ICEBERG is a small TPC serving as a proof-of-concept for our methods before we apply them to the DUNE experiment.
#
## Notes about our code
### Noise Simulation
The noise simulation code follows M. Carrettoni and O. Cremonesi, "Generation of Noise Time Series with arbitrary Power Spectrum" and works as follows: 
1. Start with our noise model in the form of a power spectrum. (P(0) = 0)
2. Divide out the normalization (alpha) used for the power spectrum.
3. Let F(w) = the square root of the unnormalized power spectrum times uniformly generated random phases.
4. Let f(t) be the inverse real fast fourier transform of F
5. Let l be a tunable overlap rate of signals. I have it tuned to allow for 50-100 copies of the signal waveform to be added to itself.
6. Generate time delays according to a Poisson distribution. Stop generating when the total delayed time is greater than the domain of f(t), T (Length in time of a waveform).
7. Generate random amplitudes a_k from Norm(0,alpha/(l*T))
8. Our noise waveform is n(t) = sum_k a_k*f(t-t_k)
