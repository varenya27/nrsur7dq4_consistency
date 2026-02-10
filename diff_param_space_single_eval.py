'''

20,21: DBUG deltaT and such 0.000244140625 0.25 2048.0

20,20: DBUG deltaT and such 0.000244140625 0.25 2048.0
compute differences between LALSim and 
gwsignal strain signals at a single param 
space poitns. The goal is to zero in on the 
discrepancies in the PE runs

comparison is done for:
1. NRSur7dq4_wrapper vs. lal_binary_black_hole
2. generate_fd_polarizations_from_td vs. SimInspiralFD 
'''

import numpy as np 
from gwsignal4gwsurr.gwsurr import NRSur7dq4_gwsurr
from gwsignal4gwsurr.NRSur7dq4_wrapper import NRSur7dq4_wrapper as NRSur7dq4_gwsurr_wrapper
from gwsignal4gwsurr.NRSur7dq4_LALSim_wrapper import NRSur7dq4_wrapper as NRSur7dq4_LALSim_gwsurr_wrapper
from bilby.gw.conversion import bilby_to_lalsimulation_spins
from bilby.gw.conversion import chirp_mass_and_mass_ratio_to_component_masses as convert_mass
import astropy.units as u
from bilby.core.utils.constants import solar_mass
import matplotlib.pyplot as plt
import lal 
import lalsimulation as lalsim
import time
from bilby.gw.source import lal_binary_black_hole
from pandas.core.generic import freq_to_period_freqstr



def generate_fd_polarizations_from_td(**parameters):
    print('--------------------------')
    print('DBUG generating with params:',parameters)
    print('--------------------------')
    # VU: inspired by LALSimInspiralGeneratorConditioning.c L486
    # Adjust deltaT depending on sampling rate
    fmax = parameters["f_max"].value
    f_nyquist = fmax
    deltaF = 0
    if "deltaF" in parameters.keys():
        deltaF = parameters["deltaF"].value

    if deltaF != 0:
        n = int(np.round(fmax / deltaF))
        if n & (n - 1):
            chirplen_exp = np.frexp(n)
            f_nyquist = np.ldexp(1, int(chirplen_exp[1])) * deltaF

    deltaT = 0.5 / f_nyquist
    parameters["deltaT"] = deltaT*u.s


    hp_,hc_ = sur_gwsignal_.generate_td_waveform(**parameters)


    epoch = lal.LIGOTimeGPS(
        hp_.times[0].value
    )

    hp = lal.CreateREAL8TimeSeries(
        "hplus", epoch, 0, parameters["deltaT"].value, lal.DimensionlessUnit, len(hp_)
    )
    hc = lal.CreateREAL8TimeSeries(
        "hcross", epoch, 0, parameters["deltaT"].value, lal.DimensionlessUnit, len(hc_),
    )

    hp.data.data = hp_.value
    hc.data.data = hc_.value

    # conditioning/tapering is done differently since this is a short waveform 
    # [cf. L#44 in LALSimInspiralGeneratorConditioning.c]
    taper = True
    lalsim.SimInspiralREAL8WaveTaper(hp.data,taper)
    lalsim.SimInspiralREAL8WaveTaper(hc.data,taper)
    
    # Adjust signal duration
    if deltaF == 0:
        chirplen = hp.data.length
        chirplen_exp = np.frexp(chirplen)
        chirplen = int(np.ldexp(1, chirplen_exp[1]))
        deltaF = 1.0 / (chirplen * deltaT)
        parameters["deltaF"] = deltaF

    else:
        chirplen = int(1.0 / (deltaF * deltaT))

    # resize waveforms to the required length
    lal.ResizeREAL8TimeSeries(hp, hp.data.length - chirplen, chirplen)
    lal.ResizeREAL8TimeSeries(hc, hc.data.length - chirplen, chirplen)

    # FFT - Using LAL routines
    hptilde = lal.CreateCOMPLEX16FrequencySeries(
        "FD H_PLUS",
        hp.epoch,
        0.0,
        deltaF,
        lal.DimensionlessUnit,
        int(chirplen / 2.0 + 1),
    )
    hctilde = lal.CreateCOMPLEX16FrequencySeries(
        "FD H_CROSS",
        hc.epoch,
        0.0,
        deltaF,
        lal.DimensionlessUnit,
        int(chirplen / 2.0 + 1),
    )

    plan = lal.CreateForwardREAL8FFTPlan(chirplen, 0)
    lal.REAL8TimeFreqFFT(hctilde, hc, plan)
    lal.REAL8TimeFreqFFT(hptilde, hp, plan)

    return hptilde, hctilde
   


#---------------params----------
mass1 = 42
mass2 = 34
a_1 = 0.5
a_2= 0.3
tilt_1=3*np.pi/4
tilt_2=3*np.pi/4
phi_12=np.pi
phi_jl=np.pi 
distance=410
theta_jn=np.pi
phi_ref=0.

mass1 = 38.
mass2 = 30.
a_1 = 0.5
a_2= 0.4
tilt_1=5*np.pi/8
tilt_2=5*np.pi/8
phi_12=3*np.pi/2
phi_jl=3*np.pi/2 
distance=10
theta_jn=np.pi
phi_ref=0.
#-------------------------------

#----------- freqs ----------------
f22_start = 0.
f_ref = 20.000
maximum_frequency = 2048. 
freqs = np.arange(0,maximum_frequency+0.25,0.25)
#----------------------------------

#------------ wf args dict --------------
waveform_arguments = {
    'reference_frequency':f_ref,
    'reference-frequency':f_ref,
    'catch_waveform_errors':True,
    'minimum_frequency':f22_start,
    'maximum_frequency':maximum_frequency,
    'waveform_approximant':'NRSur7dq4'
}
#----------------------------------------


#------------------ convert spins --------------------------
print('DBUG converting', theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2, mass1, mass2, f_ref, phi_ref)
iota, spin1x, spin1y, spin1z, spin2x, spin2y, spin2z = bilby_to_lalsimulation_spins(
theta_jn = theta_jn,
phi_jl =  phi_jl,
tilt_1 = tilt_1,
tilt_2 = tilt_2,
phi_12 = phi_12,
a_1 = a_1,
a_2 = a_2,
mass_1 = mass1*solar_mass,
mass_2 = mass2*solar_mass,
reference_frequency = f_ref,
phase = phi_ref
)
print('DBUG converted spins', iota, spin1x, spin1y, spin1z, spin2x, spin2y, spin2z)
#-----------------------------------------------------------

#======================================================
#  WF GENERATION CHECK
#======================================================

sur_gwsignal_ = NRSur7dq4_gwsurr()


# hp_gwsig_genfd,hc_gwsig_genfd=  sur_gwsignal_.generate_fd_polarizations_from_td(
hp_gwsig_genfd_temp,hc_gwsig_genfd_temp=  generate_fd_polarizations_from_td(
    mass1=mass1*u.Msun,
    mass2=mass2*u.Msun,
    spin1x=spin1x*u.dimensionless_unscaled,
    spin1y=spin1y*u.dimensionless_unscaled,
    spin1z=spin1z*u.dimensionless_unscaled,
    spin2x=spin2x*u.dimensionless_unscaled,
    spin2y=spin2y*u.dimensionless_unscaled,
    spin2z=spin2z*u.dimensionless_unscaled,
    distance=distance*u.Mpc,
    inclination=iota*u.rad,
    phi_ref=phi_ref*u.rad,
    f22_start=f22_start*u.Hz,
    f22_ref=waveform_arguments['reference-frequency']*u.Hz,
    f_max = waveform_arguments['maximum_frequency']*u.Hz,
    deltaF=(freqs[1]-freqs[0])*u.Hz,
)
hp_gwsig_genfd_temp=hp_gwsig_genfd_temp.data.data
hc_gwsig_genfd_temp=hc_gwsig_genfd_temp.data.data

hp_gwsig_genfd,hc_gwsig_genfd=  sur_gwsignal_.generate_fd_polarizations_from_td(
    mass1=mass1*u.Msun,
    mass2=mass2*u.Msun,
    spin1x=spin1x*u.dimensionless_unscaled,
    spin1y=spin1y*u.dimensionless_unscaled,
    spin1z=spin1z*u.dimensionless_unscaled,
    spin2x=spin2x*u.dimensionless_unscaled,
    spin2y=spin2y*u.dimensionless_unscaled,
    spin2z=spin2z*u.dimensionless_unscaled,
    distance=distance*u.Mpc,
    inclination=iota*u.rad,
    phi_ref=phi_ref*u.rad,
    f22_start=f22_start*u.Hz,
    f22_ref=waveform_arguments['reference-frequency']*u.Hz,
    f_max = waveform_arguments['maximum_frequency']*u.Hz,
    deltaF=(freqs[1]-freqs[0])*u.Hz,
)
hp_gwsig_genfd=hp_gwsig_genfd.data.data
hc_gwsig_genfd=hc_gwsig_genfd.data.data


# generate lalsim waveforms
hp_lal_siminspiral, hc_lal_siminspiral = lalsim.SimInspiralFD(
    mass1*solar_mass, mass2*solar_mass,
    spin1x, spin1y, spin1z,
    spin2x, spin2y, spin2z,
    distance*1.e6*lal.PC_SI, iota, phi_ref, 0.,0.,0.,
    freqs[1]-freqs[0],
    f22_start, maximum_frequency,f_ref, # f_low, f_max, f_ref
    lal.CreateDict(),
    lalsim.GetApproximantFromString('NRSur7dq4')
)

hp_lal_siminspiral=hp_lal_siminspiral.data.data
hc_lal_siminspiral=hc_lal_siminspiral.data.data

print('DBUG used params for wfgen', mass1,mass2,spin1x,spin1y,spin1z,spin2x,spin2y,spin2z,distance,iota,phi_ref,waveform_arguments['minimum_frequency'],maximum_frequency,waveform_arguments['reference_frequency'],freqs[1]-freqs[0])


#======================================================
#  WRAPPER CHECK
#======================================================

# generate gwsignal4gwsurr gwsurr waveforms
sur_gwsignal = NRSur7dq4_gwsurr_wrapper(
freqs, mass1,mass2,a_1,a_2,tilt_1,tilt_2,phi_12, phi_jl,distance,theta_jn,phi_ref,**waveform_arguments
)

# generate gwsignal4gwsurr lalsim waveforms
sur_gwsignal_lal = NRSur7dq4_LALSim_gwsurr_wrapper(
freqs, mass1,mass2,a_1,a_2,tilt_1,tilt_2,phi_12, phi_jl,distance,theta_jn,phi_ref,**waveform_arguments
)

# generate bilby waveforms
waveform_polarizations = lal_binary_black_hole(
    freqs, mass1,mass2,distance,a_1,tilt_1,phi_12,a_2,tilt_2, phi_jl,theta_jn,phi_ref,**waveform_arguments
)

# rename variables
if sur_gwsignal is not None and sur_gwsignal_lal is not None and waveform_polarizations is not None:
    hp_gwsig_gwsurr = sur_gwsignal['plus']
    hc_gwsig_gwsurr = sur_gwsignal['cross']
    hp_gwsig_lalsim = sur_gwsignal_lal['plus']
    hc_gwsig_lalsim = sur_gwsignal_lal['cross']
    hp_bilby_lalbin = waveform_polarizations['plus']
    hc_bilby_lalbin = waveform_polarizations['cross']
else: quit()


#======================================================
#  PLOT EVERYTHING
#======================================================
plt.figure(figsize=(10,4))
fig,ax = plt.subplots(2,2, figsize=(10,8), sharex=True)

#----------- NRSur7dq4_gwsurr vs SimInspiralFD --------------
ax[0][0].plot(freqs,hp_gwsig_genfd, label='gwsig_genfd hp')
ax[0][0].plot(freqs,hp_lal_siminspiral, ls='--', label='lalsim_simfd hp')
ax[0][0].legend()
ax[0][0].set_xscale('log')
ax[0][0].set_xlim(10,1000)
ax[0][0].set_title('NRSur7dq4_gwsurr vs SimInspiralFD')

ax[1][0].plot(freqs,hp_lal_siminspiral-hp_gwsig_genfd, label='lalsim_simfd - gwsig_genfd')
ax[1][0].plot(freqs,hp_gwsig_genfd_temp-hp_gwsig_genfd, label='temp - gwsig_genfd')
ax[1][0].legend()
#------------------------------------------------------------

#----------- wrapper vs lal_binary_black_hole --------------
ax[0][1].plot(freqs,hp_gwsig_gwsurr, ls='-' ,label='gwsig gwsurr hp')
ax[0][1].plot(freqs,hp_gwsig_lalsim, ls='-.',label='gwsig lalsim hp')
ax[0][1].plot(freqs,hp_bilby_lalbin, ls='--',label='bilby lalbin hp')
ax[0][1].legend()
ax[0][1].set_xscale('log')
ax[0][1].set_xlim(10,1000)
ax[0][1].set_title('%d NRSur7dq4_gwsurr_wrapper vs lal_binary'%np.random.randint(100,999))

ax[1][1].plot(freqs,hp_gwsig_gwsurr-hp_gwsig_lalsim,ls='-' , label='gwsig_gwsur - gwsig_lal')
ax[1][1].plot(freqs,hp_gwsig_gwsurr-hp_bilby_lalbin,ls='-.', label='gwsig_gwur - bilby_lal')
ax[1][1].plot(freqs,hp_gwsig_lalsim-hp_bilby_lalbin,ls='--', label='gwsig_lal - bilby_lal')
ax[1][1].legend()
#-----------------------------------------------------------

#------------- save fig ---------------
plt.tight_layout()
plt.savefig('hp_diff_single_eval.png')
#--------------------------------------

#------------- error readoff ---------------
hp_diff_wfgen   = hp_lal_siminspiral - hp_gwsig_genfd
hp_diff_wrapper = hp_gwsig_gwsurr - hp_bilby_lalbin

hp_linf_wfgen = np.linalg.norm(hp_diff_wfgen, np.inf)/np.linalg.norm(hp_lal_siminspiral, np.inf)
hp_l2_wfgen   = np.linalg.norm(hp_diff_wfgen, 2)/np.linalg.norm(hp_lal_siminspiral, 2)
hp_linf_wrapper =  np.linalg.norm(hp_diff_wrapper, np.inf)/np.linalg.norm(hp_bilby_lalbin, np.inf)
hp_l2_wrapper   =  np.linalg.norm(hp_diff_wrapper, 2)/np.linalg.norm(hp_bilby_lalbin, 2)

print('wfgen linf   = %.3e'%hp_linf_wfgen)
print('wfgen linf   = %.3e'%hp_l2_wfgen)
print('wrapper linf = %.3e'%hp_linf_wrapper)
print('wrapper l2   = %.3e'%hp_l2_wrapper)












