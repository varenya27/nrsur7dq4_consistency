'''
compute differences between LALSim and 
gwsignal strain signals at multiple param 
space points, and make histograms
The goal is to zero in on the 
discrepancies in the PE runs

comparison is done for:
1. NRSur7dq4_wrapper vs. lal_binary_black_hole
2. generate_fd_polarizations_from_td vs. SimInspiralFD 
'''

import numpy as np 
from gwsignal4gwsurr.gwsurr import NRSur7dq4_gwsurr
from gwsignal4gwsurr.utils_bilby import SurrogateWaveformGenerator
from bilby.gw.conversion import bilby_to_lalsimulation_spins
import astropy.units as u
from bilby.core.utils.constants import solar_mass
import matplotlib.pyplot as plt
import lal 
import lalsimulation as lalsim
from mpi4py import MPI
import argparse

print('done importing')
parser = argparse.ArgumentParser(description='testing parameter space for lalsim/gwsignal NRSur7dq4',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--fmin', type=float, default=20., help='minimum frequency')

args = parser.parse_args()

#---------------params----------
testing_params=dict(
mass1 = 42,
mass2 = 34,
a_1 = 0.5,
a_2= 0.3,
tilt_1=3*np.pi/4,
tilt_2=3*np.pi/4,
phi_12=np.pi,
phi_jl=np.pi ,
distance=410,
theta_jn=np.pi,
phi_ref=0.)
#-------------------------------

#----------- freqs ----------------
freqs = np.arange(0,2048.25,0.25)
f22_start = args.fmin
f_ref = 20.
maximum_frequency = 2048.
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


print('hello')
sur_gwsignal_ = NRSur7dq4_gwsurr()
def get_polarizations(
    mass1 = 42.,
    mass2 = 34.,
    a_1 = 0.5,
    a_2= 0.3,
    tilt_1=3*np.pi/4,
    tilt_2=3*np.pi/4,
    phi_12=np.pi,
    phi_jl=np.pi ,
    distance=410.,
    theta_jn=np.pi,
    phi_ref=0.,
    waveformGenerator_gwsignal_gwsurr=None,
    waveformGenerator_gwsignal_lalsim=None,
    waveformGenerator_bilby_lalbin=None
):
    parameters = dict(
        mass_1 = mass1,
        mass_2 = mass2,
        a_1 = a_1,
        a_2 = a_2,
        tilt_1 = tilt_1,
        tilt_2 = tilt_2,
        phi_12 = phi_12,
        phi_jl = phi_jl,
        theta_jn = theta_jn,
        luminosity_distance = distance,
        phase = phi_ref,
        reference_frequency = 20.0,
    )

    #------------------ convert spins --------------------------
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
    #-----------------------------------------------------------

    #======================================================
    #  WF GENERATION CHECK
    #======================================================

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

    #======================================================
    #  WRAPPER CHECK
    #======================================================


    if not isinstance(waveformGenerator_gwsignal_gwsurr, SurrogateWaveformGenerator) or not isinstance(waveformGenerator_gwsignal_lalsim, SurrogateWaveformGenerator) or not isinstance(waveformGenerator_bilby_lalbin, SurrogateWaveformGenerator):
        quit()
        
    fd_strain_gwsignal_gwsurr = waveformGenerator_gwsignal_gwsurr.frequency_domain_strain(parameters)
    fd_strain_gwsignal_lalsim = waveformGenerator_gwsignal_lalsim.frequency_domain_strain(parameters)
    fd_strain_bilby = waveformGenerator_bilby_lalbin.frequency_domain_strain(parameters)


    hp_gwsig_gwsurr, hc_gwsig_gwsurr = fd_strain_gwsignal_gwsurr['plus'], fd_strain_gwsignal_gwsurr['cross']
    hp_gwsig_lalsim, hc_gwsig_lalsim = fd_strain_gwsignal_lalsim['plus'], fd_strain_gwsignal_lalsim['cross']
    hp_bilby_lalbin, hc_bilby_lalbin = fd_strain_bilby['plus'], fd_strain_bilby['cross']

    return (
        hp_gwsig_genfd, hc_gwsig_genfd,
        hp_lal_siminspiral, hc_lal_siminspiral,
        hp_gwsig_gwsurr, hc_gwsig_gwsurr,
        hp_gwsig_lalsim, hc_gwsig_lalsim,
        hp_bilby_lalbin, hc_bilby_lalbin
    )

    
# initialize a bunch of param space points
mass1_all = [38., 39., 40., 41., 42., ]
mass2_all = [30., 31., 32., 33., 34., ]

a1_all = [0.2, 0.25,0.3, 0.33, 0.35,0.4,0.5]
a2_all = [0.2, 0.25,0.3, 0.33, 0.35,0.4,0.5]

tilt1_all = [5.*np.pi/8, 3.*np.pi/4, 7.*np.pi/8]
tilt2_all = [5.*np.pi/8, 3.*np.pi/4, 7.*np.pi/8]

phi12_all = [np.pi, 5.*np.pi/4, 3.*np.pi/2]
phijl_all = [np.pi, 5.*np.pi/4, 3.*np.pi/2]

distance_all = [480.,410., 420.,450.]
theta_jn_all = [ 0, np.pi/4, np.pi/2, 3.*np.pi/4, np.pi]

n_cases = (len(mass1_all)*len(a1_all)*len(a2_all)* len(tilt1_all)*len(phijl_all)*len(distance_all))
print('INFO evaluating %d cases'%n_cases)


errors={
    'wrapper':{ 'hp_linf':[], 'hp_l2':[], },
    'wfgen':{ 'hp_linf':[], 'hp_l2':[], },
}
errors_local={
    'wrapper':{ 'hp_linf':[], 'hp_l2':[], },
    'wfgen':{ 'hp_linf':[], 'hp_l2':[], },
}


def iter_params():
    for mass1, mass2 in zip(mass1_all, mass2_all):
        for a_1 in a1_all:
            for a_2 in a2_all:
                for tilt_1 , tilt_2 in zip(tilt1_all, tilt2_all):
                    for phi_12, phi_jl in zip(phi12_all,phijl_all):
                        for distance in distance_all:
                            for theta_jn in theta_jn_all:
                                yield mass1, mass2, a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl, distance, theta_jn

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

params_all = list(iter_params())
wf_args_gwsignal_gwsurr = waveform_arguments.copy()
wf_args_gwsignal_gwsurr['waveform_approximant']='NRSur7dq4'
wf_args_gwsignal_lalsim = waveform_arguments.copy()
wf_args_gwsignal_lalsim['waveform_approximant']='NRSur7dq4_LALSim'
wf_args_bilby_lalbin = wf_args_gwsignal_gwsurr.copy()

kwargs_gwsignal_gwsurr = {'duration': 4.0, 'start_time': 1126259598.0, 'sampling_frequency': 4096,'waveform_arguments':wf_args_gwsignal_gwsurr}
kwargs_gwsignal_lalsim = {'duration': 4.0, 'start_time': 1126259598.0, 'sampling_frequency': 4096,'waveform_arguments':wf_args_gwsignal_lalsim}
kwargs_bilby_lalbin = {'duration': 4.0, 'start_time': 1126259598.0, 'sampling_frequency': 4096,'waveform_arguments':wf_args_bilby_lalbin,'use_bilby':True}

waveformGenerator_gwsignal_gwsurr = SurrogateWaveformGenerator(**kwargs_gwsignal_gwsurr)
waveformGenerator_gwsignal_lalsim = SurrogateWaveformGenerator(**kwargs_gwsignal_lalsim)
waveformGenerator_bilby_lalbin = SurrogateWaveformGenerator(**kwargs_bilby_lalbin)

print('PROG starting evaluation')
for i,params in enumerate(params_all):
    if i % size != rank:
        continue
    mass1, mass2, a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl, distance, theta_jn = params
    (
        hp_gwsig_genfd, hc_gwsig_genfd,
        hp_lal_siminspiral, hc_lal_siminspiral,
        hp_gwsig_gwsurr, hc_gwsig_gwsurr,
        hp_gwsig_lalsim, hc_gwsig_lalsim,
        hp_bilby_lalbin, hc_bilby_lalbin
    ) = get_polarizations(
        mass1=mass1,
        mass2=mass2,
        a_1=a_1,
        a_2=a_2,
        tilt_1=tilt_1,
        tilt_2=tilt_2,
        phi_12=phi_12,
        phi_jl=phi_jl,
        distance=distance,
        theta_jn=theta_jn,
        phi_ref=0.,
        waveformGenerator_gwsignal_gwsurr=waveformGenerator_gwsignal_gwsurr,
        waveformGenerator_gwsignal_lalsim=waveformGenerator_gwsignal_lalsim,
        waveformGenerator_bilby_lalbin=waveformGenerator_bilby_lalbin
    )
    #------------ compute errors --------------
    hp_diff_wfgen   = hp_lal_siminspiral - hp_gwsig_genfd
    hp_diff_wrapper = hp_gwsig_gwsurr - hp_bilby_lalbin
    #-----------------------------------------

    #------------ store errors ---------------
    hp_linf_wfgen = np.linalg.norm(hp_diff_wfgen, np.inf)/np.linalg.norm(hp_lal_siminspiral, np.inf)
    hp_l2_wfgen   = np.linalg.norm(hp_diff_wfgen, 2)/np.linalg.norm(hp_lal_siminspiral, 2)
    hp_linf_wrapper =  np.linalg.norm(hp_diff_wrapper, np.inf)/np.linalg.norm(hp_bilby_lalbin, np.inf)
    hp_l2_wrapper   =  np.linalg.norm(hp_diff_wrapper, 2)/np.linalg.norm(hp_bilby_lalbin, 2)

    errors_local['wfgen']['hp_linf'].append(hp_linf_wfgen)
    errors_local['wfgen']['hp_l2'].append(hp_l2_wfgen)
    errors_local['wrapper']['hp_linf'].append(hp_linf_wrapper)
    errors_local['wrapper']['hp_l2'].append(hp_l2_wrapper)


    if hp_linf_wrapper > 1e-4:
        with open(f'large_errors_wrapper_rank{rank}.txt','a') as f:
            f.write('mass1=%.2f, mass2=%.2f, a1=%.2f, a2=%.2f, tilt1=%.2f, tilt2=%.2f, phi12=%.2f, phijl=%.2f, distance=%.2f, theta_jn=%.2f, hp_linf_wrapper=%.3e\n'%(
                mass1, mass2, a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl, distance, theta_jn, hp_linf_wrapper
            ))
    if hp_linf_wrapper > 1e-4:
        with open(f'large_errors_wfgen_rank{rank}.txt','a') as f:
            f.write('mass1=%.2f, mass2=%.2f, a1=%.2f, a2=%.2f, tilt1=%.2f, tilt2=%.2f, phi12=%.2f, phijl=%.2f, distance=%.2f, theta_jn=%.2f, hp_linf_wrapper=%.3e\n'%(
                mass1, mass2, a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl, distance, theta_jn, hp_linf_wrapper
            ))
    if rank==0:
        print('PROG linf_wfgen=%.3e, l2_wfgen=%.3e, linf_wrapper=%.3e, l2_wrapper=%.3e'%(
            hp_linf_wfgen, hp_l2_wfgen, hp_linf_wrapper, hp_l2_wrapper))
    
errors_all = comm.gather(errors_local, root=0)

if rank == 0:
    errors = {
        'wrapper': {'hp_linf': [], 'hp_l2': []},
        'wfgen':   {'hp_linf': [], 'hp_l2': []},
    }

    for e in errors_all:
        for k in errors:
            for m in errors[k]:
                errors[k][m].extend(e[k][m])

    # now safe to save
    np.savez(
       f"param_space_errors_{int(f22_start)}hz.npz",
        wrapper_hp_linf=np.array(errors['wrapper']['hp_linf']),
        wrapper_hp_l2=np.array(errors['wrapper']['hp_l2']),
        wfgen_hp_linf=np.array(errors['wfgen']['hp_linf']),
        wfgen_hp_l2=np.array(errors['wfgen']['hp_l2']),
    )


    #======================================================
    #  PLOTS 
    #======================================================
    fig,ax = plt.subplots(1,2, figsize=(12,5), )

    #----------- NRSur7dq4_gwsurr vs SimInspiralFD --------------
    ax[0].hist(errors['wfgen']['hp_linf'],density=True, bins=np.logspace(np.log10(1e-7),np.log10(1e-1),30), histtype='step', label='L_inf')
    ax[0].hist(errors['wfgen']['hp_l2'], density=True,bins=np.logspace(np.log10(1e-7),np.log10(1e-1),30), histtype='step', label='L_2')
    ax[0].legend()
    ax[0].set_xscale('log')
    ax[0].set_title(f'NRSur7dq4_gwsurr.gen_fd vs SimInspiralFD f_low={f22_start}')
    #------------------------------------------------------------

    #----------- wrapper vs lal_binary_black_hole --------------
    ax[1].hist(errors['wrapper']['hp_linf'], bins=np.logspace(np.log10(1e-7),np.log10(1e-1),30), histtype='step', label='L_inf')
    ax[1].hist(errors['wrapper']['hp_l2'], bins=np.logspace(np.log10(1e-7),np.log10(1e-1),30), histtype='step', label='L_2')
    ax[1].legend()
    ax[1].set_xscale('log')
    ax[1].set_title('NRSur7dq4_wrapper vs lal_binary_black_hole')
    #-----------------------------------------------------------

    #------------- save fig ---------------
    plt.tight_layout()
    plt.savefig(f'histogram_hp_wfgen_wrapper_{int(f22_start)}hz.png',dpi=150)
    #--------------------------------------















