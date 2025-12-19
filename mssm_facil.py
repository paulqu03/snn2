from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os


save_dir = "dynamic_tests"
rate_dir = "dynamic_tests/rates"
os.makedirs(rate_dir, exist_ok=True)

# run sim
def run_sim(rate):

    duration = 1*second
    defaultclock.dt = 0.1*ms

    period = 1.0 / rate
    spike_times = np.arange(period/ms, float(duration/ms), period/ms) * ms
    inp = SpikeGeneratorGroup(1, np.zeros(len(spike_times)), spike_times)

    # LIF Neuron
    gL   = 1.21*mS
    Rm   = 1/gL
    tau_m = 30*ms
    EL   = -70*mV
    v_th = -55*mV
    v_r  = -70*mV

    eqs_neuron = '''
    dv/dt = ((EL - v) + Rm*I_syn)/tau_m : volt
    I_syn : amp
    '''
    post = NeuronGroup(1, eqs_neuron, threshold='v>v_th',
                       reset='v=v_r', method='euler')
    post.v = EL

    # MSSM Synapse
    tau_c    = 49.9904393*ms
    tau_v    = 48.8095651*ms
    tau_Nt   = 1.52095675*ms
    tau_epsc = 0.436917566*ms

    alpha   = 1.28833999e-02
    k_Nt    = 7.71595702e-01
    k_NtV   = 6.46772602e+01
    k_epsc  = 1.158519302e-2

    V0 = 2.92508311
    P0 = 2.11579945e-04
    C0 = -np.log(1-P0)/V0
   

    A_SE = 1000*mA

    syn_eqs = '''
    dC/dt     = (C0 - C)/tau_c : 1
    dVres/dt  = (V0 - Vres)/tau_v : 1
    dNt/dt    = -(k_Nt/tau_Nt)*Nt : 1
    dEpsc/dt  = (-Epsc + k_epsc*Nt)/tau_epsc : 1
    P         = (1 - exp(-(C*Vres))) : 1

    I_out     = A_SE * Epsc : amp
    I_syn_post = I_out : amp (summed)
    '''

    syn = Synapses(inp, post, model=syn_eqs, method='euler', on_pre='''
    C += alpha
    release = Vres * P
    Vres -= release
    Nt += k_NtV * release
    ''')
    syn.connect()
    syn.C = C0
    syn.Vres = V0
    syn.Nt = 0
    syn.Epsc = 0

    M = StateMonitor(syn, ['I_out'], record=True)
    M1 = StateMonitor(syn, ['C'], record=True)
    M2 = StateMonitor(syn, ['Vres'], record=True)
    M3 = StateMonitor(syn, ['Nt'], record=True)
    M4 = StateMonitor(syn, ['P'], record=True)



    spike_mon = SpikeMonitor(inp)

    run(duration)

    return M.t, M.I_out[0]


# find max current
def find_stationary_peaks_dynamic(t, current, relative_tol=0.01):

    peaks, _ = find_peaks(current, distance=50)

    first_peak=current[peaks[0]]

    if len(peaks) == 0:
        return [], np.nan

    last_peak_value = current[peaks[-1]]

    stationary = []
    for p in peaks:
        if abs(current[p] - last_peak_value) <= relative_tol * abs(last_peak_value):
            stationary.append(p)

    return stationary, last_peak_value, first_peak


# simulate
rates = np.arange(10,151,10)*Hz
stationary_Iout_values = []
first_peaks_values = []

for r in rates:

    print(f"Simulating {r/Hz} Hz ...")

    t, Iout = run_sim(r)

    peak_indices, last_peak_value, first_peak = find_stationary_peaks_dynamic(t, Iout)
    stationary_Iout_values.append(last_peak_value)
    first_peaks_values.append(first_peak)

    plt.figure(figsize=(10,4))
    plt.plot(t/ms, Iout/mA, label=f"I_out(t) at {r/Hz} Hz", linewidth=1.5)

    if len(peak_indices) > 0:
        plt.plot((t[peak_indices]/ms), (Iout[peak_indices]/mA),
                 'ro', label="max current")

    plt.title(f"I_out(t) at {r/Hz} Hz with marked max current (1% deviation)")
    plt.xlabel("Time (ms)")
    plt.ylabel("I_out (mA)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    rate_path = os.path.join(rate_dir, f"Iout_rate_{r/Hz}Hz.png")
    plt.savefig(rate_path, dpi=200)
    plt.close()   


plt.figure(figsize=(7,5))
plt.plot(rates/Hz, np.array(stationary_Iout_values)/np.array(first_peaks_values), marker='o', linewidth=2)
plt.title("Current of last peak / Current of first peak")
plt.xlabel("Inputrate (Hz)")
plt.ylabel("value")
plt.grid(True)

#plt.plot(rates, np.array(first_peak), marker='o', linewidth=2)


#plt.ylim(0, 0.5)

plt.tight_layout()

save_path = os.path.join(save_dir, "first_vs_last_current_per_rate.png")
plt.savefig(save_path, dpi=200)
plt.close()
