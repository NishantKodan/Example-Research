import numpy as np
import time
import matplotlib.pyplot as plt 
from scipy.io import loadmat 



def ACF(time_step, lags, time_start, time_end, time_array, length_array, PLOT=False):
    """
    Calculates the autocorrelation of a time-dependent series and optionally plots the results.

    Args:
        time_step (float): The time step for evenly spaced time values.
        lags (int): The number of lags to calculate the autocorrelation for.
        time_start (float): The starting time offset for filtering data.
        time_end (float): The ending time offset for filtering data.
        time_array (numpy.ndarray): The array of time values.
        length_array (numpy.ndarray): The array of corresponding length values.
        PLOT (bool, optional): True to plot the autocorrelation, False otherwise. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - numpy.ndarray: The array of time shifts (in minutes) corresponding to the autocorrelation values.
            - numpy.ndarray: The array of autocorrelation values.
            - float: The 95% confidence interval for the autocorrelation.

    ACF(time_step= 1, lags=10, time_start=10, time_end=100, time_array=T, length_array=L, PLOT=False)

    """

    start = time.time()

    # Filter data within the specified time offset:
    valid_indices = np.where((time_array > time_start) & (time_array <= time_end))[0]
    x = time_array[valid_indices]
    y = length_array[valid_indices]

    # Create evenly spaced time values for interpolation:
    value_x = np.arange(x[0], x[-1], time_step)

    # Interpolate length values at evenly spaced times:
    value_y = np.interp(value_x, x, y)

    # Calculate autocorrelation for each lag:
    autocorr = [np.corrcoef(value_y, value_y)[0, 1]]  # Correlation at lag 0
    shifts = [0]
    for shift in range(1, lags):
        correlation = np.corrcoef(value_y[:-shift], value_y[shift:])[0, 1]
        autocorr.append(correlation)
        shifts.append(shift * time_step)

    shifts = np.array(shifts)
    autocorr = np.array(autocorr)

    # Calculate 95% confidence interval:
    ci = 1.96 / np.sqrt(len(value_y))

    # Plot the autocorrelation if PLOT is True:
    if PLOT:
        plt.figure()
        plt.plot(shifts, autocorr)
        plt.axhline(y=ci, linestyle='--', color='red', label='95% Confidence Interval')
        plt.axhline(y=-ci, linestyle='--', color='red')
        plt.xlabel('Lag(min)')
        plt.ylabel('ACF')
        plt.legend() 
        plt.plot()

    end = time.time()
    # print('Run time', end - start, 'seconds')

    return shifts, autocorr, ci


def passive_solution(kplus_passive, kdiss, Ntot, V, delt, t_end, m0, offset=False):

    """
    Outputs the active assembly driven Nucleolus size at different times, where two nucleoli are growing at the same rate in a shared pool of nucleolar proteins.

    Args:
        kplus_passive(float): The rate of assembly for passive model.
        kdiss (float): Dissociation constant of condensation reaction.
        Ntot (int): Total number of FIB-1 molecules in the Nucleus.
        V (float): Nuclear volume.
        delt(float): Time interval at which Nucleolus size is predicted.
        t_end (float): End time of the predicted calculation.
        m0 (float): Size of the nucleolus droplet(measured in FIB-1 molecules) at starting time.
        offset (bool, optional): Offsets the nucleolus size data to zero at rRNA transcription start time.


    Returns:
        tuple: A tuple containing:
            - numpy.ndarray: The array of times at which Nucleolus size is predicted.
            - numpy.ndarray: The array of Nucleolus size prediction at respective times.


    a, b = passive_solution(kplus_passive=13.6, kdiss=55, Ntot=8307, V=93.7, delt=1e-3, t_end=100, m0=1, offset=False)

    """

    # Set up time array
    t = np.arange(0, t_end + delt, delt)
    m1 = np.zeros(len(t))  # nucleolus size
    m1[0] = m0  # initial size

    # Calculate size prediction
    for i in range(len(t) - 1):
        m1[i+1] = ((Ntot-kdiss*V)/2)*(1-np.exp(-2*kplus_passive*t[i]/V))

    t_pred = t

    # Offsets size data 
    if offset:
        M_pred = m1 - m0 #Substracting the initial size from the prediction
    else:
        M_pred = m1

    return t_pred, M_pred



def active_solution(kplus_active, kdiss, Ntot, V, delt, t_end,t0, m0, offset=False):

    """
    Outputs the active assembly driven Nucleolus size at different times, where two nucleoli are growing at the same rate in a shared pool of nucleolar proteins.

    Args:
        kplus_active(float): The rate of assembly for active model (this rate constant also include initiation and elongation rate constants).
        kdiss (float): Dissociation constant of condensation reaction.
        Ntot (int): Total number of FIB-1 molecules in the Nucleus.
        V (float): Nuclear volume.
        delt(float): Time interval at which Nucleolus size is predicted.
        t_end (float): End time of the predicted calculation.
        t0 (float): Starting time of rRNA Transcription. 
        m0 (float): Size of the nucleolus droplet(measured in FIB-1 molecules) at start of rRNA transcription.
        offset (bool, optional): Offsets the nucleolus size data to zero at rRNA transcription start time.


    Returns:
        tuple: A tuple containing:
            - numpy.ndarray: The array of times at which Nucleolus size is predicted.
            - numpy.ndarray: The array of Nucleolus size prediction at respective times.


    a, b = active_solution(kplus_active=2.02, kdiss=55, Ntot=8307, V=93.7, delt=1e-3, t_end=100,t0=1, m0=1, offset=False)
    """

    # Set up time array
    t = np.arange(0, t_end + delt, delt)
    m1 = np.zeros(len(t))  # nucleolus size
    m1[0] = m0  # initial size

    # Calculate size prediction
    for i in range(len(t) - 1):
        if t[i] < t0:
            m1[i + 1] = m0
        else:
            m1[i + 1] = m0 * np.exp(-(2 * kplus_active / (3 * V)) * ((t[i] - t0) ** 3)) + (Ntot - kdiss * V) / 2 * (1 - np.exp(-(2 * kplus_active / (3 * V)) * ((t[i] - t0) ** 3)))

    t_pred = t

    # Offsets size data 
    if offset:
        M_pred = m1 - m0 #Substracting the initial size from the prediction
    else:
        M_pred = m1

    return t_pred, M_pred



def negative_log_likelihood(M, kplus_active, kdiss, Ntot, V, t0, m0):

    """
    Calculates the negative log-likelihood for the given parameters(t0=transcription start time) of our model function, for the experimental data of
    a single cell stage and cell lineage.

    Args:
        M (numpy.ndarray): Raw data matrix which contains individual time and length arrays of nucleolus size evolution.
        kplus_active(float): The rate of assembly for active model (this rate constant also include initiation and elongation rate constants).
        kdiss (float): Dissociation constant of condensation reaction.
        Ntot (int): Total number of FIB-1 molecules in the Nucleus.
        V (int): Nuclear volume
        t0 (float): Starting time of rRNA Transcription. 
        m0 (float): Size of the nucleolus droplet(measured in FIB-1 molecules) at start of rRNA transcription.


    Returns:
        float: Negative log-likelihood value.


    negative_log_likelihood(ab8_mat['AB8_traj_init'][0], kplus_active=2.02, kdiss=55, Ntot=19000, V=214, t0=-0.4, m0=256)

    """

    P = len(M)  # Number of data sets

    # Using params, compute one theoretical trajectory
    tpred, Mpred = active_solution(kplus_active, kdiss, Ntot, V, delt=0.001, t_end=25,t0=t0, m0=m0, offset=False)

    sum_val = 0
    tot_pts = 0

    for j in range(P):
        current = M[j]
        O = current.shape[0]
        Mexp = current[:, 1]  # Experimental data
        Texp = current[:, 0]  # Experimental time
        
        tot_pts += O

        #Correct time trajectories that begin before the average
        if Texp[0] < 0:
            Texp -= Texp[0]

        # Access correct elements from Mpred
        Mpred_order = []
        for idx in range(len(Texp)):
            Mpred_order.append(Mpred[np.where(tpred==Texp[idx])])  # Find closest value in tpred

        Mpred_order = np.array(Mpred_order)
        for i in range(O):
            sum_val += (Mexp[i] - Mpred_order[i]) ** 2

        # if j//10==0:
        #plt.plot(Texp,Mexp,label='Exp')
        #plt.plot(Texp,Mpred_order,color='yellow',label='Pred')
        #plt.legend()

    L = -0.5 * (tot_pts - 1) * np.log(np.real(sum_val))
    

    return L


def get_mean_traj(all_times,all_lengths,time_interval,standard_deviation=True):
    
    """
    Calculates the mean and standard deviation/standard error of mean for all the given lengths and times.

    Args:
        all_times (list): List of all the time arrays.
        all_lengths (list): List of all the length arrays.
        time_interval(float): Time interval between the predicted values.
        standard_deviation (bool, optional): Gives the standard deviation from the mean for all trajectories, if "False" gives us the standard error of mean.


    Returns:
        tuple: A tuple containing:
            - list: The list of times at which mean length is measured.
            - list: The list of mean lengths.
            - list: The list of STDs/SEMs.

    """

    # Calculate global min and max
    global_min = 100
    for i in range(len(all_times)):
        if min(all_times[i]) < global_min:
            global_min = min(all_times[i])

    global_max = 0
    for i in range(len(all_times)):
        if max(all_times[i]) > global_max:
            global_max = max(all_times[i])

    tpoints = np.arange(global_min, global_max + 0.5, time_interval)

    lengths = []
    times = []
    errors = []
    for t in tpoints:
        lens = []
        for i in range(len(all_times)):
            if t in all_times[i]:
                var = all_lengths[i][np.where(all_times[i] == t)]
                if len(var) > 0:
                    var = all_lengths[i][np.where(all_times[i] == t)][0]
                lens.append(var)
                
        if len(lens) > 1:
            lengths.append(np.mean(lens))
            if standard_deviation:
                errors.append(np.std(lens))
            else:
                errors.append(np.std(lens)/np.sqrt(len(lens)))  #/np.sqrt(len(lens))
            times.append(t)

        else:     
            lengths.append(np.mean(lens))
            errors.append(0)
            times.append(t)

    return times,lengths,errors

def two_filament_gillespie(model = 'passive',total_steps = 3e6,Tmax = 100):
    
    """
    Calculates the simulated trajectory of two filament growing in a limited and shared pool of monomers.

    Args:
        model (str): Either 'active' model or 'passive' model.
        total_steps (float/int): Total number of steps to take in the simulation.
        Tmax (int): Maximum time of the simulation.


    Returns:
        tuple: A tuple containing:
            - numpy.ndarray: T, Times at which lengths are predicted
            - numpy.ndarray: L_1, Simulated lengths of the first filament 
            - numpy.ndarray: L_2, Simulated lengths of the second filament

            
    two_filament_gillespie(model = 'passive',total_steps = 3e6,Tmax = 100)
    """
    if model == 'passive':
        N = 8307
        kplus_passive = 13.6
        kminus = 55 * kplus_passive
        Tmax = Tmax  # minutes
        V = 93.7

        num_steps = int(total_steps)

        T = np.zeros(num_steps)
        L_1 = np.zeros(num_steps)
        L_2 = np.zeros(num_steps)

        tcount = 1
        l1 = 1
        l2 = 1
        monomers = N

        for i in range(num_steps):
            L_1[i] = l1
            L_2[i] = l2
            T[i] = tcount

            k1 = kplus_passive * (N - l1 - l2) / V  # rate of attachment of filament 1
            k2 = kminus if l1 != 1 else 0  # rate of detachment of filament 1
            k3 = kplus_passive * (N - l1 - l2) / V  # rate of attachment of filament 2
            k4 = kminus if l2 != 1 else 0  # rate of detachment of filament 2

            k0 = k1 + k2 + k3 + k4

            tau = np.random.exponential(scale=1 / k0)

            tcount += tau

            rand = np.random.uniform(0, 1)

            if 0 < rand * k0 < k1:
                l1 += 1
                monomers -= 1
            elif k1 < rand * k0 < k1 + k2:
                l1 -= 1
                monomers += 1
            elif k1 + k2 < rand * k0 < k1 + k2 + k3:
                l2 += 1
                monomers -= 1
            else:
                l2 -= 1
                monomers += 1
                
            if T[i] >= Tmax:
                L_1 = L_1[:i+1]  
                L_2 = L_2[:i+1]  
                T = T[:i+1]  
                break
        
        return T, L_1, L_2

    else:
        N = 8307
        kplus_active = 2
        kminus = 55 * kplus_active
        Tmax = Tmax  # minutes
        V = 93.7

        num_steps = int(total_steps)  # Number of steps

        T = np.zeros(num_steps)
        L_1 = np.zeros(num_steps)
        L_2 = np.zeros(num_steps)

        tcount = 1
        l1 = 1
        l2 = 1
        monomers = N

        for i in range(num_steps):
            L_1[i] = l1
            L_2[i] = l2
            T[i] = tcount


            k1 = (tcount**2)* kplus_active * (N - l1-l2) / V  # rate of attachment of filament 1
            k2 = (tcount**2)*kplus_active *55 if l1 != 1 else 0  # rate of detachment of filament 1
            k3 = (tcount**2)* kplus_active * (N - l1-l2) / V   # rate of attachment of filament 2
            k4 = (tcount**2)*kplus_active *55 if l2 != 1 else 0  # rate of detachment of filament 2

            k0 = k1 + k2 + k3 + k4

            tau = np.random.exponential(scale=1 / k0)

            tcount += tau

            rand = np.random.uniform(0, 1)

            if 0 < rand * k0 < k1:
                l1 += 1
                monomers -= 1
            elif k1 < rand * k0 < k1 + k2:
                l1 -= 1
                monomers += 1
            elif k1 + k2 < rand * k0 < k1 + k2 + k3:
                l2 += 1
                monomers -= 1
            else:
                l2 -= 1
                monomers += 1

            if T[i] >= Tmax:
                L_1 = L_1[:i+1]  
                L_2 = L_2[:i+1]  
                T = T[:i+1]  
                break


        return T, L_1, L_2
    
    return []