import numpy
import scipy.io as io
import pywt # Wavelet
from scipy.fft import fft, fftfreq #Fourier
import matplotlib.pyplot as plt
from PIL import Image
from skimage.transform import resize
from tqdm import tqdm # for print progress 
import timeit
from scipy.signal import butter, lfilter # Butterworth Bandpass

#---------------------------------- explicit function to compute column wise sum
def colSum (arr):
    n = len(arr[0])
    m = len(arr)
    L = []
    for i in range(n):
        su = 0;
        for j in range(m):
            su += arr[j][i]
        L.append(su)
    return L

#---------------------------- for a 3D array Remove all values out for [0,1] values (have to check this more) 
def Cut_Values_Outside_Bandwidth_3D(array):
    for i in (range(0,array.shape[0])):
        for j in range(0,array.shape[1]):
            for k in range(0,array.shape[2]):
                if(array[i,j,k]>=1):
                    array[i,j,k] = 1
                if(array[i,j,k]<=0):
                    array[i,j,k] = 0
    return array

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------- The following functions resize Arrays/Images ----------------------------------------------#
#---------------------------- Resize a 2D Array to the disered size
def Resize_Img(image, width=None, height=None):
    dim=None
    (h,w) = image.shape[:2]
    if width is None and height is None:
        return image
    elif width is not None and height is not None:
        dim = (width, height)
    elif width is None:
        r = height/float(h)
        dim = (int(w*r), height)
    else:
        r = width / float(w)
        dim = (width, int(h*r))
    resized = resize(image, dim)
    return resized

#---------------------------- Resize a list of 2D Arrays Images to the disered size
def Resize_list_of_Imges(arrays,width, height):
    resized_arrays = []
    for img in (arrays):
        resized_arrays.append(Resize_Img(img,width,height))
    resized_arrays = numpy.array(resized_arrays)
    return resized_arrays

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------- The following functions readS and transform/process the burst data ----------------------------------------------#

#---------------------------- This function reads the bursts data, and group's them. 
def Read_Raw_Bursts(from_burst,to_burest,path):
    time_signal = []
    for i in range (from_burst,to_burest):
        trackdata = io.loadmat(path+numpy.str(i)+".mat")
        burst_data = trackdata['recorder_data'][0];
        time_signal.extend(burst_data)
    time_signal = numpy.array(time_signal)
    return time_signal

#---------------------------- This function removes from a "signal" all the frequencies higher than threshold_frequency_up AND lower than threshold_frequency_down
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

#---------------------------- This function reads the bursts data, denoise them by removings frequencies higher than threshold_frequency_up AND lower than threshold_frequency_down 
#---------------------------- return each burst separetely as a list of bursts
def Read_FrequencyDenoise_Bursts_Split(from_burst,to_burest,path, threshold_frequency_down, threshold_frequency_up):
    sample_Rate = 100000000  # Hertz <-- (100MHz)
    denoised_time_signal = []
    for i in tqdm(range(from_burst,to_burest)):
        trackdata = io.loadmat(path+numpy.str(i)+".mat")
        burst_data = trackdata['recorder_data'][0];
        denoised_burst_data = butter_bandpass_filter(burst_data, threshold_frequency_down, threshold_frequency_up, sample_Rate, order=9)
        denoised_time_signal.append(denoised_burst_data)   
    denoised_time_signal = numpy.array(denoised_time_signal)
    return denoised_time_signal

#---------------------------- This functions normalize a signal in time domain based on the max value of the signal, so we have a signal between [0.0,1.0] 
#---------------------------- return each burst separetely as a list of bursts
def Normalize_Signal_Split(signal):
    max_val = numpy.max(signal)
    for i in range(0,len(signal)):
        for j in range(0,len(signal[0])):
            signal[i,j] = signal[i,j]/max_val  # devide all the value with the max value 
    return signal

#---------------------------- This functions reads a group bursts denoise them, and normalize them between [0,1] 
#---------------------------- return each burst separetely as a list of bursts
def Read_FrequencyDenoise_Normalize_Bursts_Split(from_burst,to_burest,path, threshold_frequency_down, threshold_frequency_up):
    frequency_denoised_signal = Read_FrequencyDenoise_Bursts_Split(from_burst,to_burest,path, threshold_frequency_down, threshold_frequency_up) #------------ Read & Denoise Signals 
    normalized_frequency_denoised_signal = Normalize_Signal_Split(frequency_denoised_signal)
    return normalized_frequency_denoised_signal

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------- The following functions Plot -----------------------------------------------------------------------#

#---------------------------- This function we plot the time signal with the spikes
def Plot_Time_Signal(signal):
    sample_Rate = 100000000  # Hertz <-- (100MHz)
    period = 1/sample_Rate # in seconds
    period = period * 1000000 # in microseconds (μs)
    time_axis = []
    for step in range (0,len(signal)):
        time_axis.append(step*period) 
    plt.figure(figsize=(30,7))
    plt.rc('font',size=18)
    plt.rc('axes',titlesize=18)
    plt.rc('axes',labelsize=18)
    plt.rc('legend',fontsize=18)
    plt.rc('figure',titlesize=18)
    plt.plot(time_axis,signal, alpha=0.8, color='blue',label="Time data")
    plt.legend()
    plt.grid()
    plt.title("Time Domain")
    plt.ylabel("Voltage - milliVolt (mV)")
    plt.xlabel("Time - microseconds (μs)")
    plt.legend()
    plt.show()
    
#---------------------------- This function we create and plot the fft domain of a signal with sampling 100MHz
def Plot_FFT_Signal(signal,thresholds):
    n = len(signal) # Number of samples 
    sample_Rate = 100000000  # in Hertz <-- (100MHz)
    sample_Rate = sample_Rate / 1000000 # in MegaHz
    threshold_frequency_down = 1 # in MegaHz
    threshold_frequency_up = 20 # in MegaHz
    frequencies = fftfreq(n, 1/sample_Rate)
    down_threshold = 0
    while (frequencies[down_threshold]<threshold_frequency_down):
        down_threshold +=1
    up_threshold = 0
    while (frequencies[up_threshold]<threshold_frequency_up):
        up_threshold +=1
    spectrum = numpy.fft.fft(signal) 
    plt.figure(figsize=(15,5))
    plt.plot(frequencies[1:n//2],numpy.abs(spectrum[1:n//2]), alpha=0.8, color='blue',label="Domain data")
    max_val = numpy.max(numpy.abs(spectrum[1:n//2]))
    if(thresholds==True):
        plt.plot([frequencies[down_threshold],frequencies[down_threshold]], [0,max_val], linewidth=5, color='green',label="Low Threshold 1MHz")
        plt.plot([frequencies[up_threshold],frequencies[up_threshold]], [0,max_val], linewidth=5, color='red',label="High Threshold 20MHz")
    plt.legend()
    plt.grid()
    plt.title("Frequency Data")
    plt.ylabel("Amplitude")
    plt.xlabel("Frequency - MegaHertz (MHz)")
    plt.legend()
    plt.show()

    
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------- The following functions find local max -----------------------------------------------------------------------#
    
#---------------------------------- we calculate the location of the spikes/pikes
def Spikes_Location(data,threshold_value,range_value): 
    #start = timeit.default_timer() 
    potential_spikes_location = []         # We keep the locations where we have high value from the threshold_value, in other words our potential local extremes.
    for i in range(0,len(data)):          
        if data[i]>=threshold_value:
            potential_spikes_location.append(i)
    #stop = timeit.default_timer()
    #print('Time: %.3f' % (stop - start),'for potential_spikes_location') 
    
    #start = timeit.default_timer()
    spikes_location = []    
    while (len(potential_spikes_location)>0):
        potential_max_spikes_location = 0
        potential_max_value_spikes_location = 0
        for location in potential_spikes_location:
            if (data[location]>potential_max_value_spikes_location):
                potential_max_spikes_location = location
                potential_max_value_spikes_location = data[location]
        
        flag = True
        for max_location in spikes_location:
            if(abs(max_location-potential_max_spikes_location) <= 2*range_value):  # Range of data we want to find the local max is 2 because is left and right
                flag = False
                break
        if flag:
            spikes_location.append(potential_max_spikes_location)
        potential_spikes_location.remove(potential_max_spikes_location)    
    #stop = timeit.default_timer()
    #print('Time: %.3f' % (stop - start),'for spikes_location')   
    return (spikes_location)

#---------------------------- This functions finds spikes and amplitudes
def Spikes_Location_and_Amplitude(signal,threshold_activity,range_removal):
    spikes_list_location = Spikes_Location(signal,threshold_activity,range_removal)
    spikes_list_location = sorted(spikes_list_location) #order of spikes in time
    spikes_list_amplitude = []  
    for j in spikes_list_location:   
        spikes_list_amplitude.append(signal[j])  
    spikes_list_amplitude = numpy.array(spikes_list_amplitude)
    return(spikes_list_location,spikes_list_amplitude)

#---------------------------- This function we plot the time signal with the spikes
def Plot_Time_Signal_with_Spikes(signal,spikes_list_location,spikes_list_amplitude):
    sample_Rate = 100000000  # Hertz <-- (100MHz)
    period = 1/sample_Rate # in seconds
    period = period * 1000000 # in microseconds (μs)
    time_axis = []
    for step in range (0,len(signal)):
        time_axis.append(step*period) 
    spikes_list_location_in_time = []
    for i in range (0,len(spikes_list_location)):
        spikes_list_location_in_time.append(spikes_list_location[i]*period)    
    plt.figure(figsize=(30,7))
    plt.rc('font',size=18)
    plt.rc('axes',titlesize=18)
    plt.rc('axes',labelsize=18)
    plt.rc('legend',fontsize=18)
    plt.rc('figure',titlesize=18)
    plt.plot(time_axis,signal, alpha=0.8, color='blue',label="Time data")
    plt.plot(spikes_list_location_in_time,spikes_list_amplitude,'o',alpha=0.8, color='red', label="Picks - Windows with Emission Activity")
    plt.legend()
    plt.grid()
    plt.title("Time Domain")
    plt.ylabel("Voltage - milliVolt (mV)")
    plt.xlabel("Time - microseconds (μs)")
    plt.legend()
    plt.show()
    
#---------------------------- This function we create and plot a window of spike in Time in FFT and wavelet
def Plot_Frame_Time_FFT_CTW (spikes_list_location,signal,frame_size,widths,bandwidth,center_frequency):
    for i in range (1,len(spikes_list_location)):  
        data_frame = signal[(spikes_list_location[i]):(spikes_list_location[i]+frame_size)] 
        sample_Rate = 100000000  # in Hertz <-- (100MHz)
        period = 1/sample_Rate # in seconds
        period = period * 1000000 # in microseconds (μs)
        sample_Rate = sample_Rate / 1000000 # in MegaHz
        #----------------------------------------------------------------- Plots
        plt.figure(1, figsize=(34,10))
        plt.rc('font',size=18)
        plt.rc('axes',titlesize=18)
        plt.rc('axes',labelsize=18)
        plt.rc('legend',fontsize=18)
        plt.rc('figure',titlesize=18)
        plt.clf()
        plt.subplot(231) #---------------------------------- Signal in Time
        time_axis = []
        for step in range (0,len(data_frame)):
            time_axis.append(step*period) 
        plt.plot(time_axis,data_frame, alpha=0.8, color='blue',label="Time data")
        plt.legend()
        plt.grid()
        plt.title("Frame - Time Domain")
        plt.ylabel("Voltage - milliVolt (mV)")
        plt.xlabel("Time - microseconds (μs)")
        plt.legend()
        plt.show
        plt.subplot(232) #---------------------------------- FFT Signal
        n = len(data_frame) # Number of samples 
        frequencies = fftfreq(n, 1/sample_Rate)
        spectrum = numpy.fft.fft(data_frame) 
        plt.plot(frequencies[1:n//2],numpy.abs(spectrum[1:n//2]), alpha=0.8, color='blue',label="Domain data")
        plt.legend()
        plt.grid()
        plt.title("Frame - Frequency Domain")
        plt.ylabel("Amplitude")
        plt.xlabel("Frequency - MegaHertz (MHz)")
        plt.legend()
        plt.show
        ax = plt.subplot(233)
        #---------------------------------- Wavelet Transform
        cwtmatr, freqs = pywt.cwt(data=data_frame, scales=widths, wavelet='cmor'+str(bandwidth)+'-'+str(center_frequency), sampling_period=1/sample_Rate, method='fft') #'mexh' 'morl' 
        cwt_values = numpy.abs(cwtmatr)
        ax = plt.gca()
        plt.title('Frame - Wavelet Domain'+numpy.str(i))
        im = ax.imshow(cwt_values, cmap='seismic', aspect='auto')  # seismic gray
        plt.colorbar(im)
        x_values = numpy.arange(0,len(data_frame),250)
        y_values = numpy.arange(0,len(freqs),15)
        x_labels = []
        for i in x_values:
            x_labels.append(numpy.str("%.2f" % round(i*period, 4)))
        y_labels = []
        for i in y_values:
            y_labels.append(numpy.str("%.2f" % round(freqs[i], 4)))
        ax.set_xticks(x_values)
        ax.set_xticklabels(x_labels)
        ax.set_yticks(y_values)
        ax.set_yticklabels( y_labels )
        plt.ylabel("Frequency - MegaHertz (MHz)")
        plt.xlabel("Time - microseconds (μs)")
        plt.show()
               
#---------------------------- These functions we plot 1 CWT wavelet AND it is called only from the Plot_CTW_Rows_From_Saved_Wavelets. 
def Sub_Plot_Wavelet (ax,i,wavelet,freqs):
    plt.title(numpy.str(i))
    sample_Rate = 100000000  # in Hertz <-- (100MHz)
    period = 1/sample_Rate # in seconds
    period = period * 1000000 # in microseconds (μs)
    sample_Rate = sample_Rate / 1000000 # in MegaHz
    #ax = plt.gca()
    #im = ax.imshow(wavelet, cmap='seismic', aspect='auto')  # seismic gray
    im = ax.imshow(wavelet, cmap='seismic', aspect='auto',vmin=0,vmax=1)  # seismic gray
    plt.colorbar(im)
    #---------------------------- Axis
    x_values = numpy.arange(0,wavelet.shape[1],wavelet.shape[1]/4)
    y_values = numpy.arange(0,len(freqs),15)
    x_labels = []
    for i in x_values:
        x_labels.append(numpy.str("%.2f" % round(i*period, 4)))
    y_labels = []
    for i in y_values:
        y_labels.append(numpy.str("%.2f" % round(freqs[i], 4)))
    ax.set_xticks(x_values)
    ax.set_xticklabels(x_labels)
    ax.set_yticks(y_values)
    ax.set_yticklabels( y_labels )
    plt.ylabel("Frequency - MegaHertz (MHz)")
    plt.xlabel("Time - microseconds (μs)")

#---------------------------- This function we plot rows of saved wavelets 
def Plot_Rows_of_Wavelets(wavelet_data,freqs,rows_num):
    wavelets_num = len(wavelet_data)
    i = 0
    while (i < len(wavelet_data)):
        plt.figure(1, figsize=(40,6))
        plt.clf()
        for j in range(0,rows_num):
            if(i+j<wavelets_num):
                ax = plt.subplot(201+rows_num*10+j) #---------------------------------- PLOT
                Sub_Plot_Wavelet (ax,i+j,wavelet_data[i+j],freqs)
                plt.show
            else:
                break
        plt.show()
        i +=rows_num