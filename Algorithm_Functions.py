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
