import streamlit as st
import neo.io
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import scipy as sp
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.express as px
from sklearn.metrics import silhouette_score
import openpyxl
import elephant as el
import io
import os

##################################################################################################################

st.title('Neural Segments App')

f = st.sidebar.file_uploader('Select smr file to upload','smr',False)

g = io.BytesIO(f.read())

temploc = f.name

with open(temploc, 'wb') as out:
    out.write(g.read())

##################################################################################################################

@st.cache
def import_raw_smr(filename):
    
    reader = neo.io.Spike2IO(filename)
    
    block = reader.read(lazy=False)[0]
    
    segments = block.segments[0]
    
    analogsignals = segments.analogsignals

    channel_id = []
    for i in range(len(analogsignals)):
        channel_id.append(analogsignals[i].annotations['channel_id'])
    
    if (0 in channel_id) == True:
        idx = channel_id.index(0)
        raw_data = np.array(analogsignals[idx],dtype='float64').transpose()
        return raw_data[0], float(analogsignals[idx].sampling_rate), float(analogsignals[idx].t_start), float(analogsignals[idx].t_stop),1
    
    elif (1 in channel_id) == True:
        idx = channel_id.index(1)
        raw_data = np.array(analogsignals[idx],dtype='float64').transpose()
        return raw_data[0], float(analogsignals[idx].sampling_rate), float(analogsignals[idx].t_start), float(analogsignals[idx].t_stop),2

raw_data, fs, t_start, t_stop, channel = import_raw_smr(f.name)
t = np.arange(t_start,t_stop,1/fs)

lowpass_left, highpass_right = st.beta_columns(2)
lowpass_fs = lowpass_left.number_input('Lowpass Frequency Cutoff',0,1500,300,100)
highpass_fs = highpass_right.number_input('Highpass Frequency Cutoff',2000,6000,3000,100)

filtered_data = el.signal_processing.butter(raw_data, highpass_frequency=lowpass_fs, lowpass_frequency=highpass_fs, order=4, filter_function='filtfilt', sampling_frequency=fs, axis=- 1)

##################################################################################################################

left_inverted, dummy1,dummy2 = st.beta_columns(3)
inverted = left_inverted.checkbox('Invert Peaks')
threshold_slider = st.slider('Median Absolute Deviations (Noise Estimate)',1,20,4)

@st.cache
def peak_detection(filtered_data,inverted,threshold_slider):
    
    if inverted == True:
        threshold = threshold_slider*np.median(np.abs(filtered_data)/0.6745)
        peaks = sp.signal.find_peaks(filtered_data, height=threshold)
        
    else:
        threshold = -threshold_slider*np.median(np.abs(filtered_data)/0.6745)
        peaks = sp.signal.find_peaks(-filtered_data, height=-threshold)
        
    return peaks[0], threshold

peak_indices, threshold = peak_detection(filtered_data,inverted,threshold_slider)

##################################################################################################################

fig = go.Figure(data=go.Scatter(
    x = t, 
    y = filtered_data, 
    mode = 'lines',
    name='Filtered Data',
    line = dict(color='darkgreen'),
    showlegend=False))

fig.add_trace(go.Scatter(
    x=peak_indices/fs,
    y=[filtered_data[j] for j in peak_indices],
    mode='markers',
    marker = dict(color = 'red'),
    showlegend=False))

fig.add_shape(type='line',
              x0=t_start,x1=t_stop,
              y0=threshold,y1=threshold,
              line = dict(color='black'),
              name='Threshold')

#fig.update_layout(xaxis=dict(rangeslider=dict(visible=True),type="-"))

st.plotly_chart(fig)

##################################################################################################################

@st.cache
def get_spikes(peak_indices, spike_window_ms):
    
    spike_window = int(spike_window_ms*10**-3*fs)
    
    peak_indices = peak_indices[(peak_indices > spike_window) & (peak_indices < peak_indices[-1] - spike_window)]
    spikes = np.empty([1, spike_window*2])
    for k in peak_indices:
        temp_spike = filtered_data[k - spike_window:k + spike_window]
        spikes = np.append(spikes, temp_spike.reshape(1, spike_window*2), axis=0)
    
    spikes = np.delete(spikes,0,axis=0)
    
    return peak_indices, spikes

peak_indices, spikes = get_spikes(peak_indices, 1)

##################################################################################################################

st.subheader('Spike Sorting')

sorting_required = st.checkbox('Is spike sorting required?')

@st.cache
def spike_sorting(spikes,clusters):
    
    pca = PCA(n_components = 2)
    features = pca.fit_transform(spikes)
    
    kmeans = KMeans(n_clusters=clusters, random_state=0).fit(features)
    labels = kmeans.labels_
    
    return features, labels
    
if sorting_required == True:
        
    clusters = st.selectbox('How many clusters are there?',range(2,7))    
    
    features, labels = spike_sorting(spikes,clusters)
        
    colour_list = list(['red','blue','orange','cyan','magenta','yellow'])
    colour_label = np.array(colour_list)[labels]
        
    fig2 = go.Figure(go.Scatter(
        x = features[:,0],
        y = features[:,1],
        mode = 'markers',
        marker = dict(color = colour_label)))
        
    st.plotly_chart(fig2)
        
    fig3 = go.Figure(go.Scatter(
        x = t, 
        y = filtered_data, 
        mode = 'lines',
        name='Filtered Data',
        line = dict(color='darkgreen'),
        showlegend=False))
        
    fig3.add_trace(go.Scatter(
        x=peak_indices/fs,
        y=[filtered_data[j] for j in peak_indices],
        mode='markers',
        marker = dict(color = colour_label),
        showlegend=False))
        
    st.plotly_chart(fig3)
        
    desired_clusters = st.selectbox('Select Desired Cluster',colour_list[0:clusters])
        
    silhouette = silhouette_score(features,labels)
        
    peak_indices = np.array(peak_indices[colour_label == desired_clusters])
    spikes = spikes[colour_label == desired_clusters]
        
else:
    desired_clusters = 'red'
    labels = np.zeros(len(peak_indices),dtype='int32')
    colour_label = np.array(list(['red']))[labels]
    silhouette = np.nan
    clusters = 1

##################################################################################################################

spiketrain = peak_indices/fs

@st.cache    
def snr(spikes):
    
    S_avg = np.matrix(spikes).mean(axis=0)
    S_avg_i = []

    for i in range(len(spikes)):
        spike_temp = np.matrix(spikes[i,:])
        S_avg_i.append(S_avg - spike_temp)
    
    peak_to_peak = S_avg.max() - S_avg.min()

    S_avg_i = np.array(S_avg_i).ravel()
    resid = np.std(S_avg_i)

    snr = peak_to_peak/(5*resid)
        
    return snr

snr_value = float(snr(spikes))
num_spikes = int(len(spiketrain))
firing_rate = float(len(spiketrain)/(t_stop - t_start))

##################################################################################################################

@st.cache
def isi_func(spiketrain):
    isi = []
    for i in range(0,len(spiketrain)-1):
        isi.append(spiketrain[i+1] - spiketrain[i])
        
    return np.array(isi)

isi = isi_func(spiketrain)
isi_mean = isi.mean()

n_bins = round((isi.max() - isi.min())/(10**-3))
hist_values = np.histogram(isi,bins=n_bins)
hist_x = hist_values[0]
hist_y = hist_values[1]
isi_mode = hist_y[np.where(hist_x == hist_x.max())]

# df_isi = pd.DataFrame(isi)
# df_isi.columns = ['ISI'+ ' of ' + desired_clusters]
# st.subheader('Inter-spike interval')
# fig5 = px.histogram(df_isi,nbins=n_bins)
# st.plotly_chart(fig5)

percent_isi_violations = (sum(isi < 1/1000)/len(isi))*100

##################################################################################################################

@st.cache
def burst_calc_srdjan(spiketrain):

    burst_logical = []
    prev_burst = False
    inter_bi = []
    intra_bi = []
    
    for i in range(0,len(spiketrain)-1):
        
        if (isi[i] <= 0.5*isi_mean) and prev_burst == False:
            prev_burst = True 
            burst_logical.append(True)
            inter_bi.append(spiketrain[i])
            intra_bi.append(spiketrain[i])    
       
        elif (isi[i] <= 0.5*isi_mean) and prev_burst == True:
            prev_burst = True 
            burst_logical.append(True)
            intra_bi.append(spiketrain[i])
            
        elif (isi[i] > 0.5*isi_mean) and prev_burst == False:
            prev_burst = False 
            burst_logical.append(False)
            
        elif (isi[i] > 0.5*isi_mean) and prev_burst == True:
            prev_burst = False 
            burst_logical.append(True)
            intra_bi.append(spiketrain[i])
    

    
    return intra_bi, inter_bi, burst_logical

#intra_bi, inter_bi, burst_logical = burst_calc_srdjan(spiketrain)

@st.cache
def burst_calc_luka(peaks, fs_data01, burst_th, spike_freq, t_start, t_stop, num_spikes):
    bursts_list = []
    ibi_list = []
    prev_burst = False 
    intra_burst_count = 0
    for x in range(0,len(peaks)-1): 
        t1 = peaks[x]
        t2 = peaks[x+1]
        delta_t = t2-t1
        inst_freq = 1 / delta_t        
        if inst_freq >= spike_freq*burst_th and prev_burst == False:
            bursts_list.append(peaks[x])
            ibi = peaks[x] - peaks[x-1]
            ibi_list.append(ibi)
            prev_burst = True
            intra_burst_count = intra_burst_count + 1
        elif inst_freq >= spike_freq*burst_th and prev_burst == True:
            prev_burst = True
            intra_burst_count = intra_burst_count + 1
        elif inst_freq < spike_freq*burst_th and prev_burst == True:
            prev_burst = False  
        elif inst_freq < spike_freq*burst_th and prev_burst == False:
            prev_burst = False
    
    bursts = np.array(bursts_list)
    n_bursts = len(bursts)
    burst_freq = n_bursts/(t_stop - t_start)
    average_ibi = np.mean(ibi_list)
    average_ibi = average_ibi/fs_data01
    
    return burst_freq, spike_freq/burst_freq, intra_burst_count

burst_freq, burst_index, intra_burst_count = burst_calc_luka(spiketrain, fs, 1.5, firing_rate, t_start, t_stop, num_spikes)

# st.subheader('Bursting Spikes')

# fig3 = go.Figure(go.Scatter(
#     x = t, 
#     y = filtered_data, 
#     mode = 'lines',
#     name='Filtered Data',
#     line = dict(color='darkgreen'),
#     showlegend=False))

# peak_indices = np.delete(peak_indices,0,axis=0)

# fig3.add_trace(go.Scatter(
#     x=peak_indices[burst_logical]/fs,
#     y=[filtered_data[j] for j in peak_indices[burst_logical]],
#     mode='markers',
#     marker = dict(color = 'black'),
#     showlegend=False))

# st.plotly_chart(fig3)

##################################################################################################################

st.sidebar.subheader('Spiking Features')


val1 = num_spikes
val2 = round(float(firing_rate),2)
val3 = round(snr_value,2)
val4 = round(percent_isi_violations,2)
val5 = round(silhouette,2)
val6 = round(burst_freq,2)
val7 = round(isi_mean/isi_mode[0],2)
val8 = round(intra_burst_count/num_spikes*100,2)
val9 = channel
val10 = lowpass_fs
val11 = highpass_fs
val12 = clusters
val13 = inverted
val14 = desired_clusters
val15 = round(t_stop - t_start,2)

name1 = 'Number of Spikes'
name2 = 'Firing Rate (Hz)'
name3 = 'SNR'
name4 = 'ISI Violations (%)'
name5 = 'Silhouette Score'
name6 = 'Mean Burst Frequency (Hz)'
name7 = 'Burst Index'
name8 = 'Spikes in Burst (%)'
name9 = 'Channel'
name10 = 'Lowpass Cutoff (Hz)'
name11 = 'Highpass Cutoff (Hz)'
name12 = 'Number of Clusters'
name13 = 'Invert Peaks?'
name14 = 'Desired Cluster'
name15 = 'Segment Duration (s)'

st.sidebar.write(name1, val1)
st.sidebar.write(name2, val2)
st.sidebar.write(name3, val3)
st.sidebar.write(name4, val4)
st.sidebar.write(name5, val5)
st.sidebar.write(name6, val6)
st.sidebar.write(name7, val7)
st.sidebar.write(name8, val8)
st.sidebar.write(name9, val9)
st.sidebar.write(name10, val10)
st.sidebar.write(name11, val11)
st.sidebar.write(name12, val12)
st.sidebar.write(name13, val13)
st.sidebar.write(name14, val14)
st.sidebar.write(name15, val15)

os.remove(temploc)
