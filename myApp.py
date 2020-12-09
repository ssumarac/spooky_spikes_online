import streamlit as stlit
import neo.io
import os
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import scipy as sp
import spikeinterface.extractors as se
import spikeinterface.toolkit as st
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.express as px

stlit.title('Neural Segments App')

def file_selector(folder_path='.'):
    filenames = []

    for file in os.listdir(folder_path):
        if file.endswith('.smr'):
          filenames.append(file)
        
    filenames.sort()
 
    selected_filename = stlit.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)

sidebar_filename = file_selector()

stlit.subheader('Peak Detection')

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
        return np.array(analogsignals[idx],dtype='float64').transpose(), int(analogsignals[idx].sampling_rate), float(analogsignals[idx].t_start), float(analogsignals[idx].t_stop)
    
    elif (1 in channel_id) == True:
        idx = channel_id.index(1)
        return np.array(analogsignals[idx],dtype='float64').transpose(), int(analogsignals[idx].sampling_rate), float(analogsignals[idx].t_start), float(analogsignals[idx].t_stop)

raw_data, fs, t_start, t_stop = import_raw_smr(sidebar_filename)
t = np.arange(t_start,t_stop,1/fs)

def filtering():

    lowpass_left, highpass_right = stlit.beta_columns(2)
    lowpass_fs = lowpass_left.number_input('Lowpass Frequency Cutoff',0,1500,300,100)
    highpass_fs = highpass_right.number_input('Highpass Frequency Cutoff',2000,6000,3000,100)
    
    recording = se.NumpyRecordingExtractor(timeseries=raw_data, geom=None, sampling_frequency=fs)
    recording_bp = st.preprocessing.bandpass_filter(recording, freq_min=lowpass_fs, freq_max=highpass_fs)
    
    
    return recording_bp.get_traces().flatten();

filtered_data = filtering()

def peak_detection(filtered_data):

    left_inverted, dummy1,dummy2 = stlit.beta_columns(3)
    inverted = left_inverted.checkbox('Invert Peaks')
    threshold_slider = stlit.slider('Median Absolute Deviations (Noise Estimate)',1,20,4)
    
    if inverted == True:
        threshold = threshold_slider*np.median(np.abs(filtered_data)/0.6745)
        peaks = sp.signal.find_peaks(filtered_data, height=threshold)
        
    else:
        threshold = -threshold_slider*np.median(np.abs(filtered_data)/0.6745)
        peaks = sp.signal.find_peaks(-filtered_data, height=-threshold)
        
    return peaks[0], threshold

peak_indices, threshold = peak_detection(filtered_data)

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

stlit.plotly_chart(fig)

def get_spikes(peak_indices, spike_window_ms):
    
    spike_window = int(spike_window_ms*10**-3*int(fs))
    
    peak_indices = peak_indices[(peak_indices > spike_window) & (peak_indices < peak_indices[-1] - spike_window)]
    spikes = np.empty([1, spike_window*2])
    for k in peak_indices:
        temp_spike = filtered_data[k - spike_window:k + spike_window]
        spikes = np.append(spikes, temp_spike.reshape(1, spike_window*2), axis=0)
    
    spikes = np.delete(spikes,0,axis=0)
    
    return peak_indices, spikes

peak_indices, spikes = get_spikes(peak_indices, 1)

stlit.subheader('Spike Sorting')

def spike_sorting():

    sorting_required = stlit.checkbox('Is spike sorting required?')
    
    if sorting_required == True:
    
        
        pca = PCA(n_components=2) 
        features = pca.fit_transform(spikes)
        #total_var = pca.explained_variance_ratio_.sum() * 100
    
        clusters = stlit.selectbox('How many clusters are there?',range(2,7))
        
        #spectral_model = SpectralClustering(n_clusters = clusters) 
        kmeans = KMeans(n_clusters=clusters, random_state=0).fit(features)
          
        #labels = spectral_model.fit_predict(df_features) 
        labels = kmeans.labels_
        
        colour_list = list(['red','blue','orange','cyan','magenta','yellow'])
        colour_label = np.array(colour_list)[labels]
        
        fig2 = go.Figure(go.Scatter(
            x = features[:,0],
            y = features[:,1],
            mode = 'markers',
            marker = dict(color = colour_label)))
        
        stlit.plotly_chart(fig2)
        
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
        
        stlit.plotly_chart(fig3)
        
        desired_clusters = stlit.selectbox('Select Desired Cluster',colour_list[0:clusters])
        
        return np.array(peak_indices[colour_label == desired_clusters]), spikes[colour_label == desired_clusters],desired_clusters
        
    else:
        desired_clusters = 'red'
        labels = np.zeros(len(peak_indices),dtype='int32')
        colour_label = np.array(list(['red']))[labels]
        
        return np.array(peak_indices), spikes, desired_clusters
        
peak_indices, spikes, desired_clusters = spike_sorting()

spiketrain = peak_indices/fs
    
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


def isi_func(spiketrain):
    isi = []
    for i in range(0,len(spiketrain)-1):
        isi.append(spiketrain[i+1] - spiketrain[i])
        
    return np.array(isi)

isi = isi_func(spiketrain)
isi_mean = isi.mean()

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


fig3 = go.Figure(go.Scatter(
    x = t, 
    y = filtered_data, 
    mode = 'lines',
    name='Filtered Data',
    line = dict(color='darkgreen'),
    showlegend=False))

peak_indices = np.delete(peak_indices,0,axis=0)

fig3.add_trace(go.Scatter(
    x=peak_indices[burst_logical]/fs,
    y=[filtered_data[j] for j in peak_indices[burst_logical]],
    mode='markers',
    marker = dict(color = 'black'),
    showlegend=False))

stlit.plotly_chart(fig3)

stlit.sidebar.subheader('Spiking Features')

stlit.sidebar.write('Number of Spikes: ',num_spikes)
stlit.sidebar.write('Firing Rate (Hz): ', round(float(firing_rate),2))
stlit.sidebar.write('SNR'': ',round(snr_value,2))
stlit.sidebar.write('Burst Ratio',round(len(intra_bi)/num_spikes,2))
stlit.sidebar.write('Inter-Burst Rate (Hz)',round(1/isi_func(inter_bi).mean(),2))
stlit.sidebar.write('Intra-Burst Rate (Hz)',round(1/isi_func(intra_bi).mean(),2))



df_isi = pd.DataFrame(isi)
df_isi.columns = ['ISI'+ ' of ' + desired_clusters]

stlit.subheader('Inter-spike interval')
fig5 = px.histogram(df_isi)
stlit.plotly_chart(fig5)




# df_ibi_inter = pd.DataFrame(inter_bi)
# df_ibi_inter.columns = ['IBI_Inter'+ ' of ' + desired_clusters]

# stlit.subheader('Inter-burst interval')
# fig6 = px.histogram(df_isi)
# stlit.plotly_chart(fig6)




# df_ibi_intra = pd.DataFrame(intra_bi)
# df_ibi_intra.columns = ['IBI_Intra'+ ' of ' + desired_clusters]

# stlit.subheader('Intra-burst interval')
# fig7 = px.histogram(df_isi)
# stlit.plotly_chart(fig7)




# df_statistics = pd.DataFrame([num_spikes,
#                               snr_value,
#                               firing_rate,
#                               ])
    
# df_statistics.columns = [desired_clusters]
# df_statistics.index = ['Number of Spikes',
#                        'SNR',
#                        'Firing Rate', 
#                        ]

# stlit.dataframe(df_statistics)

