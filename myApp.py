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
import elephant as el
from quantities import s, Hz
import plotly.express as px

stlit.title('Pre-Processing App')

sidebar_path = stlit.sidebar.text_input('Path to .smr files','/Users/srdjan/GoogleDrive/Toronto/Segmented_data/SMR_files')
smr_files = []

for file in os.listdir(sidebar_path):
    if file.endswith('.smr'):
        smr_files.append(file)
        
smr_files.sort()

file_numbers = list(map(str,list(range(1,len(smr_files)+1))))

numbered_smr_files =[]

for x in range(len(smr_files)):
    numbered_smr_files.append(file_numbers[x] + ': ' + smr_files[x])

sidebar_filename_numbered = stlit.sidebar.selectbox('Select a file',numbered_smr_files)
sidebar_filename = smr_files[numbered_smr_files.index(sidebar_filename_numbered)]

stlit.subheader('Peak Detection')

def import_raw_smr(filename, path):
    
    os.chdir(path)
    
    reader = neo.io.Spike2IO(filename)

    block = reader.read(lazy=False)[0]
    
    segments = block.segments[0]
    
    analogsignals = segments.analogsignals
    
    t_start = segments.t_start
    t_stop = segments.t_stop

    for i in range(len(analogsignals)):
        annotations = analogsignals[i].annotations
        
        if annotations['channel_id'] == 0 or 1:
            return analogsignals[i], analogsignals[i].sampling_rate, t_start, t_stop


raw_data, fs, t_start, t_stop = import_raw_smr(sidebar_filename,sidebar_path)

raw_data = np.array(raw_data,dtype='float64').transpose()
t = np.arange(t_start,t_stop,1/fs)
fs = int(fs)

lowpass_left, highpass_right = stlit.beta_columns(2)
lowpass_fs = lowpass_left.number_input('Lowpass Frequency Cutoff',0,1500,300,100)
highpass_fs = highpass_right.number_input('Highpass Frequency Cutoff',2000,6000,3000,100)

recording = se.NumpyRecordingExtractor(timeseries=raw_data, geom=None, sampling_frequency=fs)
recording_bp = st.preprocessing.bandpass_filter(recording, freq_min=lowpass_fs, freq_max=highpass_fs)
filtered_data = recording_bp.get_traces().flatten();

left_inverted, dummy1,dummy2 = stlit.beta_columns(3)
inverted = left_inverted.checkbox('Invert Peaks')
threshold_slider = stlit.slider('Median Absolute Deviations (Noise Estimate)',1,20,4)

if inverted == True:
    threshold = threshold_slider*np.median(np.abs(filtered_data)/0.6745)
    peaks = sp.signal.find_peaks(filtered_data, height=threshold)
    
else:
    threshold = -threshold_slider*np.median(np.abs(filtered_data)/0.6745)
    peaks = sp.signal.find_peaks(-filtered_data, height=-threshold)
    
peak_indices = peaks[0]

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
              x0=float(t_start),x1=float(t_stop),
              y0=threshold,y1=threshold,
              line = dict(color='black'),
              name='Threshold')

#fig.update_layout(xaxis=dict(rangeslider=dict(visible=True),type="-"))

stlit.plotly_chart(fig)

spike_window = int( 1*10**-3*int(fs))
peak_indices = peak_indices[(peak_indices > spike_window) & (peak_indices < peak_indices[-1] - spike_window)]
spikes = np.empty([1, spike_window*2])
for k in peak_indices:
    temp_spike = filtered_data[k - spike_window:k + spike_window]
    spikes = np.append(spikes, temp_spike.reshape(1, spike_window*2), axis=0)

spikes = np.delete(spikes,0,axis=0)

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

pca = PCA(n_components=2) 
features = pca.fit_transform(spikes)
total_var = pca.explained_variance_ratio_.sum() * 100


stlit.subheader('Spike Sorting')

sorting_required = stlit.checkbox('Is spike sorting required?')

if sorting_required == True:

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
    
else:
    desired_clusters = 'red'
    labels = np.zeros(len(peak_indices),dtype='int32')
    colour_label = np.array(list(['red']))[labels]
    
stlit.subheader('Spiking Features')
    
spiketrain = np.array(peak_indices[colour_label == desired_clusters])/fs * s
        
snr_list = float(snr(spikes[colour_label == desired_clusters]))
num_spikes = int(len(spiketrain))
#firing_rate = float(len(spiketrain)/(t_stop - t_start))
firing_rate = el.statistics.mean_firing_rate(spiketrain,t_start,t_stop)

isi_array = el.statistics.isi(spiketrain)
percent_isi_violations = (sum(isi_array < 1/1000)/len(isi_array))*100

dispersion_index = np.std(isi_array)**2/np.mean(isi_array)
cv = el.statistics.cv(spiketrain)
burst_index = float(np.mean(isi_array)/sp.stats.mode(isi_array,axis=None)[0])
asymmetry_index = 1/burst_index

# burst_array = []
# mean_isi = float(np.mean(isi_array))

# for b in isi_array:
#     burst_array.append

df_statistics = pd.DataFrame([num_spikes,
                              snr_list,
                              float(firing_rate),
                              percent_isi_violations,
                              float(dispersion_index),
                              cv,
                              burst_index,
                              asymmetry_index
                              ])
    
df_statistics.columns = [desired_clusters]
df_statistics.index = ['Number of Spikes',
                       'SNR','Firing Rate', 
                       '% of ISI Violations',
                       'Dispersion Index',
                       'Coefficient of Variation',
                       'Burst Index',
                       'Asymmetry Index'
                       ]

stlit.dataframe(df_statistics)

df_isi = pd.DataFrame(isi_array)
df_isi.columns = ['ISI'+ ' of ' + desired_clusters]

fig5 = px.histogram(df_isi)
                     
stlit.plotly_chart(fig5)
   



    