import streamlit as st
import neo.io
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import scipy as sp
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import elephant as el
import io
import os
import elephant as el
from math import floor
from scipy.signal import welch, find_peaks
from scipy import signal
from sonpy import lib as sonp

@st.cache
def upload_smr_file(f):
    
    filename = f.name
    f_bytes = io.BytesIO(f.read())
    
    with open(filename, 'wb') as out:
        out.write(f_bytes.read())
        
    return filename

@st.cache
def get_channel_list(FilePath):
    data = sonp.SonFile(sName=FilePath, bReadOnly=True)
    channel_list = [f'Channel {i + 1} ({str(data.ChannelType(i)).split(".")[-1]})' for i in range(data.MaxChannels())]
    return channel_list

@st.cache
def import_raw_smr(filename, channel_index):
    
    reader = neo.io.Spike2IO(filename)
    
    block = reader.read(lazy=False)[0]
    
    segments = block.segments[0]
    
    analogsignals = segments.analogsignals

    raw_data = np.array(analogsignals[channel_index],dtype='float64').transpose()[0]
    fs = float(analogsignals[channel_index].sampling_rate)
    t_start = float(analogsignals[channel_index].t_start)
    t_stop = float(analogsignals[channel_index].t_stop)
    return raw_data, fs, t_start, t_stop
    
@st.cache
def bandpass_filter(raw_data, lowpass_fs, highpass_fs, fs):
    return el.signal_processing.butter(raw_data, highpass_frequency=lowpass_fs, lowpass_frequency=highpass_fs, order=4, filter_function='filtfilt', sampling_frequency=fs, axis=- 1)

@st.cache
def peak_detection(filtered_data,inverted,threshold_slider):
    
    if inverted == True:
        threshold = threshold_slider*np.median(np.abs(filtered_data)/0.6745)
        peaks = sp.signal.find_peaks(filtered_data, height=threshold)
        
    else:
        threshold = -threshold_slider*np.median(np.abs(filtered_data)/0.6745)
        peaks = sp.signal.find_peaks(-filtered_data, height=-threshold)
                
    peaks = peaks[0]
        
    return peaks, threshold

@st.cache
def get_spikes(filtered_data, peak_indices, spike_window):
    
    peak_indices = peak_indices[(peak_indices > spike_window) & (peak_indices < peak_indices[-1] - spike_window)]
    spikes = np.empty([1, spike_window*2])
    for k in peak_indices:
        temp_spike = filtered_data[k - spike_window:k + spike_window]
        spikes = np.append(spikes, temp_spike.reshape(1, spike_window*2), axis=0)
    
    spikes = np.delete(spikes,0,axis=0)
    
    return peak_indices, spikes

@st.cache
def spike_sorting(spikes,clusters):
    
    pca = PCA(n_components = 2)
    features = pca.fit_transform(spikes)
    
    kmeans = KMeans(n_clusters=clusters, random_state=0).fit(features)
    labels = kmeans.labels_
    
    return features, labels

def main():
    
    FilePath = os.getcwd() + "/" + f
    st.write(FilePath)

    get_channel_list = get_channel_list(FilePath)
    
    channel = st.selectbox("Select Channel",get_channel_list)
    channel_index = get_channel_list.index(channel)

    raw_data_full, fs, t_start, t_stop = import_raw_smr(filename, channel_index)
    t_full = np.arange(t_start,t_stop,1/fs)

    tFrom_to_tTo = st.text_input("Enter tFrom to tTo in the following format: tFrom,tTo", str(0)+","+str(floor(t_stop)))
    st.write("tMax = ", round(t_stop,4))

    tFrom = float(tFrom_to_tTo.split(",")[0])
    tTo = float(tFrom_to_tTo.split(",")[1])

    t = t_full[(t_full >= tFrom) & (t_full <= tTo)] - tFrom
    raw_data = raw_data_full[(t_full >= tFrom) & (t_full <= tTo)]
    
    t_start = t[0]
    t_stop = t[-1]
    
    channel_id = select_channel + 1

    st.header('Spiketrain Analysis')

    bandpass_filter_input = st.text_input("Enter bandpass filter cutoffs in the following format: highpass,lowpass", str(300)+","+str(3000))

    highpass_fs = float(bandpass_filter_input.split(",")[0])
    lowpass_fs = float(bandpass_filter_input.split(",")[1])
    
    filtered_data = bandpass_filter(raw_data, highpass_fs, lowpass_fs, fs)
    
    threshold_slider = float(st.text_input('Enter threshold crossing factor k (k*MAD estimate of noise)',value=2))
    inverted = st.checkbox('Invert Peaks')
    
    peak_indices, threshold = peak_detection(filtered_data,inverted,threshold_slider)
    peak_values = np.array([filtered_data[j] for j in peak_indices])

    fig = go.Figure(data=go.Scatter(
        x = t, 
        y = filtered_data, 
        mode = 'lines',
        name='Filtered Data',
        line = dict(color='darkgreen'),
        showlegend=False))
    
    fig.add_trace(go.Scatter(
        x=peak_indices/fs,
        y=peak_values,
        mode='markers',
        marker = dict(color = 'red'),
        showlegend=False))
    
    fig.add_shape(type='line',
        x0=t_start,x1=t_stop,
        y0=threshold,y1=threshold,
        line = dict(color='black'),
        name='Threshold')

    fig.update_layout(
        title="Filtered MER Segment",
        title_x=0.5,
        xaxis_title="Time (s)",
        yaxis_title="Voltage (V)")
    
    fig_update = st.plotly_chart(fig)
    
    peak_indices, spikes = get_spikes(filtered_data, peak_indices, int(1*10**-3*fs))
    peak_values = np.array([filtered_data[j] for j in peak_indices])
    
    sorting_required = st.checkbox('Is spike sorting required?')

    if sorting_required == True:
        fig_update.empty()

        ## ADDED AUTOMATIC FUNCTIONALITY
        ss = []
        for c in range(2, 7):
            features_temp, label_temp = spike_sorting(spikes,c)
            ss.append(silhouette_score(features_temp,label_temp))
        ##
        
        clusters_auto = np.array(ss).argmax()+2

        clusters = st.selectbox(label='How many clusters are there?',options=range(2,7),index=range(2, 7).index(clusters_auto))    
        
        features, labels = spike_sorting(spikes,clusters)
            
        colour_list = list(['red','blue','orange','cyan','magenta','yellow'])
        colour_label = np.array(colour_list)[labels]
            
        fig2 = go.Figure(go.Scatter(
                x = features[:,0],
                y = features[:,1],
                mode = 'markers',
                marker = dict(color = colour_label)))

        fig2.update_layout(
                title="Low-dimensional Feature Space of Spike Waveforms",
                title_x=0.5,
                xaxis_title="Principal Component 1",
                yaxis_title="Principal Component 2")
            
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
        
        fig3.add_shape(type='line',
            x0=t_start,x1=t_stop,
            y0=threshold,y1=threshold,
            line = dict(color='black'),
            name='Threshold')
        
        fig3.update_layout(
            title="Filtered MER Segment with Sorted Spikes",
            title_x=0.5,
            xaxis_title="Time (s)",
            yaxis_title="Voltage (V)")

        st.plotly_chart(fig3)

        ## AUTOMATICALLY SELECT DESIRED CLUSTERS
        desired_spikes = []

        for d in range(0,clusters):
            desired_spikes.append(np.median(peak_values[colour_label == colour_list[d]]))

        desired_spikes = np.absolute(np.array(desired_spikes))

        desired_clusters = colour_list[desired_spikes.argmax()]
        ##
            
        desired_clusters = st.selectbox(label='Select Desired Cluster',options=colour_list[0:clusters],index=colour_list[0:clusters].index(desired_clusters))
            
        silhouette = silhouette_score(features,labels)
            
        peak_indices = np.array(peak_indices[colour_label == desired_clusters])
        spikes = spikes[colour_label == desired_clusters]
            
    else:
        desired_clusters = 'red'
        labels = np.zeros(len(peak_indices),dtype='int32')
        colour_label = np.array(list(['red']))[labels]
        silhouette = np.nan
        clusters = 1

    spiketrain = peak_indices/fs
    isi = el.statistics.isi(spiketrain)

    firing_rate = 1/isi.mean()
    snr_value = el.waveform_features.waveform_snr(spikes)
    percent_isi_violations = (sum(isi < 1/1000)/len(isi))*100

    st.subheader('Spiketrain Features')

    iqr = np.subtract(*np.percentile(isi, [75, 25]))
    
    bins = round((isi.max() - isi.min())/(2*iqr*len(isi)**(-1/3))) # The Freedman-Diaconis rule is very robust and works well in practice
    hist = np.histogram(isi, bins)
    
    isi_mode = hist[1][hist[0].argmax()]
    isi_mean = isi.mean()
    isi_std = isi.std()

    firing_rate = 1/isi_mean
    burst_index = isi_mean/isi_mode
    pause_index = isi.max()/isi.mean()

    pause_index = sum(isi[isi>3*isi_mean])/t_stop

    st.write(sum(isi>3*isi_mean))

    cv = isi_std/isi_mean


    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(label = "Firing Rate", value = round(firing_rate,3))
    
    with col2:
        st.metric(label = "Burst Index", value = round(burst_index,3))

    with col3:
        st.metric(label = "Pause Index", value = round(pause_index,3))

    with col4:
        st.metric(label = "Coefficient of Variation", value = round(cv,3))


    
    snr_value = el.waveform_features.waveform_snr(spikes)
    
    percent_isi_violations = (sum(isi < 1/1000)/len(isi))*100


    st.header('Power Spectral Density (LFP)')

    F = 10
    
    freqs, psd = welch(raw_data, nfft=F*fs, fs=fs, nperseg=fs)
    
    fig4 = go.Figure(data=go.Scatter(
            x = freqs, 
            y = psd,
            name='Power Spectral Density',
            line = dict(color='black'),
            showlegend=False))

    fig4.update_layout(xaxis_range=[0,100])

    fig4.update_layout(
        title="Power Spectral Density of MER Segment",
        title_x=0.5,
        xaxis_title="Frequency (Hz)",
        yaxis_title="Power Spectral Density (V^2 / Hz)")
    
    st.plotly_chart(fig4)
    
    freqs = freqs[0:100*F]
    psd = psd[0:100*F]
    
    beta_peaks_index = find_peaks(psd)[0]
    beta_peaks = psd[beta_peaks_index]
    beta_peaks_freqs = freqs[beta_peaks_index]
    
    low_beta_peak_logical = np.logical_and((beta_peaks_freqs) > 13, beta_peaks_freqs < 20)
    high_beta_peak_logical = np.logical_and((beta_peaks_freqs) > 21, beta_peaks_freqs < 30)
    
    if all(np.invert(low_beta_peak_logical)):
        low_beta_peak = max(psd[13*F:20*F])
    else:
        low_beta_peak = max(beta_peaks[low_beta_peak_logical])
    
    
    if all(np.invert(high_beta_peak_logical)):
        low_beta_peak = max(psd[21*F:30*F])
    else:
        high_beta_peak = max(beta_peaks[high_beta_peak_logical])
    
    low_beta = low_beta_peak/np.median(psd)
    high_beta = high_beta_peak/np.median(psd)

    st.subheader('LFP Features (Signal-to-noise as Power)')
    
    col11, col22 = st.columns(2)

    with col11:
        st.metric(label = "Low Beta (13-20 Hz)", value = round(float(low_beta),2))
    
    with col22:
        st.metric(label = "High Beta (21-30 Hz)", value = round(float(high_beta),2))

    st.header('Spike Shapes')


    resamp_freq = 1000

    spikes_resamp = []
    for x in range(len(spikes)): 

        spikes_resamp.append(signal.resample(spikes[x,:], resamp_freq))
                                
    spikes_resamp = np.array(spikes_resamp)

    t_resamp = np.linspace(0, 2, resamp_freq)

    av_spike = spikes.mean(0)
    av_spike_resamp = signal.resample(av_spike, resamp_freq)
    av_spike_resamp_gradient = np.gradient(av_spike_resamp, 1/resamp_freq)


    if av_spike_resamp_gradient.max() > av_spike_resamp_gradient.min():
        sw_loc = av_spike_resamp_gradient.argmax()
    else:
        sw_loc = av_spike_resamp_gradient.argmin()
        
    half_spike_width_loc = np.argwhere(np.diff(np.sign(av_spike_resamp - av_spike_resamp[sw_loc]))).flatten()
    half_spike_width_loc = np.array([half_spike_width_loc[0],half_spike_width_loc[-1]])

    half_spike_width = np.abs(half_spike_width_loc[0] - half_spike_width_loc[-1])/resamp_freq

    peak_latency = np.abs(av_spike_resamp.argmax() - av_spike_resamp.argmin())/resamp_freq

    if np.abs(av_spike_resamp.max()) > np.abs(av_spike_resamp.min()):
        spike_amplitude_ratio = np.abs(av_spike_resamp.max())/np.abs(av_spike_resamp.min())
    else:
        spike_amplitude_ratio = np.abs(av_spike_resamp.min()/av_spike_resamp.max())

    fig5 = go.Figure([
        go.Scatter(
            name='Average Spike',
            x=t_resamp,
            y=av_spike_resamp,
            mode='lines',
            line=dict(color='rgb(31, 119, 180)'),
        ),
        go.Scatter(
            name='Half Spike Width',
            x=t_resamp[half_spike_width_loc],
            y=av_spike_resamp[half_spike_width_loc],
            mode='lines',
            line=dict(color='red'),
        ),

        go.Scatter(
            name='Upper Bound',
            x=t_resamp,
            y=av_spike_resamp + 2*spikes_resamp.std(axis=0),
            mode='lines',
            marker=dict(color="#444"),
            line=dict(width=0),
            showlegend=False
        ),
        go.Scatter(
            name='Lower Bound',
            x=t_resamp,
            y=av_spike_resamp - 2*spikes_resamp.std(axis=0),
            marker=dict(color="#444"),
            line=dict(width=0),
            mode='lines',
            fillcolor='rgba(68, 68, 68, 0.3)',
            fill='tonexty',
            showlegend=False
        )
    ])
    
    fig5.update_layout(
        title="Spike shape of MER Segment",
        title_x=0.5,
        xaxis_title="Time (ms)",
        yaxis_title="Voltage (V)"
    )

    st.plotly_chart(fig5)


    col111, col222, col333 = st.columns(3)    

    with col111:
        st.metric(label = "Half Spike Width", value = round(half_spike_width,4))
    
    with col222:
        st.metric(label = "Peak Latency", value = round(peak_latency,4))

    with col333:
        st.metric(label = "Spike Amplitude Ratio", value = round(spike_amplitude_ratio,4))

    st.sidebar.subheader('Sorting Metrics')

    val1 = round(float(firing_rate),2)
    val2 = round(snr_value,2)
    val3 = round(percent_isi_violations,2)
    val4 = round(silhouette,2)
    val5 = channel_id
    val6 = highpass_fs
    val7 = lowpass_fs
    val8 = clusters
    val9 = inverted
    val10 = threshold_slider
    val11 = sorting_required
    
    name1 = 'Firing Rate (Hz)'
    name2 = 'SNR'
    name3 = 'ISI Violations (%)'
    name4 = 'Silhouette Score'
    name5 = 'Channel'
    name6 = 'Highpass Cutoff (Hz)'
    name7 = 'Lowpass Cutoff (Hz)'
    name8 = 'Number of Clusters'
    name9 = 'Invert Peaks?'
    name10 = 'Threshold'
    name11 = 'Spike Sorting Required?'
    
    st.sidebar.metric(label = name1, value = val1)
    st.sidebar.metric(label = name2, value = val2)
    st.sidebar.metric(label = name3, value = val3)
    st.sidebar.metric(label = name4, value = val4)

    st.sidebar.subheader('Sorting Settings')

    st.sidebar.write(name5, val5)
    st.sidebar.write(name6, val6)
    st.sidebar.write(name7, val7)
    st.sidebar.write(name8, val8)
    st.sidebar.write(name9, val9)
    st.sidebar.write(name10, val10)
    st.sidebar.write(name11, val11)

st.title('TNBSanalyzer - By Srdjan Sumarac')
    
f = st.file_uploader('Select smr file to upload','smr',False)
#st.write(os.getcwd())

if f is not None:
    filename = upload_smr_file(f)
    main()

    #st.write(os.listdir())
    
    if os.path.isfile(filename):
        os.remove(filename)   
