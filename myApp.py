import streamlit as st
st.set_page_config(page_title = "TNBS Spooky Spikes Online", page_icon = "tnbs_logo.png")

from PIL import Image
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
from astropy.timeseries import LombScargle
import math
from scipy.signal import welch, find_peaks,peak_widths
from sklearn.mixture import GaussianMixture
from scipy.stats import median_abs_deviation

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("Password incorrect")
        return False
    else:
        # Password correct.
        return True

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
    channel_list_adc = [match for match in channel_list if "Adc" in match]
    
    return channel_list_adc

@st.cache
def import_raw_smr(FilePath, channel_index):
    MyFile = sonp.SonFile(FilePath, True)
    
    dMaxTime = MyFile.ChannelMaxTime(channel_index)*MyFile.GetTimeBase()
    
    dPeriod = MyFile.ChannelDivide(channel_index)*MyFile.GetTimeBase()
    nPoints = floor(dMaxTime/dPeriod)
    
    fs = 1/dPeriod
    
    t = np.arange(0, nPoints*dPeriod, dPeriod)
    raw_data =  np.array(MyFile.ReadFloats(channel_index, nPoints, 0))
    return raw_data, t, fs, t[-1]

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

def spike_oscillations(raw_data, spiketrain,fs,lag_time,time_interval, to_plot):
    spiketrain = np.array(spiketrain, dtype='int')
    spiketrain_index = np.array(spiketrain, dtype='int')
    spike_data = np.zeros(raw_data.shape, dtype='int')
    spike_data[spiketrain] = 1
    
    spiketrain = spiketrain/fs
    
    t = np.arange(0,len(raw_data))/fs
    t_stop = t[-1]
    
    spiketrain_trimmed = spiketrain[(spiketrain>lag_time) & (spiketrain<t_stop - lag_time)]
    spiketrain_index_trimmed = spiketrain_index[(spiketrain>lag_time) & (spiketrain<t_stop - lag_time)]
    lag = math.floor(lag_time*fs)
    lead = lag
    time_interval = math.floor(time_interval*fs)
    bins = np.arange(0,lag,time_interval)
    
    
    # LAGGING
    
    spike_data_matrix_lag = []
    
    for s in spiketrain_index_trimmed:
        temp = spike_data[s - lag:s]
        spike_data_matrix_lag.append(temp)
    
    spike_data_matrix_lag = np.stack(spike_data_matrix_lag, axis=0)
    
    binned_spike_data_lag = []
    binned_spike_data_matrix_lag = []
    for r in range(len(spike_data_matrix_lag)):
        for b in bins:
            binned_spike_data_lag.append(sum(spike_data_matrix_lag[r][b:b+time_interval]))
            
        binned_spike_data_matrix_lag.append(np.array(binned_spike_data_lag))
        binned_spike_data_lag = []
    
    
    binned_spike_data_matrix_lag = np.stack(binned_spike_data_matrix_lag, axis=0)
    
    
    autocorr_lag = np.sum(binned_spike_data_matrix_lag, axis = 0)
    
    
    # LEADING
    
    spike_data_matrix_lead = []
    
    for s in spiketrain_index_trimmed:
        temp = spike_data[s:s+lead]
        spike_data_matrix_lead.append(temp)
    
    spike_data_matrix_lead = np.stack(spike_data_matrix_lead, axis=0)
    
    binned_spike_data_lead = []
    binned_spike_data_matrix_lead = []
    for r in range(len(spike_data_matrix_lead)):
        for b in bins:
            binned_spike_data_lead.append(sum(spike_data_matrix_lead[r][b:b+time_interval]))
            
        binned_spike_data_matrix_lead.append(np.array(binned_spike_data_lead))
        binned_spike_data_lead = []
    
    binned_spike_data_matrix_lead = np.stack(binned_spike_data_matrix_lead, axis=0)
    
    autocorr_lead = np.sum(binned_spike_data_matrix_lead, axis = 0)
    
    
    
    if float(fs).is_integer() == False:
        autocorr_lag = np.delete(autocorr_lag,-1)
        autocorr_lead = np.delete(autocorr_lead,-1)
    
    autocorr_lead[0] = np.mean([autocorr_lag[-1],autocorr_lead[1]])
    
    
    autocorr = np.concatenate([autocorr_lag, autocorr_lead])
    autocorr = (autocorr/len(spike_data_matrix_lead))/time_interval
    autocorr = autocorr/np.mean(autocorr)
    autocorr = autocorr - np.mean(autocorr)
    
    binned_time = np.linspace(-lag_time,lag_time,len(autocorr))
    
    #freqs_autocorr, psd_autocorr = welch(autocorr, fs = 1/0.01)
    freqs_autocorr, psd_autocorr = LombScargle(binned_time, autocorr).autopower()
    
    def find_nearest(array,value):
        idx = np.searchsorted(array, value, side="left")
        if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
            return idx
        else:
            return idx
    
    delta_peaks, _ = find_peaks(psd_autocorr[find_nearest(freqs_autocorr,0):find_nearest(freqs_autocorr,4)])
    theta_peaks, _ = find_peaks(psd_autocorr[find_nearest(freqs_autocorr,4):find_nearest(freqs_autocorr,8)])
    alpha_peaks, _ = find_peaks(psd_autocorr[find_nearest(freqs_autocorr,8):find_nearest(freqs_autocorr,12)])
    beta_peaks, _ = find_peaks(psd_autocorr[find_nearest(freqs_autocorr,12):find_nearest(freqs_autocorr,35)])
    gamma_peaks, _ = find_peaks(psd_autocorr[find_nearest(freqs_autocorr,35):find_nearest(freqs_autocorr,50)])
    
    if (len(delta_peaks) == 0) or (len(theta_peaks) == 0) or (len(alpha_peaks) == 0) or (len(beta_peaks) == 0) or (len(gamma_peaks) == 0):
        delta_freq = -100
        theta_freq = -100
        alpha_freq = -100
        beta_freq = -100
        gamma_freq = -100
        
        delta_power = -100
        theta_power = -100
        alpha_power = -100
        beta_power = -100
        gamma_power = -100
    
    else:
        theta_peaks = theta_peaks + find_nearest(freqs_autocorr,4)
        alpha_peaks  = alpha_peaks + find_nearest(freqs_autocorr,8)
        beta_peaks = beta_peaks + find_nearest(freqs_autocorr,12)
        gamma_peaks = gamma_peaks + find_nearest(freqs_autocorr,35)
        
        delta_peak_width = peak_widths(psd_autocorr, delta_peaks,rel_height=1)[0][psd_autocorr[delta_peaks].argmax()]
        theta_peak_width = peak_widths(psd_autocorr, theta_peaks,rel_height=1)[0][psd_autocorr[theta_peaks].argmax()]
        alpha_peak_width = peak_widths(psd_autocorr, alpha_peaks,rel_height=1)[0][psd_autocorr[alpha_peaks].argmax()]
        beta_peak_width = peak_widths(psd_autocorr, beta_peaks,rel_height=1)[0][psd_autocorr[beta_peaks].argmax()]
        gamma_peak_width = peak_widths(psd_autocorr, gamma_peaks,rel_height=1)[0][psd_autocorr[gamma_peaks].argmax()]
        
        delta_peak = psd_autocorr[delta_peaks[psd_autocorr[delta_peaks].argmax()]]
        delta_freq = freqs_autocorr[delta_peaks[psd_autocorr[delta_peaks].argmax()]]
        delta_power = sum(psd_autocorr[find_nearest(freqs_autocorr,delta_freq-delta_peak_width*0.1/2):find_nearest(freqs_autocorr,delta_freq+delta_peak_width*0.1/2)])
        
        theta_peak = psd_autocorr[theta_peaks[psd_autocorr[theta_peaks].argmax()]]
        theta_freq = freqs_autocorr[theta_peaks[psd_autocorr[theta_peaks].argmax()]]
        theta_power = sum(psd_autocorr[find_nearest(freqs_autocorr,theta_freq-theta_peak_width*0.1/2):find_nearest(freqs_autocorr,theta_freq+theta_peak_width*0.1/2)])
        
        alpha_peak = psd_autocorr[alpha_peaks[psd_autocorr[alpha_peaks].argmax()]]
        alpha_freq = freqs_autocorr[alpha_peaks[psd_autocorr[alpha_peaks].argmax()]]
        alpha_power = sum(psd_autocorr[find_nearest(freqs_autocorr,alpha_freq-alpha_peak_width*0.1/2):find_nearest(freqs_autocorr,alpha_freq+alpha_peak_width*0.1/2)])
        
        beta_peak = psd_autocorr[beta_peaks[psd_autocorr[beta_peaks].argmax()]]
        beta_freq = freqs_autocorr[beta_peaks[psd_autocorr[beta_peaks].argmax()]]
        beta_power = sum(psd_autocorr[find_nearest(freqs_autocorr,beta_freq-beta_peak_width*0.1/2):find_nearest(freqs_autocorr,beta_freq+beta_peak_width*0.1/2)])
        
        gamma_peak = psd_autocorr[gamma_peaks[psd_autocorr[gamma_peaks].argmax()]]
        gamma_freq = freqs_autocorr[gamma_peaks[psd_autocorr[gamma_peaks].argmax()]]
        gamma_power = sum(psd_autocorr[find_nearest(freqs_autocorr,gamma_freq-gamma_peak_width*0.1/2):find_nearest(freqs_autocorr,gamma_freq+gamma_peak_width*0.1/2)])
    
    
    psd_power = {'Band':['Delta [0-4Hz]', 'Theta [4-8Hz]', 'Alpha [8-12Hz]', 'Beta [12-35Hz]', 'Gamma [35-50Hz]'],'Frequency [Hz]':[delta_freq, theta_freq, alpha_freq, beta_freq, gamma_freq],'Power [dB]':[10*math.log10(delta_power), 10*math.log10(theta_power), 10*math.log10(alpha_power), 10*math.log10(beta_power), 10*math.log10(gamma_power)]}
    
    psd_power_df = pd.DataFrame(psd_power)
    
    return psd_power_df, freqs_autocorr, psd_autocorr, binned_time, autocorr

def main():
    FilePath = os.getcwd() + "/" + f.name

    channel_list = get_channel_list(FilePath)
    
    channel = st.selectbox("Select Channel",channel_list)
    select_channel = int(channel.split()[1])-1

    raw_data, t, fs, t_stop = import_raw_smr(filename, select_channel)

    
    
    t_start = t[0]
    t_stop = t[-1]
    
    channel_id = select_channel + 1

    
    tab1, tab2, = st.tabs(["Spiketrain Analysis", "Spike Shapes"])
    
    with tab1:
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
        
        sorting_required = st.checkbox('Spike Sorting')

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

        st.subheader('Quality Metrics')

        col111, col222, col333 = st.columns(3)

        with col111:
            st.metric(label = "Signal to Noise Ratio", value = round(snr_value,2))
        
        with col222:
            st.metric(label = "ISI Violations (%)", value = round(percent_isi_violations,2))

        with col333:
            st.metric(label = "Silhouette Score", value = round(silhouette,2))

        iqr = np.subtract(*np.percentile(isi, [75, 25]))
        
        bins = round((isi.max() - isi.min())/(2*iqr*len(isi)**(-1/3))) # The Freedman-Diaconis rule is very robust and works well in practice
        hist = np.histogram(isi, bins)
        
        isi_mode = hist[1][hist[0].argmax()]
        isi_mean = isi.mean()
        isi_std = isi.std()

        firing_rate = 1/isi_mean

        X = np.ravel(isi).reshape(-1, 1)
        M_best = GaussianMixture(n_components=2, covariance_type="spherical").fit(X)
        means = M_best.means_
        burst_index = np.max(means) / np.min(means)
        cv = median_abs_deviation(isi) / np.median(isi)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(label = "Firing Rate", value = round(firing_rate,3))
        
        with col2:
            st.metric(label = "Burst Index", value = round(burst_index,3))

        with col3:
            st.metric(label = "Coefficient of Variation", value = round(cv,3))


        
        snr_value = el.waveform_features.waveform_snr(spikes)
        
        percent_isi_violations = (sum(isi < 1/1000)/len(isi))*100

        st.header('Oscillations')

        oscillations_required = st.checkbox('Spiketrain Oscillation Analysis')

        if oscillations_required == True:
        
            lag_time = 0.5
            time_interval = 0.01
            psd_power_df, freqs_autocorr, psd_autocorr, binned_time, autocorr = spike_oscillations(raw_data = raw_data, spiketrain = peak_indices,fs = fs,lag_time = 0.5,time_interval = 0.01, to_plot = True)

            fig4_pre= go.Figure(data=go.Scatter(
                    x = binned_time, 
                    y = autocorr,
                    name='Autocorrelation',
                    line = dict(color='black'),
                    showlegend=False))

            fig4_pre.update_layout(xaxis_range=[-0.5,0.5])

            fig4_pre.update_layout(
                title="Spiketrain Autocorrelation Function",
                title_x=0.5,
                xaxis_title="Lag (s)",
                yaxis_title="Autocorrelation")

            st.plotly_chart(fig4_pre)
            
            fig4 = go.Figure(data=go.Scatter(
                    x = freqs_autocorr, 
                    y = psd_autocorr,
                    name='LombScargle',
                    line = dict(color='black'),
                    showlegend=False))

            fig4.update_layout(xaxis_range=[0,50])

            fig4.update_layout(
                title="Lomb-Scargle Periodogram",
                title_x=0.5,
                xaxis_title="Frequency (Hz)",
                yaxis_title="Power Spectral Density (V^2 / Hz)")
            
            st.plotly_chart(fig4)

            st.subheader('Spiketrain Oscillation Features')

            st.table(psd_power_df)


    with tab2:
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

    # st.sidebar.subheader('Sorting Metrics')

    # val1 = round(float(firing_rate),2)
    # val2 = round(snr_value,2)
    # val3 = round(percent_isi_violations,2)
    # val4 = round(silhouette,2)
    # val5 = channel_id
    # val6 = highpass_fs
    # val7 = lowpass_fs
    # val8 = clusters
    # val9 = inverted
    # val10 = threshold_slider
    # val11 = sorting_required
    
    # name1 = 'Firing Rate (Hz)'
    # name2 = 'SNR'
    # name3 = 'ISI Violations (%)'
    # name4 = 'Silhouette Score'
    # name5 = 'Channel'
    # name6 = 'Highpass Cutoff (Hz)'
    # name7 = 'Lowpass Cutoff (Hz)'
    # name8 = 'Number of Clusters'
    # name9 = 'Invert Peaks?'
    # name10 = 'Threshold'
    # name11 = 'Spike Sorting Required?'
    
    # st.sidebar.metric(label = name1, value = val1)
    # st.sidebar.metric(label = name2, value = val2)
    # st.sidebar.metric(label = name3, value = val3)
    # st.sidebar.metric(label = name4, value = val4)

    # st.sidebar.subheader('Sorting Settings')

    # st.sidebar.write(name5, val5)
    # st.sidebar.write(name6, val6)
    # st.sidebar.write(name7, val7)
    # st.sidebar.write(name8, val8)
    # st.sidebar.write(name9, val9)
    # st.sidebar.write(name10, val10)
    # st.sidebar.write(name11, val11)

st.title('TNBS Spooky Spikes Online')

st.info('By Srdjan Sumarac & Luka Zivkovic')
st.image(Image.open('tnbs_logo.png'),width=150)

if check_password():
    
    f = st.file_uploader('Select smr file to upload','smr',False)
    #st.write(os.getcwd())

    if f is not None:
        
        filename = upload_smr_file(f)
        main()

        #st.write(os.listdir())
        
        if os.path.isfile(filename):
            os.remove(filename)   
