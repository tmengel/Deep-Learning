#!/usr/bin/env python3
"""Utility functions for the final project."""
import os
import warnings

try:
    import pandas as pd
    import numpy as np
    import uproot
    import tensorflow as tf
    import tensorflow.keras as keras
    from tensorflow.keras import models, layers

except ImportError:
    warnings.warn("You do not have pandas, numpy, or uproot installed. "
                  "You will not be able to use the functions in utils.py.")
    
##################################### Data Functions #######################################

def GetData(filename, treename="timing"):
    '''
    Returns TFile as a pandas dataframe
    '''
    file = uproot.open(filename) # open file
    tree = file[treename] # get tree
    npdf = tree.arrays(library="np") # convert to numpy array
    df =  pd.DataFrame(npdf, columns=npdf.keys()) # convert to pandas dataframe
    df = df[df["trace"].apply(lambda x: x.shape[0] == 300)].reset_index(drop=True) # remove traces with wrong shape
    return df

def GetTraces(df):
    '''
    Returns traces as a numpy array
    '''
    values = df["trace"].values # get traces
    traces = np.zeros((values.shape[0], 300)) # initialize array
    for i in range(values.shape[0]): # loop over traces
        trace = np.array(values[i]).reshape(300, 1) # reshape trace
        traces[i][:] = trace[:, 0] # add trace to array
    return traces

def GetPhases(df):
    '''
    Returns phases as a numpy array
    '''
    phases = df["phase"].values # get phases
    phase = np.zeros((phases.shape[0], 1)) # initialize array
    for i in range(phases.shape[0]): # loop over phases
        if phases[i] > 0: # check if phase is valid
            phase[i] = phases[i] # add phase to array
        else: # if phase is invalid
            phase[i] = 0.0 # set phase to 0
    return phase

def OneHotEncodePileup(pileup):
    '''
    Returns one hot encoded pileup
    '''
    pileup_one_hot = np.zeros((pileup.shape[0], 2)) # initialize array
    for i in range(pileup.shape[0]): # loop over pileup
        if pileup[i] == 0: # check if pileup is 0
            pileup_one_hot[i][0] = 0
            pileup_one_hot[i][1] = 1
        else: # if pileup is not 0
            pileup_one_hot[i][0] = 1
            pileup_one_hot[i][1] = 0
    return pileup_one_hot # return one hot encoded pileup

def LoadData(filename):
    '''
    Loads data from file returns traces, phases, and one hot encoded pileup
    '''
    df = GetData(filename) # get data
    traces = GetTraces(df) # get traces
    phases = GetPhases(df) # get phases
    pileup_one_hot = OneHotEncodePileup(phases) # get one hot encoded pileup
    return traces, phases, pileup_one_hot
    
def CreateMockData(filename, pileup_split=0.5, phase_min=0, phase_max=100, amplitude_min=0.5, amplitude_max=1.5):
    '''
    Creates mock data for testing from non pileup data
    '''
    df = GetData(filename) # get data
    no_pileup = df["trace"].values[df["pileup"].values == False] # get non pileup traces
    
    n_samples = 2*no_pileup.shape[0] # get number of samples
           
    rand_phase_shifts = np.random.uniform(phase_min, phase_max, n_samples) # get random phase shifts
    rand_amplitude_shifts = np.random.uniform(amplitude_min, amplitude_max, n_samples) # get random amplitude shifts
    rand_trace_idx = np.random.randint(0, no_pileup.shape[0], n_samples) # get random trace indices
    
    phases = np.zeros((n_samples, 1)) # initialize array
    amplitudes = np.zeros((n_samples, 1)) # initialize array
    traces_convoluted = np.zeros((n_samples, 1, 300)) # initialize array
    traces_truth = np.zeros((n_samples, 2, 300)) # initialize array
    
    no_pileup = np.concatenate([no_pileup, no_pileup], axis=0) # double non pileup traces
    np.random.shuffle(no_pileup) # shuffle non pileup traces
    
    n_pileup_count = 0 # initialize pileup count
    n_no_pileup_count = 0 # initialize non pileup count
    for i in range(n_samples):
        traces_truth[i][0][:] = no_pileup[i][:300]
        # generate rand number from 0 to 1
        rand = np.random.uniform(0, 1)
        if rand < pileup_split: # pileup
            traces_truth[i][1][:] = np.roll(no_pileup[rand_trace_idx[i]][:300], int(rand_phase_shifts[i]))*rand_amplitude_shifts[i]
            traces_truth[i][1][:int(rand_phase_shifts[i])] = no_pileup[rand_trace_idx[i]][:int(rand_phase_shifts[i])]*rand_amplitude_shifts[i]
            phases[i] = rand_phase_shifts[i]
            amplitudes[i] = rand_amplitude_shifts[i]
            n_pileup_count += 1
        else: # no pileup
            traces_truth[i][1][:] = np.zeros_like(no_pileup[rand_trace_idx[i]][:300])
            phases[i] = 0.0
            amplitudes[i] = 0.0
            n_no_pileup_count += 1
        
        traces_convoluted[i][0][:] = traces_truth[i][0][:] + traces_truth[i][1][:]
        # renormalize
        norm = np.max(traces_convoluted[i][0][:])
        traces_convoluted[i][0][:] = traces_convoluted[i][0][:]/norm
    
    print("Created {} samples: {} % pileup, {} % no pileup".format(n_samples, n_pileup_count/n_samples*100, n_no_pileup_count/n_samples*100))
        
    return traces_convoluted, traces_truth, phases, amplitudes
        
##################################### Plotting Functions #######################################
def PlotTraces(traces, phases = None, onehot = None, n =10):
    '''
    Plots n random traces with their corresponding phases
    '''
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("You do not have matplotlib installed. "
                      "You will not be able to use the functions in utils.py.")
    
    # make sure traces is correct shape
    shape_needed = (traces.shape[0], 300)
    if traces.shape != shape_needed:
        try:
            traces = traces.reshape(shape_needed)
        except:
            raise ValueError("Traces must be of shape {}".format(shape_needed))
    
    time = np.arange(0, 600.0, 2.0) # time array x-axis
    ygrid = n//5 # number of rows 
    if n%5 != 0: # check if there are extra traces
        ygrid += 1 # add extra row
    fig, ax = plt.subplots(ygrid, 5 ,figsize=(20, ygrid*3), sharex=True, sharey=True) # initialize figure
    for i in range(n): # loop over traces
        rand_idx = np.random.randint(0, traces.shape[0]) # get random trace
        ax[i//5, i%5].plot(time, traces[rand_idx][:], label="Trace {}".format(rand_idx)) # plot trace
        ax[i//5, i%5].plot([0, 600], [0, 0], 'k-', lw=2) # plot x-axis
        if i%5 == 0: # check if first column
            ax[i//5, i%5].set_ylabel("Amplitude (mV)") # set y-axis label
        if i//5 == ygrid-1: # check if last row
            ax[i//5, i%5].set_xlabel("Time (ns)") # set x-axis label
            
        ax[i//5, i%5].set_xlim(0, 600) # set x-axis limits
        ax[i//5, i%5].set_ylim(-0.1, 1.15) # set y-axis limits
        ax[i//5, i%5].text(0.05, 0.9, "Trace #{}".format(rand_idx), fontsize=10, color='black', transform=ax[i//5, i%5].transAxes,fontweight='bold') # add trace number
        
        if onehot is not None: # check if one hot encoded pileup is given
            if onehot[rand_idx][0] == 1: # check if pileup
                if phases is not None: # check if phases are given
                    ax[i//5, i%5].text(0.55, 0.9, "Shift: {0:.2f} ns".format(phases[rand_idx][0]), fontsize=10, color='blue', transform=ax[i//5, i%5].transAxes) # add phase
                else: # if phases are not given
                    ax[i//5, i%5].text(0.55, 0.9, "Pileup", fontsize=10, color='blue', transform=ax[i//5, i%5].transAxes) # add pileup
            else: # if no pileup
                ax[i//5, i%5].text(0.55, 0.9, "No Pileup", fontsize=10, color='red', transform=ax[i//5, i%5].transAxes) # add no pileup
                
    plt.show() # show plot
    return fig  # return figure

##################################### Models #######################################
     
##################################### Autoregression #######################################
TraceAutoencoder = keras.Sequential(
    [
      layers.Flatten(input_shape=(1, 300), name="encoder-flatten1"), # flatten layer
      layers.Reshape((300, 1), name="encoder-reshape1"), # reshape layer
      layers.Conv1D(kernel_size=300, filters=300, strides=1, activation='tanh', name="encoder-conv1"), # convolutional layer
      layers.MaxPooling1D(pool_size=1, strides=2, name="encoder-maxpool1"), # max pooling layer
      layers.Conv1D(kernel_size=1, filters=300, strides=1, activation='relu', name="encoder-conv2"), # convolutional layer
      layers.MaxPooling1D(pool_size=1, strides=2, name="encoder-maxpool2"), # max pooling layer
      layers.Flatten(name="encoder-flatten2"), # flatten layer
      layers.Dense(300, activation='relu', name="encoder-dense1"), # dense layer
      layers.Reshape((300, 1), name="encoder-reshape2"), # reshape layer
      layers.Dense(64, activation='tanh', name="encoder-dense2"), # dense layer
      layers.Conv1DTranspose(kernel_size=1, filters=2, activation='tanh', name="decoder-conv1"),
      layers.Reshape((2, 300), name="decoder-reshape1"),
      layers.LSTM(64, return_sequences=True, name="decoder-lstm0"),
      layers.Dense(150, activation='tanh', name="decoder-dense1"),
      layers.Dense(300, activation='linear', name="decoder-dense2")    
    ],
    name="autoencoder"
)

##################################### Phase Regression #######################################
PhaseRegressor = keras.Sequential(
    [
      layers.Conv1D(kernel_size=1, filters=300, strides=1, activation='relu', name="phase-regressor-conv1", input_shape=(2,300)),
      layers.MaxPooling1D(pool_size=1, strides=2, name="phase-regressor-maxpool1"),
      layers.LSTM(124, return_sequences=True, name="phase-regressor-lstm"),
      layers.Dense(64, activation='relu', name="phase-regressor-dense1"),
      layers.Dense(16, activation='relu', name="phase-regressor-dense2"),
      layers.Dense(1, activation='linear', name="phase-regressor-output")
    ],
    name="phase_regressor"
)

##################################### Amplitude Regression #######################################
AmplitudeRegressor = keras.Sequential(
    [
      layers.Conv1D(kernel_size=1, filters=300, strides=1, activation='relu', name="amplitude-regressor-conv1", input_shape=(2,300)),
      layers.MaxPooling1D(pool_size=1, strides=2, name="amplitude-regressor-maxpool1"),
      layers.LSTM(124, return_sequences=True, name="amplitude-regressor-lstm"),
      layers.Dense(64, activation='relu', name="amplitude-regressor-dense1"),
      layers.Dense(16, activation='relu', name="amplitude-regressor-dense2"),
      layers.Dense(1, activation='linear', name="amplitude-regressor-output")
    ],
    name="amplitude_regressor"
)

##################################### Pileup Classifier #######################################
PileupClassifier = keras.Sequential(
    [
      layers.Reshape((300, 1), name="pileup-classifier-reshape1", input_shape=(1, 300)),
      layers.Conv1D(kernel_size=300, filters=124, strides=1, activation='relu', name="pileup-classifier-conv1"),
      layers.MaxPooling1D(pool_size=1, strides=2, name="pileup-classifier-maxpool1"),
      layers.Flatten(name="pileup-classifier-flatten"),
      layers.Dense(256, activation='relu', name="pileup-classifier-dense1"),
      layers.Dense(64, activation='relu', name="pileup-classifier-dense2"),
      layers.Dense(2, activation='softmax', name="pileup-classifier-output")
    ],
    name="pileup_classifier"
) 

##################################### Model Functions #######################################
def GetModel(name):
    '''
    returns model with specified name, compiled with optimzier and loss function
    '''
    if name == "TraceAutoencoder":
        model = TraceAutoencoder
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.003), loss=keras.losses.MeanSquaredError(), metrics=[keras.metrics.MeanSquaredError()])
    elif name == "PhaseRegressor":
        model = PhaseRegressor
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.003), loss=keras.losses.MeanSquaredError(), metrics=[keras.metrics.MeanSquaredError()])
    elif name == "AmplitudeRegressor":
        model = AmplitudeRegressor
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.003), loss=keras.losses.MeanSquaredError(), metrics=[keras.metrics.MeanSquaredError()])
    elif name == "PileupClassifier":
        model = PileupClassifier
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.003), loss=keras.losses.CategoricalCrossentropy(), metrics=[keras.metrics.CategoricalAccuracy()])
    else:
        raise ValueError("Invalid model name")
    return model

def TrainModel(name, x, y, outfile, weightfile=None, epochs=1, batch_size=64, validation_split=0.2):
    '''
    trains model with specified parameters
    '''
    # check if outfile exists    
    model = GetModel(name) # get model
    if weightfile is not None: # load weights if specified
        model.load_weights(weightfile) # load weights
    model.fit(x, y, epochs=epochs, batch_size=batch_size, validation_split=validation_split) # train model
    model.save_weights(outfile) # save weights
    return model
