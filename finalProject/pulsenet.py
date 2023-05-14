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
    from scipy import interpolate


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

def EncodePileup(pileup):
    '''
    Returns one hot encoded pileup
    '''
    pileup_one_hot = np.zeros((pileup.shape[0], 1)) # initialize array
    for i in range(pileup.shape[0]): # loop over pileup
        if pileup[i] == 0: # check if pileup is 0
            pileup_one_hot[i] = 0 # set pileup to 0
        else: # if pileup is not 0
            pileup_one_hot[i] = 1 # set pileup to 1
    return pileup_one_hot # return one hot encoded pileup

def LoadData(filename):
    '''
    Loads data from file returns traces, phases, and one hot encoded pileup
    '''
    df = GetData(filename) # get data
    traces = GetTraces(df) # get traces
    phases = GetPhases(df) # get phases
    pileup_one_hot = EncodePileup(phases) # get one hot encoded pileup
    return traces, phases, pileup_one_hot

def getRandomPileupTraces(tt1,tt2,rndphase,scale):
  newtot = np.zeros_like(tt1)
  newtt1 = np.zeros_like(tt1)
  newtt2 = np.zeros_like(tt2)
  std2 = np.std(tt2[:60]) # gets deviation for baseline
  for i in range(len(tt1)):
    newtt1[i] = tt1[i]
    if(i<rndphase):
      newtot[i] = tt1[i]
      newtt2[i] = np.random.normal(0,std2) # gaussian random for baseline
    else:
      i2 = int(i-rndphase)
      newtt2[i] = (tt2[i2+1]-tt2[i2])*(rndphase-int(rndphase))+tt2[i2]
      newtt2[i] *= scale
      newtot[i] = tt1[i] + newtt2[i]
  max = np.max(newtot)
  nmin = np.min(newtot)
  min = newtt2[-1] if newtt1[-1]>newtt2[-1] else  newtt1[-1] #normalizes bottom
  scale = max-nmin
  # print(max,nmin,min,scale)
  return (newtot)/max,newtt1/max,newtt2/max

def CreateData(filename, pileup_split=0.5, phase_min=0.1, phase_max=100, amplitude_min=0.5, amplitude_max=1.5):
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
            t = np.linspace(0, 600, 300) # x axis
            f = interpolate.interp1d(t, no_pileup[rand_trace_idx[i]][:]) # interpolate
            traces_truth[i][1][:int(rand_phase_shifts[i])+1] = no_pileup[rand_trace_idx[i]][:int(rand_phase_shifts[i])+1]
            traces_truth[i][1][int(rand_phase_shifts[i])+1:] = f(t[int(rand_phase_shifts[i])+1:] - rand_phase_shifts[i])
            traces_truth[i][1][:] *= rand_amplitude_shifts[i]
            # traces_truth[i][1][:] = np.roll(no_pileup[rand_trace_idx[i]][:300], int(rand_phase_shifts[i]))*rand_amplitude_shifts[i]
            # traces_truth[i][1][:int(rand_phase_shifts[i])] = no_pileup[rand_trace_idx[i]][:int(rand_phase_shifts[i])]*rand_amplitude_shifts[i]
            phases[i] = rand_phase_shifts[i]
            amplitudes[i] = rand_amplitude_shifts[i]
            n_pileup_count += 1
        else: # no pileup
            traces_truth[i][1][:] = np.zeros_like(no_pileup[rand_trace_idx[i]][:300])
            phases[i] = 0
            amplitudes[i] = 0.0
            n_no_pileup_count += 1
        
        traces_convoluted[i][0][:] = traces_truth[i][0][:] + traces_truth[i][1][:]
        # renormalize
        norm = np.max(traces_convoluted[i][0][:])
        traces_convoluted[i][0][:] = traces_convoluted[i][0][:]/norm
        traces_truth[i][0][:] = traces_truth[i][0][:]/norm
        traces_truth[i][1][:] = traces_truth[i][1][:]/norm
    
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
            if onehot[rand_idx] == 1: # check if pileup
                if phases is not None: # check if phases are given
                    ax[i//5, i%5].text(0.55, 0.9, "Shift: {0:.2f} ns".format(phases[rand_idx][0]), fontsize=10, color='blue', transform=ax[i//5, i%5].transAxes) # add phase
                else: # if phases are not given
                    ax[i//5, i%5].text(0.55, 0.9, "Pileup", fontsize=10, color='blue', transform=ax[i//5, i%5].transAxes) # add pileup
            else: # if no pileup
                ax[i//5, i%5].text(0.55, 0.9, "No Pileup", fontsize=10, color='red', transform=ax[i//5, i%5].transAxes) # add no pileup
                
    plt.show() # show plot
    return fig  # return figure

##################################### Models #######################################

class TraceDiscriminatorBaseNEW(keras.Model):
    
    def __init__(self, name="trace_discriminator_baseNEW", **kwargs):
        super(TraceDiscriminatorBase, self).__init__(name=name, **kwargs)
        self.Flatten1 = layers.Flatten(input_shape=(1, 300), name="discriminator-base-flatten1")
        self.ReshapeInput = layers.Reshape((300, 1), name="discriminator-base-reshape1")
        self.Conv1D1 = layers.Conv1D(kernel_size=300, filters=300, strides=1, activation='tanh', name="discriminator-base-conv1")
        # self.MaxPooling1D1 = layers.MaxPooling1D(pool_size=1, strides=2, name="discriminator-base-maxpool1")
        self.Conv1D2 = layers.Conv1D(kernel_size=1, filters=300, strides=1, activation='relu', name="discriminator-base-conv2")
        # self.MaxPooling1D2 = layers.MaxPooling1D(pool_size=1, strides=2, name="discriminator-base-maxpool2")
        self.Flatten = layers.Flatten(name="discriminator-base-flatten")
        self.Dense1 = layers.Dense(300, activation='relu', name="discriminator-base-dense1")
        self.ReshapeDense = layers.Reshape((300, 1), name="discriminator-base-reshape2")
        self.Dense2 = layers.Dense(64, activation='tanh', name="discriminator-base-dense2")
        self.Conv1DTranspose = layers.Conv1DTranspose(kernel_size=1, filters=2, activation='tanh', name="discriminator-base-conv3")
        self.ReshapeOutput = layers.Reshape((2, 300), name="discriminator-base-reshape3")
        self.LSTM = layers.Bidirectional(layers.LSTM(300, return_sequences=True, name="discriminator-base-lstm0"))
        self.DenseOut =  layers.Dense(300, activation='relu', name="discriminator-base-dense3")
        

    
    def call(self, inputs):
        x = self.Flatten1(inputs)
        x = self.ReshapeInput(x)
        x = self.Conv1D1(x)
        # x = self.MaxPooling1D1(x)
        x = self.Conv1D2(x)
        # x = self.MaxPooling1D2(x)
        x = self.Flatten(x)
        x = self.Dense1(x)
        x = self.ReshapeDense(x)
        x = self.Dense2(x)
        x = self.Conv1DTranspose(x)
        x = self.ReshapeOutput(x)
        x = self.LSTM(x)  
        return self.DenseOut(x)

class TraceDiscriminatorHeadNEW(keras.Model):
    
    def __init__(self, name="trace_discriminator_headNEW", **kwargs):
        super(TraceDiscriminatorHead, self).__init__(name=name, **kwargs)
        self.Dense = layers.Dense(150, activation='tanh', name="discriminator-head-dense1", input_shape=(2, 300))
        self.DenseOutput = layers.Dense(300, activation='linear', name="discriminator-head-output")
    
    def call(self, inputs):
        x = self.Dense(inputs)
        return self.DenseOutput(x) 

class TraceClassifierBaseNEW(keras.Model):
    
    def __init__(self, name="trace_classifier_baseNEW", **kwargs):
        super(TraceClassifierBase, self).__init__(name=name, **kwargs)  
        self.ReshapeInput = layers.Reshape((300, 1), name="classifier-base-reshape1")
        self.Conv1D1 = layers.Conv1D(kernel_size=300, filters=300, strides=1, activation='relu', name="classifier-base-conv1")
        # self.MaxPooling1D1 = layers.MaxPooling1D(pool_size=1, strides=1, name="classifier-base-maxpool1")
        self.Flatten = layers.Flatten(name="classifier-base-flatten")
        self.Dense1 = layers.Dense(256, activation='tanh', name="classifier-base-dense1")
        self.Dense2 = layers.Dense(300, activation='tanh', name="classifier-base-dense2")
        self.ReshapeOutput = layers.Reshape((1, 300), name="classifier-base-reshape2")
        
    def call(self, inputs):
        x = self.ReshapeInput(inputs)
        x = self.Conv1D1(x)
        # x = self.MaxPooling1D1(x)
        x = self.Flatten(x)
        x = self.Dense1(x)
        x = self.Dense2(x)
        return self.ReshapeOutput(x)
    
class TraceClassifierHeadNEW(keras.Model):
        
        def __init__(self, name="trace_classifier_headNEW", **kwargs):
            super(TraceClassifierHead, self).__init__(name=name, **kwargs) 
            self.FlattenInput = layers.Flatten(name="classifier-head-flatten1", input_shape=(1, 300)) 
            self.DenseOutput = layers.Dense(1, activation='sigmoid', name="classifier-head-output")
            
        def call(self, inputs):
            x = self.FlattenInput(inputs)
            return self.DenseOutput(x)
    
class TracePhaseRegressorNEW(keras.Model):
    
    def __init__(self, name="trace_phase_regressorNEW", **kwargs):
        super(TracePhaseRegressor, self).__init__(name=name, **kwargs)
        self.Conv1D1 = layers.Conv1D(kernel_size=1, filters=300, strides=1, activation='relu', name="phase-regressor-conv1")
        # self.MaxPooling1D1 = layers.MaxPooling1D(pool_size=1, strides=1, name="phase-regressor-maxpool1")
        self.Flatten = layers.Flatten(name="phase-regressor-flatten")
        self.Dense1 = layers.Dense(256, activation='tanh', name="phase-regressor-dense1")
        self.Dense2 = layers.Dense(124, activation='tanh', name="phase-regressor-dense2")
        self.DenseOutput = layers.Dense(1, activation='linear', name="phase-regressor-output")
        
    def call(self, inputs):
        x = self.Conv1D1(inputs)
        # x = self.MaxPooling1D1(x)
        x = self.Flatten(x)
        x = self.Dense1(x)
        x = self.Dense2(x)
        return self.DenseOutput(x)
       
class TraceAmplitudeRegressorNEW(keras.Model):
    
    def __init__(self, name="trace_phase_regressorNEW", **kwargs):
        super(TraceAmplitudeRegressor, self).__init__(name=name, **kwargs)
        self.Conv1D1 = layers.Conv1D(kernel_size=1, filters=300, strides=1, activation='relu', name="amplitude-regressor-conv1")
        # self.MaxPooling1D1 = layers.MaxPooling1D(pool_size=1, strides=1, name="amplitude-regressor-maxpool1")
        self.Flatten = layers.Flatten(name="amplitude-regressor-flatten")
        self.Dense1 = layers.Dense(256, activation='tanh', name="amplitude-regressor-dense1")
        self.Dense2 = layers.Dense(124, activation='tanh', name="amplitude-regressor-dense2")
        self.DenseOutput = layers.Dense(1, activation='linear', name="amplitude-regressor-output")
        
    def call(self, inputs):
        x = self.Conv1D1(inputs)
        # x = self.MaxPooling1D1(x)
        x = self.Flatten(x)
        x = self.Dense1(x)
        x = self.Dense2(x)
        return self.DenseOutput(x)
              
class TraceClassifierDiscriminatorHeadNEW(keras.Model):
    
    def __init__(self, name="trace_classifier_discriminator_headNEW", **kwargs):
        super(TraceClassifierDiscriminatorHead, self).__init__(name=name, **kwargs)
        self.Conv1DTrace = layers.Conv1D(kernel_size=300, filters=300, strides=1, activation='tanh', name="classifier-discriminator-head-conv1-trace")
        # self.MaxPooling1DTrace = layers.MaxPooling1D(pool_size=1, strides=2, name="classifier-discriminator-head-maxpool1-trace")
        self.Conv1DTrace = layers.Conv1D(kernel_size=1, filters=300, strides=1, activation='relu', name="classifier-discriminator-head-conv2-trace")
        # self.MaxPooling1DTrace = layers.MaxPooling1D(pool_size=1, strides=2, name="classifier-discriminator-head-maxpool2-trace")
        
        self.Conv1DClass = layers.Conv1D(kernel_size=1, filters=300, strides=1, activation='relu', name="classifier-discriminator-head-conv1-class")
        # self.MaxPooling1DClass = layers.MaxPooling1D(pool_size=1, strides=2, name="classifier-discriminator-head-maxpool1-class")
        self.Conv1DClass = layers.Conv1D(kernel_size=1, filters=300, strides=1, activation='relu', name="classifier-discriminator-head-conv2-class")
        # self.MaxPooling1DClass = layers.MaxPooling1D(pool_size=1, strides=2, name="classifier-discriminator-head-maxpool2-class")
        
        self.FlattenTrace = layers.Flatten(name="classifier-discriminator-head-flatten-trace")
        self.FlattenClass = layers.Flatten(name="classifier-discriminator-head-flatten-class")
        
        self.Dense1Trace = layers.Dense(300, activation='relu', name="classifier-discriminator-head-dense1-trace")
        self.ReshapeTrace = layers.Reshape((1, 300), name="classifier-discriminator-head-reshape-trace")
        self.Dense1Class = layers.Dense(300, activation='relu', name="classifier-discriminator-head-dense1-class")
        self.ReshapeClass = layers.Reshape((1, 300), name="classifier-discriminator-head-reshape-class")
        
        self.Merge = layers.Concatenate(name="classifier-discriminator-head-merge", axis=1)
        
        self.Dense1Merged = layers.Dense(300, activation='tanh', name="classifier-discriminator-head-dense1-merged")
        self.DenseOutput = layers.Dense(300, activation='linear', name="classifier-discriminator-head-dense2-merged")
    
    def call(self, inputs):
        input_trace, input_classifier = inputs
        x_trace = self.Conv1DTrace(input_trace)
        # x_trace = self.MaxPooling1DTrace(x_trace)
        x_trace = self.Conv1DTrace(x_trace)
        # x_trace = self.MaxPooling1DTrace(x_trace)
        x_trace = self.FlattenTrace(x_trace)
        x_trace = self.Dense1Trace(x_trace)
        x_trace = self.ReshapeTrace(x_trace)
        
        x_class = self.Conv1DClass(input_classifier)
        # x_class = self.MaxPooling1DClass(x_class)
        x_class = self.Conv1DClass(x_class)
        # x_class = self.MaxPooling1DClass(x_class)
        x_class = self.FlattenClass(x_class)
        x_class = self.Dense1Class(x_class)
        x_class = self.ReshapeClass(x_class)
        
        x = self.Merge([x_trace, x_class])
        x = self.Dense1Merged(x)
        return self.DenseOutput(x)
    
class TraceDiscriminatorBase(keras.Model):
    
    def __init__(self, name="trace_discriminator_base", **kwargs):
        super(TraceDiscriminatorBase, self).__init__(name=name, **kwargs)
        self.Flatten1 = layers.Flatten(input_shape=(1, 300), name="discriminator-base-flatten1")
        self.ReshapeInput = layers.Reshape((300, 1), name="discriminator-base-reshape1")
        self.Conv1D1 = layers.Conv1D(kernel_size=300, filters=300, strides=1, activation='tanh', name="discriminator-base-conv1")
        self.MaxPooling1D1 = layers.MaxPooling1D(pool_size=1, strides=2, name="discriminator-base-maxpool1")
        self.Conv1D2 = layers.Conv1D(kernel_size=1, filters=300, strides=1, activation='relu', name="discriminator-base-conv2")
        self.MaxPooling1D2 = layers.MaxPooling1D(pool_size=1, strides=2, name="discriminator-base-maxpool2")
        self.Flatten = layers.Flatten(name="discriminator-base-flatten")
        self.Dense1 = layers.Dense(300, activation='relu', name="discriminator-base-dense1")
        self.ReshapeDense = layers.Reshape((300, 1), name="discriminator-base-reshape2")
        self.Dense2 = layers.Dense(64, activation='tanh', name="discriminator-base-dense2")
        self.Conv1DTranspose = layers.Conv1DTranspose(kernel_size=1, filters=2, activation='tanh', name="discriminator-base-conv3")
        self.ReshapeOutput = layers.Reshape((2, 300), name="discriminator-base-reshape3")
        self.LSTM = layers.LSTM(300, return_sequences=True, name="discriminator-base-lstm0")

    
    def call(self, inputs):
        x = self.Flatten1(inputs)
        x = self.ReshapeInput(x)
        x = self.Conv1D1(x)
        x = self.MaxPooling1D1(x)
        x = self.Conv1D2(x)
        x = self.MaxPooling1D2(x)
        x = self.Flatten(x)
        x = self.Dense1(x)
        x = self.ReshapeDense(x)
        x = self.Dense2(x)
        x = self.Conv1DTranspose(x)
        x = self.ReshapeOutput(x)
        return self.LSTM(x)  

class TraceDiscriminatorHead(keras.Model):
    
    def __init__(self, name="trace_discriminator_head", **kwargs):
        super(TraceDiscriminatorHead, self).__init__(name=name, **kwargs)
        self.Dense = layers.Dense(150, activation='tanh', name="discriminator-head-dense1", input_shape=(2, 300))
        self.DenseOutput = layers.Dense(300, activation='linear', name="discriminator-head-output")
    
    def call(self, inputs):
        x = self.Dense(inputs)
        return self.DenseOutput(x) 

class TraceClassifierBase(keras.Model):
    
    def __init__(self, name="trace_classifier_base", **kwargs):
        super(TraceClassifierBase, self).__init__(name=name, **kwargs)  
        self.ReshapeInput = layers.Reshape((300, 1), name="classifier-base-reshape1")
        self.Conv1D1 = layers.Conv1D(kernel_size=300, filters=300, strides=1, activation='relu', name="classifier-base-conv1")
        self.MaxPooling1D1 = layers.MaxPooling1D(pool_size=1, strides=1, name="classifier-base-maxpool1")
        self.Flatten = layers.Flatten(name="classifier-base-flatten")
        self.Dense1 = layers.Dense(256, activation='tanh', name="classifier-base-dense1")
        self.Dense2 = layers.Dense(300, activation='tanh', name="classifier-base-dense2")
        self.ReshapeOutput = layers.Reshape((1, 300), name="classifier-base-reshape2")
        
    def call(self, inputs):
        x = self.ReshapeInput(inputs)
        x = self.Conv1D1(x)
        x = self.MaxPooling1D1(x)
        x = self.Flatten(x)
        x = self.Dense1(x)
        x = self.Dense2(x)
        return self.ReshapeOutput(x)
    
class TraceClassifierHead(keras.Model):
        
        def __init__(self, name="trace_classifier_head", **kwargs):
            super(TraceClassifierHead, self).__init__(name=name, **kwargs) 
            self.FlattenInput = layers.Flatten(name="classifier-head-flatten1", input_shape=(1, 300)) 
            self.DenseOutput = layers.Dense(1, activation='sigmoid', name="classifier-head-output")
            
        def call(self, inputs):
            x = self.FlattenInput(inputs)
            return self.DenseOutput(x)
    
class TracePhaseRegressor(keras.Model):
    
    def __init__(self, name="trace_phase_regressor", **kwargs):
        super(TracePhaseRegressor, self).__init__(name=name, **kwargs)
        self.Conv1D1 = layers.Conv1D(kernel_size=1, filters=300, strides=1, activation='relu', name="phase-regressor-conv1")
        self.MaxPooling1D1 = layers.MaxPooling1D(pool_size=1, strides=1, name="phase-regressor-maxpool1")
        self.Flatten = layers.Flatten(name="phase-regressor-flatten")
        self.Dense1 = layers.Dense(256, activation='tanh', name="phase-regressor-dense1")
        self.Dense2 = layers.Dense(124, activation='tanh', name="phase-regressor-dense2")
        self.DenseOutput = layers.Dense(1, activation='linear', name="phase-regressor-output")
        
    def call(self, inputs):
        x = self.Conv1D1(inputs)
        x = self.MaxPooling1D1(x)
        x = self.Flatten(x)
        x = self.Dense1(x)
        x = self.Dense2(x)
        return self.DenseOutput(x)
       
class TraceAmplitudeRegressor(keras.Model):
    
    def __init__(self, name="trace_phase_regressor", **kwargs):
        super(TraceAmplitudeRegressor, self).__init__(name=name, **kwargs)
        self.Conv1D1 = layers.Conv1D(kernel_size=1, filters=300, strides=1, activation='relu', name="amplitude-regressor-conv1")
        self.MaxPooling1D1 = layers.MaxPooling1D(pool_size=1, strides=1, name="amplitude-regressor-maxpool1")
        self.Flatten = layers.Flatten(name="amplitude-regressor-flatten")
        self.Dense1 = layers.Dense(256, activation='tanh', name="amplitude-regressor-dense1")
        self.Dense2 = layers.Dense(124, activation='tanh', name="amplitude-regressor-dense2")
        self.DenseOutput = layers.Dense(1, activation='linear', name="amplitude-regressor-output")
        
    def call(self, inputs):
        x = self.Conv1D1(inputs)
        x = self.MaxPooling1D1(x)
        x = self.Flatten(x)
        x = self.Dense1(x)
        x = self.Dense2(x)
        return self.DenseOutput(x)
              
class TraceClassifierDiscriminatorHead(keras.Model):
    
    def __init__(self, name="trace_classifier_discriminator_head", **kwargs):
        super(TraceClassifierDiscriminatorHead, self).__init__(name=name, **kwargs)
        self.Conv1DTrace = layers.Conv1D(kernel_size=300, filters=300, strides=1, activation='tanh', name="classifier-discriminator-head-conv1-trace")
        self.MaxPooling1DTrace = layers.MaxPooling1D(pool_size=1, strides=2, name="classifier-discriminator-head-maxpool1-trace")
        self.Conv1DTrace = layers.Conv1D(kernel_size=1, filters=300, strides=1, activation='relu', name="classifier-discriminator-head-conv2-trace")
        self.MaxPooling1DTrace = layers.MaxPooling1D(pool_size=1, strides=2, name="classifier-discriminator-head-maxpool2-trace")
        
        self.Conv1DClass = layers.Conv1D(kernel_size=1, filters=300, strides=1, activation='relu', name="classifier-discriminator-head-conv1-class")
        self.MaxPooling1DClass = layers.MaxPooling1D(pool_size=1, strides=2, name="classifier-discriminator-head-maxpool1-class")
        self.Conv1DClass = layers.Conv1D(kernel_size=1, filters=300, strides=1, activation='relu', name="classifier-discriminator-head-conv2-class")
        self.MaxPooling1DClass = layers.MaxPooling1D(pool_size=1, strides=2, name="classifier-discriminator-head-maxpool2-class")
        
        self.FlattenTrace = layers.Flatten(name="classifier-discriminator-head-flatten-trace")
        self.FlattenClass = layers.Flatten(name="classifier-discriminator-head-flatten-class")
        
        self.Dense1Trace = layers.Dense(300, activation='tanh', name="classifier-discriminator-head-dense1-trace")
        self.ReshapeTrace = layers.Reshape((1, 300), name="classifier-discriminator-head-reshape-trace")
        self.Dense1Class = layers.Dense(300, activation='tanh', name="classifier-discriminator-head-dense1-class")
        self.ReshapeClass = layers.Reshape((1, 300), name="classifier-discriminator-head-reshape-class")
        
        self.Merge = layers.Concatenate(name="classifier-discriminator-head-merge", axis=1)
        
        self.Dense1Merged = layers.Dense(300, activation='tanh', name="classifier-discriminator-head-dense1-merged")
        self.DenseOutput = layers.Dense(300, activation='tanh', name="classifier-discriminator-head-dense2-merged")
    
    def call(self, inputs):
        input_trace, input_classifier = inputs
        x_trace = self.Conv1DTrace(input_trace)
        x_trace = self.MaxPooling1DTrace(x_trace)
        x_trace = self.Conv1DTrace(x_trace)
        x_trace = self.MaxPooling1DTrace(x_trace)
        x_trace = self.FlattenTrace(x_trace)
        x_trace = self.Dense1Trace(x_trace)
        x_trace = self.ReshapeTrace(x_trace)
        
        x_class = self.Conv1DClass(input_classifier)
        x_class = self.MaxPooling1DClass(x_class)
        x_class = self.Conv1DClass(x_class)
        x_class = self.MaxPooling1DClass(x_class)
        x_class = self.FlattenClass(x_class)
        x_class = self.Dense1Class(x_class)
        x_class = self.ReshapeClass(x_class)
        
        x = self.Merge([x_trace, x_class])
        x = self.Dense1Merged(x)
        return self.DenseOutput(x)
    
class TraceNet(keras.Model):
    
    def __init__(self, name="trace_net", **kwargs):
        super(TraceNet, self).__init__(name=name, **kwargs)
        self.TraceDiscriminatorBase = TraceDiscriminatorBase("discriminator_base")
        self.TraceClassifierBase = TraceClassifierBase("classifier_base")
        self.TraceClassifierHead = TraceClassifierHead("classifier_head")
        self.TraceClassifierDiscriminatorHead = TraceClassifierDiscriminatorHead("classifier_discriminator_head")
        self.TracePhaseRegressor = TracePhaseRegressor("phase_regressor")
        self.TraceAmplitudeRegressor = TraceAmplitudeRegressor("amplitude_regressor")
        
    def compile(self, optimizer=None, loss=None, metrics=None):
        super(TraceNet, self).compile()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        self.reg_loss_fn = tf.keras.losses.MeanSquaredError()
        self.class_loss_fn = tf.keras.losses.BinaryCrossentropy()
        self.trace_metric = tf.keras.metrics.MeanSquaredError(name="trace_mse")
        self.phase_metric = tf.keras.metrics.MeanSquaredError(name="phase_mse")
        self.amplitude_metric = tf.keras.metrics.MeanSquaredError(name="amplitude_mse")
        self.class_metric = tf.keras.metrics.BinaryCrossentropy(name="class_bce")

        self.loss = [self.reg_loss_fn, self.class_loss_fn]
    
    def call(self, inputs):
        # feature vectors
        discriminator_feature_vec = self.TraceDiscriminatorBase(inputs)
        classifer_feature_vec = self.TraceClassifierBase(inputs)
        # classifier output
        classifier_output = self.TraceClassifierHead(classifer_feature_vec)
        # trace output
        trace_output = self.TraceClassifierDiscriminatorHead([discriminator_feature_vec, classifer_feature_vec])
        # phase and amplitude output
        phase_output = self.TracePhaseRegressor(trace_output)
        amplitude_output = self.TraceAmplitudeRegressor(trace_output)
        
        return trace_output, phase_output, amplitude_output, classifier_output

    def classify(self, inputs):
        classifer_feature_vec = self.TraceClassifierBase(inputs)
        classifier_output = self.TraceClassifierHead(classifer_feature_vec)
        return classifier_output
    
    def trace(self, inputs): 
        discriminator_feature_vec = self.TraceDiscriminatorBase(inputs)
        classifer_feature_vec = self.TraceClassifierBase(inputs)
        trace_output = self.TraceClassifierDiscriminatorHead([discriminator_feature_vec, classifer_feature_vec])
        return trace_output        
    
    def phase(self, inputs):
        discriminator_feature_vec = self.TraceDiscriminatorBase(inputs)
        classifer_feature_vec = self.TraceClassifierBase(inputs)
        trace_output = self.TraceClassifierDiscriminatorHead([discriminator_feature_vec, classifer_feature_vec])
        phase_output = self.TracePhaseRegressor(trace_output)
        return phase_output
    
    def amplitude(self, inputs):
        discriminator_feature_vec = self.TraceDiscriminatorBase(inputs)
        classifer_feature_vec = self.TraceClassifierBase(inputs)
        trace_output = self.TraceClassifierDiscriminatorHead([discriminator_feature_vec, classifer_feature_vec])
        amplitude_output = self.TraceAmplitudeRegressor(trace_output)
        return amplitude_output
    
    def predict(self, inputs, mode=None):
        if mode is None:
            return self.call(inputs)
        elif mode == "class":
            return self.classify(inputs)
        elif mode == "trace":
            return self.trace(inputs)
        elif mode == "phase":
            return self.phase(inputs)
        elif mode == "amplitude":
            return self.amplitude(inputs) 
        else:
            print("Invalid mode. Please choose from 'class', 'trace', 'phase', 'amplitude'.")  
    
    def evaluate(self, x, y):
        y_hat_trace, y_hat_phase, y_hat_amplitude, y_hat_class = self.predict(x)
        trace_loss = self.reg_loss_fn(y[0], y_hat_trace)
        phase_loss = self.reg_loss_fn(y[1], y_hat_phase)
        amplitude_loss = self.reg_loss_fn(y[2], y_hat_amplitude)
        class_loss = self.class_loss_fn(y[3], y_hat_class)
        self.trace_metric.update_state(y[0], y_hat_trace)
        self.phase_metric.update_state(y[1], y_hat_phase)
        self.amplitude_metric.update_state(y[2], y_hat_amplitude)
        self.class_metric.update_state(y[3], y_hat_class)
        return trace_loss, phase_loss, amplitude_loss, class_loss, self.trace_metric.result(), self.phase_metric.result(), self.amplitude_metric.result(), self.class_metric.result()
    
            
        
            
            
        
            
            
        
            
            
            
         
        