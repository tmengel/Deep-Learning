import pulsenet as pn
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers, models
import pandas as pd


### File Paths ###
filename = "ysoTracesNoPileup.root"

discriminator_head_weights = "weights/discriminator_head_initial.h5"
discriminator_base_weights = "weights/discriminator_base_initial.h5"
classifier_head_weights = "weights/classifier_head_initial.h5"
classifier_base_weights = "weights/classifier_base_initial.h5"
phase_weights = "weights/phase_initial.h5"
amplitude_weights = "weights/amplitude_initial.h5"
transfer_head_weights = "weights/transfer_head_initial.h5"

discriminator_base_fine_tuned_weights = "weights/discriminator_base_fine_tuned.h5"
classifier_base_fine_tuned_weights = "weights/classifier_base_fine_tuned.h5"
phase_fine_tuned_weights = "weights/phase_fine_tuned.h5"
amplitude_fine_tuned_weights = "weights/amplitude_fine_tuned.h5"
transfer_head_fine_tuned_weights = "weights/transfer_head_fine_tuned.h5"

discriminator_initial_history ="history/discriminator_initial_history.h5"
classifier_initial_history = "history/classifier_initial_history.h5"
phase_initial_history = "history/phase_initial_history.h5"
amplitude_initial_history = "history/amplitude_initial_history.h5"

transfer_history = "history/transfer_history.h5"
fine_tune_history = "history/fine_tune_history.h5"
control_history = "history/control_history.h5"

transfer_model = "weights/transfer_model.h5"
fine_tuned_model = "weights/fine_tuned_model.h5"
control_model = "weights/control_model.h5"

transfer_head_weights_trace ="weights/transfer_head_trace.h5"
transfer_history_trace = "history/transfer_history_trace.h5"

discriminator_base_control_weights = "weights/discriminator_base_control.h5"
classifier_base_control_weights = "weights/classifier_base_control.h5"
phase_control_weights = "weights/phase_control.h5"
amplitude_control_weights = "weights/amplitude_control.h5"
transfer_head_control_weights = "weights/transfer_head_control.h5"


transfer_head_long_transfer_weights = "weights/transfer_head_long_tranfer.h5"
transfer_history_long_transfer = "history/transfer_history_long_transfer.h5"
transfer_model_long_transfer = "weights/transfer_model_long_transfer.h5"

discriminator_base_fine_tuned_weights_long_transfer = "weights/discriminator_base_fine_tuned_weights_long_transfer.h5"
classifier_base_fine_tuned_weights_long_transfer = "weights/classifier_base_fine_tuned_weights_long_transfer.h5"
phase_regressor_fine_tuned_weights_long_transfer = "weights/phase_regressor_fine_tuned_weights_long_transfer.h5"
amplitude_regressor_fine_tuned_weights_long_transfer = "weights/amplitude_regressor_fine_tuned_weights_long_transfer.h5"
transfer_head_fine_tuned_weights_long_transfer = "weights/transfer_head_fine_tuned_weights_long_transfer.h5"

fine_tuned_long_transfer_model = "weights/fine_tuned_long_transfer_model.h5"
fine_tuned_long_transfer_history = "history/fine_tuned_long_transfer_history.h5"


# Model
input = layers.Input(shape=(1, 300))  # Returns a placeholder tensor
classifer_feature_vec = pn.TraceClassifierBase(name = "classifier_base")(input)
discriminator_feature_vec = pn.TraceDiscriminatorBase(name = "discriminator_base")(input)
trace_output = pn.TraceClassifierDiscriminatorHead(name = "transfer_head")([discriminator_feature_vec, classifer_feature_vec])
phase_output = pn.TracePhaseRegressor(name="phase_regressor")(trace_output)
amplitude_output = pn.TraceAmplitudeRegressor(name ="amplitude_regressor")(trace_output)
classifier_output = pn.TraceClassifierHead(name="classifier_head")(classifer_feature_vec)

# load new data
x_trace, y_trace, y_phase, y_amp = pn.CreateData(filename, pileup_split=0.5, phase_min=0.1, phase_max=20, amplitude_min=0.5, amplitude_max=1.5)

# Transfer Model
# create model
model = models.Model(inputs=input, outputs=[trace_output, phase_output, amplitude_output])
model.compile(optimizer='adam', loss="mse", metrics=['accuracy'])
model_names = [model.layers[i].name for i in range(len(model.layers))]
# load weights
model.layers[model_names.index("discriminator_base")].load_weights(discriminator_base_weights)
model.layers[model_names.index("classifier_base")].load_weights(classifier_base_weights)
model.layers[model_names.index("phase_regressor")].load_weights(phase_weights)
model.layers[model_names.index("amplitude_regressor")].load_weights(amplitude_weights)
# freeze layers
model.layers[model_names.index("discriminator_base")].trainable = False
model.layers[model_names.index("classifier_base")].trainable = False
model.layers[model_names.index("phase_regressor")].trainable = False
model.layers[model_names.index("amplitude_regressor")].trainable = False                           
# print summary
model.summary()                                                         
# Train Discriminator Head
history = model.fit(x_trace, [y_trace, y_phase, y_amp], epochs=500, batch_size=128, validation_split=0.2, verbose=1)
# save weights
model.layers[model_names.index("transfer_head")].save_weights(transfer_head_weights)
# save model
model.save(transfer_model)
# save history
pd.DataFrame(history.history, index=history.epoch, columns=history.history.keys()).to_hdf(transfer_history, key="hist")


# Fine Tune Model

model = models.Model(inputs=input, outputs=[trace_output, phase_output, amplitude_output])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss="mse", metrics=['accuracy'])
model_names = [model.layers[i].name for i in range(len(model.layers))]
# load weights
model.layers[model_names.index("discriminator_base")].load_weights(discriminator_base_weights)
model.layers[model_names.index("classifier_base")].load_weights(classifier_base_weights)
model.layers[model_names.index("phase_regressor")].load_weights(phase_weights)
model.layers[model_names.index("amplitude_regressor")].load_weights(amplitude_weights)
model.layers[model_names.index("transfer_head")].load_weights(transfer_head_weights)                     
# print summary
model.summary()                                                         
# Train Discriminator Head
history = model.fit(x_trace, [y_trace, y_phase, y_amp], epochs=500, batch_size=128, validation_split=0.2, verbose=1)
# save weights
model.layers[model_names.index("discriminator_base")].save_weights(discriminator_base_fine_tuned_weights)
model.layers[model_names.index("classifier_base")].save_weights(classifier_base_fine_tuned_weights)
model.layers[model_names.index("phase_regressor")].save_weights(phase_fine_tuned_weights)
model.layers[model_names.index("amplitude_regressor")].save_weights(amplitude_fine_tuned_weights)
model.layers[model_names.index("transfer_head")].save_weights(transfer_head_fine_tuned_weights)
# save model
model.save(fine_tuned_model)
# save history
pd.DataFrame(history.history, index=history.epoch, columns=history.history.keys()).to_hdf(fine_tune_history, key="hist")