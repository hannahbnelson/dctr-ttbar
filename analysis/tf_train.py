import numpy as np
import pandas as pd
import awkward as ak
import tensorflow as tf
import matplotlib.pyplot as plt
import topcoffea.modules.utils as utils
import pickle
import gzip
import logging
import time


current_day = time.strftime("%Y%m%d-%H%m", time.localtime())
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s', filename=f"train_{current_day}.log")

rando = 1234

fSMEFT = "/afs/crc.nd.edu/user/h/hnelson2/dctr/analysis/dctr_SMEFT.pkl.gz"
fpowheg = "/afs/crc.nd.edu/user/h/hnelson2/dctr/analysis/dctr_powheg.pkl.gz" 

inputs_smeft= pickle.load(gzip.open(fSMEFT)).get()
inputs_powheg = pickle.load(gzip.open(fpowheg)).get()
inputs_powheg = inputs_powheg.sample(frac=1, random_state=rando).reset_index(drop=True)

smeft_train = inputs_smeft.sample(frac=0.7, random_state=rando)
smeft_test = inputs_smeft.drop(smeft_train.index)

powheg_train = inputs_powheg.iloc[:1958489].sample(frac=0.7, random_state=rando)
powheg_test = inputs_powheg.iloc[:1958489].drop(powheg_train.index)

truth_smeft = np.ones_like(smeft_train['weights'])
truth_powheg = np.zeros_like(powheg_train['weights'])

weights_smeft = smeft_train['weights']
weights_powheg = powheg_train['weights']

# z_0 = inputs_smeft.to_numpy()
# z_1 = inputs_powheg.dropna().to_numpy()

z_0 = smeft_train
z_1 = powheg_train
w_0 = z_0['weights']
w_1 = z_1['weights']
y_0 = np.ones_like(z_0['weights'])
y_1 = np.zeros_like(z_1['weights'])

z = np.concatenate([z_0, z_1], axis=0).astype(np.float32)
w = np.concatenate([w_0, w_1], axis=0).astype(np.float32)
y = np.concatenate([y_0, y_1], axis=0).astype(np.float32)

dataset = tf.data.Dataset.from_tensor_slices((z, y, w)).shuffle(len(z)).batch(128)
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(z)

# Define the neural network model
def create_model(input_dim):
    model = tf.keras.Sequential([
        normalizer, 
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(300, activation='relu'),
        tf.keras.layers.Dropout(0.2, seed=rando),
        tf.keras.layers.Dense(300, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

# Define custom loss function
def custom_loss(y_true, y_pred, weights):
    term_1 = tf.reduce_sum(weights * y_true * -tf.math.log(y_pred + 1e-8))
    term_0 = tf.reduce_sum(weights * (1 - y_true) * -tf.math.log(1 - y_pred + 1e-8))
    return term_1 + term_0

# Training step
@tf.function
def train_step(x, y, weights):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = custom_loss(y, predictions, weights)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Compute r(z)
def compute_r(z):
    f_z = model.predict(z)
    return f_z / (1 - f_z + 1e-8)

# Initialize the model
input_dim = z.shape[1]
model = create_model(input_dim)

# Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Training loop
epochs = 500
for epoch in range(epochs):
    epoch_loss = 0
    for x_batch, y_batch, w_batch in dataset:
        loss = train_step(x_batch, tf.expand_dims(y_batch, axis=-1), w_batch)
        epoch_loss += loss.numpy()
    # print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")
    logger.info(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")

    if epoch % 50 == 0:
        model.save(f"trained_model_{epoch}_{current_day}.h5")

model.save(f"trained_model_final_{current_day}.h5")

#r_z_0 = compute_r(smeft_test)
#r_z_1 = compute_r(powheg_test)

# Reweighting and closure test
#weights_0_to_1 = w_0 * r_z_0.flatten()
#weights_1_to_0 = w_1 / (r_z_1.flatten() + 1e-8)

# Make histograms
#bins = np.linspace(0, np.max([z_0.max(), z_1.max()]), 50)
#fig = plt.figure(figsize=(12, 6))

# Original distributions
#plt.hist(z_0['mtt'], bins=bins, weights=w_0, alpha=0.5, label="Original $z_0$ (dσ₀)", density=True)
#plt.hist(z_1['mtt'], bins=bins, weights=w_1, alpha=0.5, label="Original $z_1$ (dσ₁)", density=True)

# Reweighted distributions
#plt.hist(z_0['mtt'], bins=bins, weights=weights_0_to_1, histtype='step', lw=2, label="Reweighted $z_0 \\to dσ₁$", density=True)
#plt.hist(z_1['mtt'], bins=bins, weights=weights_1_to_0, histtype='step', lw=2, label="Reweighted $z_1 \\to dσ₀$", density=True)

# Plot settings
#plt.xlabel("z")
#plt.ylabel("Density")
#plt.legend()
#plt.title("Closure Test: Reweighting Distributions")
#plt.grid()
#fig.savefig("dnn_output.png")

