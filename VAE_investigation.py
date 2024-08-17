# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 15:59:44 2024

@author: Enrico
"""



import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
import os
##############################################################################
# Define the Sampling Layer
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.keras.backend.random_normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Define the VAE loss layer
class VAELossLayer(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var, y_true, y_pred = inputs
        reconstruction_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(y_true, y_pred))
        kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
        return reconstruction_loss + kl_loss

# Refactor Conv1DVAE to extend tf.keras.Model
class Conv1DVAE(tf.keras.Model):
    def __init__(self, input_dim, latent_dim, conv_filters=[32, 64, 128], kernel_size=3):
        super(Conv1DVAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.conv_filters = conv_filters
        self.kernel_size = kernel_size

        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

    def build_encoder(self):
        encoder_inputs = tf.keras.Input(shape=(self.input_dim, 1))
        x = encoder_inputs
        for filters in self.conv_filters:
            x = layers.Conv1D(filters, self.kernel_size, activation="relu", padding="same")(x)
            x = layers.MaxPooling1D(2, padding="same")(x)
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation="relu")(x)
        z_mean = layers.Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        return models.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    def build_decoder(self):
        latent_inputs = tf.keras.Input(shape=(self.latent_dim,))
        dense_units = self.conv_filters[-1] * (self.input_dim // (2 ** len(self.conv_filters)))
        x = layers.Dense(dense_units, activation="relu")(latent_inputs)
        x = layers.Reshape((self.input_dim // (2 ** len(self.conv_filters)), self.conv_filters[-1]))(x)
        for filters in reversed(self.conv_filters):
            x = layers.UpSampling1D(2)(x)
            x = layers.Conv1D(filters, self.kernel_size, activation="relu", padding="same")(x)
        decoder_outputs = layers.Conv1D(1, self.kernel_size, activation="tanh", padding="same")(x)
        return models.Model(latent_inputs, decoder_outputs, name="decoder")

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        decoded_outputs = self.decoder(z)
        loss_layer = VAELossLayer()([z_mean, z_log_var, inputs, decoded_outputs])
        self.add_loss(loss_layer)
        return decoded_outputs

    def encode(self, x_data):
        z_mean, _, _ = self.encoder.predict(x_data)
        return z_mean

    def decode(self, z_data):
        return self.decoder.predict(z_data)

    def reconstruct(self, x_data):
        z_mean = self.encode(x_data)
        return self.decode(z_mean)


def rescale_data(x_data, min_val, max_val):
    return 2 * (x_data - min_val) / (max_val - min_val) - 1

def inverse_rescale_data(x_data, min_val, max_val):
    return (x_data + 1) * (max_val - min_val) / 2 + min_val

  # Map cluster labels to true labels using the Hungarian algorithm
def map_clusters_to_labels(y_true, y_clusters):
# Compute confusion matrix
    cm = confusion_matrix(y_true, y_clusters)
    
    # Apply the Hungarian algorithm (maximize accuracy, so we negate the cm)
    row_ind, col_ind = linear_sum_assignment(-cm)
    
    # Create a mapping from cluster label to true label
    label_mapping = {col_ind[i]: i for i in range(len(col_ind))}
    
    # Apply the mapping to the cluster labels
    corrected_clusters = np.array([label_mapping[cluster] for cluster in y_clusters])
    
    return corrected_clusters
###############################################################################
if __name__ == "__main__":
    # Assuming input sequences are of length 128
    input_dim = 128
    latent_dim = 100
    conv_filters = [32, 64,128]
    
    #breakpoint()
    # Initialize the VAE
    vae = Conv1DVAE(input_dim, latent_dim,conv_filters)
    # Print summaries of encoder and decoder
    print("Encoder Summary:")
    vae.encoder.summary()
    
    print("Decoder Summary:")
    vae.decoder.summary()
    
    print("VAE Summary:")
    # Call the model on a batch of data to build it
    sample_input = tf.random.normal(shape=(1, 128, 1))  # Batch size of 1, input_dim=128, 1 channel
    vae(sample_input)
    vae.summary()

    # Generate synthetic data for demonstration purposes
    def generate_heartbeat(normal=True, length=128):
        if normal:
            return np.sin(np.linspace(0, 2 * np.pi, length)) + np.random.normal(0, 0.1, length)
        else:
            return np.sin(np.linspace(0, 2 * np.pi, length)) + np.random.normal(0, 0.3, length) + np.sin(np.linspace(0, 6 * np.pi, length))

    # Create a dataset
    n_samples = 100000
    x_data = np.array([generate_heartbeat(normal=i % 2 == 0) for i in range(n_samples)])
    x_data = np.expand_dims(x_data, axis=-1)  # Add channel dimension
    
    # Labels: 0 for normal, 1 for abnormal
    y_data = np.array([i % 2 for i in range(n_samples)])
    # Determine min and max for scaling
    min_val = np.min(x_data)
    max_val = np.max(x_data)
    
    # Rescale data
    x_data_rescaled = rescale_data(x_data, min_val, max_val)
    path_to_my_weights = 'C:/Users/Enrico/Desktop/Progetti/23 VARIATIONAL AUTOENCODER/VAE_weights.h5'
    if not os.path.exists(path_to_my_weights):
        print('Starting Training..')
        # Shuffle the data
        #np.random.shuffle(x_data)
        # Train the VAE
        #vae.train(x_data, epochs=50, batch_size=128)
        vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        vae.fit(x_data_rescaled, x_data_rescaled, epochs=20, batch_size=128,validation_split=0.2)
        vae.save_weights('VAE_weights.h5')
    else:
        print('Loading Weight..')
        # Load weights into the model
        vae.load_weights(path_to_my_weights)
    # Encode data into the latent space
    z_mean = vae.encode(x_data_rescaled)

    # Decode from the latent space
    reconstructed_data_rescaled = vae.decode(z_mean)
    # Inverse rescale reconstructed data
    reconstructed_data = inverse_rescale_data(reconstructed_data_rescaled, min_val, max_val)
    
    # Visualize some original vs reconstructed data
    for i in range(2):
        plt.figure(figsize=(10, 2))
        plt.subplot(1, 2, 1)
        plt.plot(x_data[i].squeeze(), label="Original")
        plt.title("Original")
        plt.subplot(1, 2, 2)
        plt.plot(reconstructed_data[i].squeeze(), label="Reconstructed")
        plt.title("Reconstructed")
        # Save the figure
        filename = f'figure_{i}.png'  # Change the file extension as needed (e.g., .jpg, .pdf)
        plt.savefig(filename)
          
        # # Optionally, clear the current figure to free memory
        # plt.close()
        # plt.show()
    
    ###########################################################################
    z_mean, z_log_var, z = vae.encoder.predict(x_data_rescaled)
    kmeans = KMeans(n_clusters=2, random_state=42)  # Adjust the number of clusters as needed
    #PCA  ######################################################################
    pca_vis = PCA(n_components=10)
    z_pca_vis = pca_vis.fit_transform(z_mean)
    # Apply KMeans
    clusters = kmeans.fit_predict(z_pca_vis)
    # Correct the cluster labels based on the true labels
    corrected_clusters_PCA = map_clusters_to_labels(y_data, clusters)
    
    # Now compute the confusion matrix and accuracy for PCA
    cm_PCA = confusion_matrix(y_data, corrected_clusters_PCA)
    accuracy_PCA = accuracy_score(y_data, corrected_clusters_PCA)
    
    print(f"Confusion Matrix PCA:\n{cm_PCA}")
    print(f"Accuracy PCA: {accuracy_PCA}")

    ###########################################################################
    
    ##### TSNE #################################################################
    tSNE = TSNE(
    n_components=2,
    perplexity=300,
    #learning_rate=600,
    #n_iter=500,
    # early_exaggeration=12,
    # metric='euclidean',
    init='pca',
    #method='exact'
    #random_state=42
    )
    z_TSNE_vis = tSNE.fit_transform(z_mean)
    #Apply KMeans
    # Manually set initial centroids
    # initial_centroids = np.array([
    #     [-150.0, 0.0],
    #     [150.0, 9.0]
    # ])
    #kmeans = KMeans(n_clusters=2, init=initial_centroids, n_init=1, algorithm='elkan')  
    clusters_TSNE = kmeans.fit_predict(z_TSNE_vis )
    # Correct the cluster labels based on the true labels
    corrected_clusters_TSNE = map_clusters_to_labels(y_data, clusters_TSNE)
    cm_TSNE = confusion_matrix(y_data, corrected_clusters_TSNE)
    accuracy_TSNE = accuracy_score(y_data, corrected_clusters_TSNE)
    
    print(f"Confusion Matrix TSNE:\n{cm_TSNE}")
    print(f"Accuracy TSNE: {accuracy_TSNE}")
    
    ############################################################################
    ### plotting ##############################################################
    # Visualization of clusters in 2D
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    # Plot for PCA visualization
    scatter_PCA = axes[0].scatter(z_pca_vis[:, 0], z_pca_vis[:, 1], c=clusters, cmap='viridis', s=50, alpha=0.7)
    axes[0].set_title('KMeans Clusters after PCA')
    axes[0].set_xlabel('KMeans Component 1')
    axes[0].set_ylabel('KMeans Component 2')
    plt.colorbar(scatter_PCA, ax=axes[0], label='True Labels')
    
    # Plot for t-SNE visualization
    scatter_TSNE = axes[1].scatter(z_TSNE_vis[:, 0], z_TSNE_vis[:, 1], c=corrected_clusters_TSNE, cmap='viridis', s=50, alpha=0.7)
    axes[1].set_title('KMeans Clusters after t-SNE')
    axes[1].set_xlabel('t-SNE Component 1')
    axes[1].set_ylabel('t-SNE Component 2')
    plt.colorbar(scatter_TSNE, ax=axes[1], label='True Labels')
    
    # Show the combined plot
    plt.tight_layout()
    plt.show()
   
    fig.savefig("kmeans_pca_tsne_clusters.png")

    # Visualize the confusion matrix ##########################################
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    # Plot for PCA Confusion Matrix
    sns.heatmap(cm_PCA, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_xlabel("Predicted Cluster Labels")
    axes[0].set_ylabel("True Labels")
    axes[0].set_title("PCA - Confusion Matrix between Clusters and True Labels")
    
    # Plot for t-SNE Confusion Matrix
    sns.heatmap(cm_TSNE, annot=True, fmt='d', cmap='Blues', ax=axes[1])
    axes[1].set_xlabel("Predicted Cluster Labels")
    axes[1].set_ylabel("True Labels")
    axes[1].set_title("t-SNE - Confusion Matrix between Clusters and True Labels")
    
    # Show the combined plot
    plt.tight_layout()
    plt.show()
    fig.savefig("cm_pca_tsne_clusters.png")
    

