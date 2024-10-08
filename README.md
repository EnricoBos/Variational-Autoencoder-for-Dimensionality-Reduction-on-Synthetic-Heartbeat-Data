# Variational Autoencoder for Dimensionality Reduction on Synthetic Heartbeat Data

This project demonstrates the application of a Variational Autoencoder (VAE) for dimensionality reduction, using synthetic "Heartbeat" data as a test case. The analysis includes a comparison between the effectiveness of Principal Component Analysis (PCA) and t-SNE in visualizing and separating the latent space generated by the VAE. The results highlight the capabilities of these techniques in uncovering meaningful patterns in high-dimensional data


## Why Use a Variational Autoencoder (VAE) Instead of a Traditional Autoencoder (AE)?
A Variational Autoencoder (VAE) offers several advantages over a traditional Autoencoder (AE) when it comes to analyzing and interpreting the latent space:
 -  Latent Space Representation: 
	 - Traditional AE: 
		Maps input data to a fixed latent space. The latent variables are direct encodings of the data, and there are no constraints on their distribution. This can result in a latent space 	that may be 	irregular or sparse, 		making it harder to explore and interpret.
	- VAE: 
	Maps input data to a distribution in the latent space (usually Gaussian). Instead of producing a single point for each input, it produces a distribution characterized by a mean and variance. This 	probabilistic approach creates a more continuous and structured latent space, which can be smoother and more interpretable.

-  Regularization:
	- Traditional AE: 
	Does not include explicit regularization on the latent space. The lack of constraints can lead to overfitting and a latent space that may not generalize well to unseen data or produce meaningful 	interpolations.
	- VAE: 	Incorporates a regularization term (KL divergence) that forces the latent space distributions to approximate a standard Gaussian. This regularization ensures that the latent space is well-structured 	and smooth, which helps in generating new samples and understanding the data distribution better.

-  Data Generation:
	- Traditional AE: 
	Primarily focuses on reconstruction and does not inherently support data generation. The latent space does not guarantee that new samples will be realistic or consistent with the original data.
	- VAE: 
	Specifically designed to generate new data samples by sampling from the latent space distributions. This capability allows for exploring the data space more effectively and creating new instances that 	adhere to the learned distribution.

-  Latent Space Analysis:
	- Traditional AE: 
	The latent space may be difficult to navigate and analyze due to its potential irregularity. Understanding how variations in the latent space affect the data can be challenging.
	- VAE: 	The continuous and probabilistic nature of the latent space makes it easier to analyze. The structured latent space allows for smoother interpolation and exploration of how different latent variables 	influence the data.


## About the Project
The project implements a 1D Convolutional Variational Autoencoder (VAE) to investigate dimensionality reduction techniques.
Synthetic data is generated, using: 
- Normal Heartbeat: A smooth sine wave with slight random noise.
- Abnormal Heartbeat: A more irregular waveform, combining a base sine wave with increased noise and additional higher-frequency oscillations.

The VAE is trained to learn a compressed latent representation of the data. The learned latent space is then compared using PCA and t-SNE for dimensionality reduction. The results show that PCA achieves better separation of data in the latent space compared to t-SNE, with the confusion matrices at the end being nearly identical.


## Implementation
Implemented VAE model including:
-  1D Convolutional VAE: Implements a VAE using 1D convolutional layers to encode and decode synthetic heartbeat signals
-  Synthetic Data Generation: Generates normal and abnormal heartbeat data with noise to simulate real-world scenarios
-  Dimensionality Reduction: Compares the latent space representation using PCA and t-SNE
-  Evaluation: Visualizes and evaluates the separation of normal and abnormal heartbeat data in the reduced space


## Environment
* Python 3.10.10 
* Tensorflow V.2.10.0 


## Executing program
The main functionality of this project is contained within main.py. Here is a breakdown of the key step
* Initialize the VAE: `vae = Conv1DVAE(input_dim=128, latent_dim=100, conv_filters=[32, 64, 128])`
* Generate and Preprocess Synthetic Data
* Train the VAE:
   - `vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))`
   - `vae.fit(x_data_rescaled, x_data_rescaled, epochs=15, batch_size=128, validation_split=0.2)`

* Dimensionality Reduction:
  - `z_mean = vae.encode(x_data_rescaled)`
  - `pca_result = PCA(n_components=2).fit_transform(z_mean)`
  - `tsne_result = TSNE(n_components=2).fit_transform(z_mean)`

* Evaluate and Visualize: The project includes visualization of the reduced data space and confusion matrices.


## Results
The project demonstrates that PCA provides a better separation of the normal and abnormal synthetic heartbeat data in the reduced space compared to t-SNE. The confusion matrix indicates that both methods perform similarly in terms of classification accuracy.

- Example of synthetic data
  
	-  Normal (top), abnormal (bottom)

![figure_0](https://github.com/user-attachments/assets/4aaeba60-1e96-4f47-9467-91bd9e8d375e)



![figure_1](https://github.com/user-attachments/assets/0ab53b89-1e6c-44e8-bc59-004a4004aa93)


  
- Principal Components

![kmeans_pca_tsne_clusters](https://github.com/user-attachments/assets/d591d164-1a50-41ad-a4f8-0adb39eb1e45)

  
- Confusion Matrix

  ![cm_pca_tsne_clusters](https://github.com/user-attachments/assets/e44366df-777d-43ae-8171-1586d3c91cd2)



## Authors
* Enrico Boscolo
