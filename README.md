# DAVA

The code related to the paper below: DAG-Aware Variational Autoencoder for Social Propagation Graph Generation

## Statement
Due to Twitter's policy restrictions and file size upload limitations, some datasets with existing links can be accessed through universally recognized ways:

[Twitter](https://www.dropbox.com/s/7ewzdrbelpmrnxu/rumdetect2017.zip)

[Weibo](https://www.dropbox.com/s/46r50ctrfa0ur1o/rumdect.zip?dl=0)

## Start

main.py is the entry point of the function program.


We have conducted these experiments on the server with Intel(R) Xeon(R) Gold 6326 CPU @ 2.90GHz and 188GB RAM. The integrated developing environment is PyTorch on 3090Ti.
In our study, we employed two distinct models for graph-level and edge-level generation, detailed as follows:

**Graph-Level Generation Model (GRU_VAE_plain):**
- Input Layer: Linear transformation with an input feature size of 99 and an output size of 128, including bias.
- Recurrent Layer: A Gated Recurrent Unit (GRU) with an input size of 128, a hidden size of 64, and two layers, configured to have the batch size as the first dimension.
- Encoding Layer (VAE): A linear layer for the Variational Autoencoder (VAE) encoding with an input size of 64 and an output size of 6, including bias.
- Output Layer: A sequential module consisting of:
  - A linear layer with an input size of 64 and an output size of 128, including bias.
  - A Rectified Linear Unit (ReLU) activation function.
  - A final linear layer with an input size of 128 and an output size of 6, including bias.
- ReLU Activation Function: Employed within the network for introducing non-linearity.

**Edge-Level Generation Model (SimpleAttentionModel):**
- Query Transformation Layers:
  - The first query linear layer with an input size of 12 and an output size of 24, including bias. The 12 here refers to the concatenation of the 6 features output by the VAE with 6 user-specific features.
  - The second query linear layer, expanding the feature size from 24 to 99, including bias.
- Key Transformation Layers:
  - The first key linear layer with an input size of 6 and an output size of 24, including bias.
  - The second key linear layer, expanding the feature size from 24 to 99, including bias.

**Training Parameters:**
- The DAVA is trained with a batch size of 16 and for a total of 100 epochs. We use an early stopping criterion based on 0.1% validation loss fluctuation with the patience of 3 epochs.
- The Adam optimizer is utilized with an initial learning rate of 0.001.
- To divide the training dataset and the test dataset, a 10-fold cross-validation strategy is used. In each fold, we train DAVA using the training set. Then each scale of propagation cascade in the test set representing our expected cascade scale is specified for DAVA to generate new propagation cascade graphs. Next, we employ three evaluation scenarios (including the authoritative MMD and downstream task scenarios) and compare them with SOTAs to measure the performance of DAVA. 
- Glorot initialization is used for weight parameter initialization.
- In the KL constraint part of the loss function, the latent variable distribution is enforced to conform to an exponential distribution with a lambda of 1.
- An attention mechanism with 4 heads is opted.

The configurations for these models were fine-tuned to achieve the optimal results presented in our paper. These settings were instrumental in effectively capturing the complexities inherent in graph-level and edge-level generation tasks.
