# Reproducing Data Reconstruction via Neuron Exclusivity Analysis: A Deep Dive

- Luca Frattini l.frattini@student.tudelft.nl
- Stefaan Riedijk e.s.riedijk@student.tudelft.nl
- Xueyuan Chen x.chen-47@student.tudelft.nl

Gitlab repository: https://github.com/LucaFrat/Data_Reconstruction_via_ExAN
Tech blog: https://github.com/LucaFrat/Data_Reconstruction_via_ExAN/blob/main/README.md

## Instruction on how to run the code

Step 1: Prepare the Python Environment
Open a terminal (or command prompt) and run the following command:
```
pip install -r requirements.txt
```
Step 2: Run the Attack Script
Once the Python environment is set up, you can proceed to run the attack.py script. This script contains the implementation of the data reconstruction attack. Execute the following command:
```
python attack.py
```

## Introduction

In this blog post, we delve into the paper "Exploring the Security Boundary of Data Reconstruction via Neuron Exclusivity Analysis" by Pan et al., which was published by a research group from Fudan University as part of a Symposium organized by USENIX, focusing on Cyber Security. The paper presents a novel algorithm designed to retrieve training batch images from leaked neural network gradients. The primary objective of this innovative technique is to reconstruct the input data of a given neural network's average gradient from a batch more effectively than previous attack algorithms. Furthermore, this technique holds potential for enhancing the privacy of neural networks by exploiting the properties of Exclusively Activated Neurons (ExANs).

An ExAN is defined as a neuron in which the ReLU activation function is activated solely by a single sample within a batch. ExANs are considered to provide vital information on the feasibility of data reconstruction attacks, playing a central role in the algorithm presented in the paper. Identifying ExANs as the bottleneck in attack algorithms has led to improved results and carries significant implications for machine learning engineers, enabling them to bolster the privacy of their neural networks.


## Reproduction Exploration

During the reproduction exploration, we attempted to implement B1, B2, and B3 from the pseudo-code in the paper's appendix to obtain the Activation pattern, which indicates the ExANs information needed to reconstruct the samples from the batch. We primarily focused on reproducing two results: 1. Sampled reconstruction results on RetinaMNIST, and 2. Sampled results on the ISIC skin cancer dataset, reconstructed from the average gradient of VGG-13. Initially, we familiarized ourselves with the datasets and the models. The first result requires a Fully Connected Network (FCN) model with a single hidden layer, 512 features, and ReLU activation functions. We set up the model accordingly using PyTorch with the same hyperparameters. Then, we explored the RetinaMNIST medical images dataset by summarizing its characteristics and visualizing random samples. Similarly, for result 2, we imported the VGG-13 model using the pre-setup models in the nn module of PyTorch and explored the ISIC dataset as well. Since the mechanisms behind these two results are essentially the same, we decided to focus on reproducing result 1.

To simplify the reproduction, we used a batch size of 8. According to the paper, the proposed algorithm applies to both initiated networks and highly trained ones. We chose to use a simple input - the average gradient of this batch after training the FCN for one epoch.

During the implementation of B1, we faced some challenges due to ambiguous statements in the algorithm and made soft fixes on the run:

**Algorithm B.1** Determine $\{(g_c^m)_\text{bar}\}_{c=1}^K$ for $m=1,\dots,M$.
1. **Input**: The gradient of $W_H$, i.e., $\bar{G_H}$.
2. **Output**: Reconstructed labels $\{Y_1,\dots,Y_M\}$ and loss vectors $\{(g_{mc})_{c=1}^K\}_{m=1}^M$.
3. Compute $r_c := \frac{[G_H]_c}{[G_H]_1}$ for every $c$ in $1,\dots,K$.
4. Find all the disjoint index groups $\{I_m\}_{m=1}^M$ where $(r_2)_j$ is constant whenever $j \in I_m$. $M$ is hence the inferred batch size and $I_m$ is the index set of the exclusively activated neurons at the last ReLU layer.
5. for all $c$ in $1,\dots,K$ do
6. $\quad$ for all $m$ in $1,\dots,M$ do
7. $\qquad$ Select an arbitrary index $j$ from $I_m$.
8. $\qquad$ $g_{mc} / g_{m1} \leftarrow [r_c]_j$.
9. $\quad$ end for
10. end for
11. for all $m$ in $1,\dots,M$ do
12. $\quad$ $Y_m \leftarrow$ Apply Algorithm B.2 to $(g_{mc})_{c=1}^K$.
13. $\quad$ Estimate the upper bound of feasible range of $g_{m1}$ as $\delta_m \leftarrow g_{m1} / g_{mY_m}$
14. $\quad$ Fix $g_{m1} = 2 \times \delta_m/3$.
$\qquad$ ▷ This is practiced in all our experiments.
15. $\quad$ Calculate each $g_{mc}$ according to the ratio.
16. end for

* The main issue was that it was not clear either in the pseudo code or in the paper on how to retrieve sample-specific information based on only the average gradient. The reconstructed $g_{cm}$ (sample-wise) is required in algorithm B3. But it's not explained how the gradient related a specific sample.
* In step 4 of Algorithm B1, grouping duplicated ratios of the 2nd class can give an indication of how many ExANs are there in the last ReLU layer. However, this step hardly implies a correct estimation of the batch size M. The $r_{2_j}$ values are very small, where almost all of the values are nearly 0, except for $-\infty$ and $\infty$. We tried using small tolerance thresholds (1e-8) for comparison, but the result varies significantly.
* Step 12 assumes the input for B2 - $g_{cm}$ for all classes are known. But until this step, they are unknown except for the ratios of $g_{cm}/g_{1m}$ from step 8. Because the author states that $g_{1m}$ appears to be always positive, we then switched to use the ratios as the input for Algorithm B2.
* The algorithm is based on a strong assumption that in a batch, there must be at least two ExANs at the last ReLU layer and at least one ExAN at the other ReLU layers. This condition is challenging to reproduce.

After adjustments, the implementation of B1 looks like the following:

```python
def algo_B1(grads):
    """
    Algorithm B.1
    Input: The gradient of (the weight the last hidden layer W_H), which is G_H_bar
    Output: Reconstructed labels,
            loss vectors 'small_grads',
            Index sets
    """

    # The last Hidden layer
    # Our FCN has only one hidden layer, its size is 512
    G_H_bar = grads['Layer_0_w'].T
    nr_of_classes = cons.OUT_SIZE

    # r_c c=1 to K
    r_c = []

    # Compute ratio vector r_c = [G_H_bar]_c / [G_H_bar]_1 for each class
    for i in range(nr_of_classes):
        r_c.append(G_H_bar[i, :]/G_H_bar[0, :])

    # !!! Find all duplicates in r_2
    # and group them by the duplicated value
    index_sets = helpers.find_duplicate_indices(r_c[1])
    print(f'Estimated M is {len(index_sets.keys())}')

    # sample-wise
    ratio_gcm_g1m = [[] for _ in range(cons.BATCH_SIZE_TRAIN)]

    # Get all values of g_c_m_bar / g_1_m_bar
    for c in range(nr_of_classes):
        for m in range(cons.BATCH_SIZE_TRAIN):
            try:
                j = random.choice(index_sets[m].tolist())
                ratio_gcm_g1m[m].append(r_c[c][j])
            except:
                pass

    # Upper bound delta_m, sample-wise
    Deltas = []
    # Final output g_c_m, sample-wise
    g_c_m = []

    for m in range(cons.BATCH_SIZE_TRAIN):
        Y_m = algo_B2(ratio_gcm_g1m[m])

        Delta_m = 1 / ratio_gcm_g1m[Y_m]
        Deltas.append(Delta_m)

        g_1_m = 2 * Delta_m / 3

        g_c_m.append([ratio * g_1_m for ratio in ratio_gcm_g1m[m]])

    return Y_m, g_c_m, index_sets
```


Despite these challenges, we persevered in our efforts to reproduce the results presented in the paper, aiming to better understand the nuances of the proposed algorithm and its potential applications for data reconstruction and privacy enhancement.

## Possible Future Work

Given the challenges faced during the reproduction process, there is potential for future work in several areas, including the following:

- Address the issues encountered in implementing B1 and B3 from the pseudo-code: By refining the algorithm and providing clearer instructions, researchers can ensure a smoother implementation process and a more accurate reproduction of the results. This could involve clarifying ambiguous statements or providing more context for specific steps.

- Explore different neural network architectures and datasets to further test the algorithm's effectiveness: By expanding the scope of testing to include various architectures and datasets, researchers can better understand the generalizability of the algorithm and its effectiveness in a wider range of scenarios. This could help identify any limitations or edge cases that may not have been apparent in the original experiments.

- Investigate potential improvements to the algorithm based on insights gained during the reproduction process: Insights gained during the reproduction process can be valuable for improving the original algorithm. Researchers can use these insights to propose modifications, optimizations, or alternative approaches that may lead to better results or a more efficient implementation.
