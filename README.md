# Comparing Graph Clustering Methods

## Overview
This project investigates the performance of various graph clustering methods, comparing traditional algorithms like Spectral Clustering and Louvain's Algorithm with newer deep learning approaches: GraphEncoder, MinCutPool, and Structural Deep Convolutional Network (SDCN). Our study focuses on synthetic graphs generated from the Stochastic Block Model and evaluates performance based on normalized mutual information criteria.

## Methods
- **Spectral Clustering**: Utilizes the eigenvalue decomposition of the graph similarity matrix.
- **Louvain's Algorithm**: A greedy community detection algorithm that maximizes modularity.
- **GraphEncoder**: Employs a deep learning autoencoder for graph embedding followed by k-means clustering.
- **MinCutPool**: Uses graph neural networks to solve a relaxed form of the min-cut problem.
- **Structural Deep Convolutional Network (SDCN)**: Combines autoencoder and graph neural network approaches, optimizing both local and global structure information.

## Key Findings
- **Spectral Clustering** achieved the highest normalized mutual information (NMI) scores across all complexity levels.
- **SDCN** showed relatively strong performance but did not surpass Spectral Clustering.
- **Louvain's Algorithm** performed well at lower graph complexities.
- **GraphEncoder and MinCutPool** struggled to achieve high NMI scores, potentially due to the lack of non-graphical features in the dataset.

## Significance
The results underscore the efficacy of traditional methods like Spectral Clustering in community detection tasks. However, SDCN's deep learning framework suggests that incorporating machine learning can enhance performance, especially in complex scenarios or when non-graphical features are present. This insight motivates further exploration into how deep learning can be effectively applied to graph data analytics.

## Repository
The code and implementation of our experiments are available at [comparative-graph-clustering](https://github.com/arifmoh2/community-detetction-deep-learning).

### Collaborators
- Filip Ryzner
- Mohammad Mustafa Arif

_Publication Date: May 15, 2022_
