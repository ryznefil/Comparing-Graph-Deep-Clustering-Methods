# Comparing Graph Clustering Methods

## Abstract
This study evaluates classical and deep learning-based graph clustering methods. Classical methods like Spectral Clustering have been successful, but newer deep learning approaches promise better scalability. We compare Spectral Clustering, Louvain's Algorithm, and three deep learning methods: GraphEncoder, MinCutPool, and Structural Deep Convolutional Network (SDCN). Our findings show that while deep learning methods offer scalability, they underperform classical methods in the absence of non-graphical features. SDCN, however, outperforms Louvain's Algorithm, highlighting the potential of deep learning in graph-based tasks.

## Introduction
Graph clustering algorithms categorize objects with similar properties into groups. These algorithms are crucial in various domains, including sociology, biology, and computer science. The study focuses on evaluating the performance of both classical and deep learning-based graph clustering methods using the Stochastic Block Model and Normalized Mutual Information (NMI) criteria.

## Methods
- **Classical Methods**: Spectral Clustering and Louvain's Algorithm.
- **Deep Learning Methods**: GraphEncoder, MinCutPool, and SDCN.
- **Datasets**: Synthetic graphs generated from the Stochastic Block Model.
- **Evaluation Metric**: Normalized Mutual Information (NMI).

## Results
- Spectral Clustering showed the highest performance across all complexity levels.
- Louvain's Algorithm was sensitive to imbalanced community sizes.
- Deep learning methods, except for SDCN, performed poorly without additional non-graphical features.
- SDCN demonstrated the potential of deep learning methods, outperforming Louvain's Algorithm.

## Conclusion
Classical methods remain superior in precision, but deep learning methods, particularly SDCN, show promise for scalability and performance in graph clustering tasks. The study suggests that the success of deep learning methods may depend on the availability of rich non-graphical features.

## Potential Impact
The insights from this study can guide the development of more effective graph clustering algorithms, combining the precision of classical methods with the scalability of deep learning approaches.

---

The code for all experimental work is available at [GitHub](https://github.com/arifmoh2/community-detetction-deep-learning).

**Note**: This description is a summary of the research paper provided. For detailed methodology, results, and discussions, please refer to the full paper.
