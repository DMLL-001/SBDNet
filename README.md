# SBDNet: A Scale and Edge Guided Bi-Decoding Network for Land Parcel Extraction

Remote sensing-based agricultural land parcel extraction is important for managing agricultural production, monitoring farmland utilization, and supporting agricultural development planning. High-precision parcel extraction requires the simultaneous acquisition of boundary and semantic information, which is usually achieved by multitask learning. However, semantic segmentation tasks require deeper features to capture global information, while edge detection relies more on shallow features to better capture boundary details. It is difficult to learn the features of both by the same network structure. In addition, small targets are easily lost in the process, and the boundary may be broken, further affecting the accuracy of the task. To address this challenge, we propose the scale and edge guided bidecoding network (SBDNet), a novel parcel extraction framework that employs a multitask cotraining strategy. The encoder shares parameters between different tasks to improve efficiency, while the decoding phase uses U- and bidirectional flow-shaped dual decoding architectures to extract deep semantic features and shallow edge features, respectively. In addition, we incorporate a scale-attention mechanism and edge guidance modules to improve the detection of small and fragmented parcels and enhance edge coherence. Experimental results show that SBDNet outperforms existing methods, such as HRNet, DeepLabV3+, SegFormer, and semantic edge-aware networks in terms of F1 score and intersection over union (IoU). Compared with the second-ranked method, SBDNet improves the F1 score and IoU by 1.22% and 1.43%, respectively, in terms of semantic accuracy, and 1.32% and 1.88%, respectively, in terms of edge accuracy.

## Results

### Network Architecture
![SBDNet Architecture](imgs/BsiNet.png)

### Comparison with Other Methods
![Comparison Results](imgs/comparison_results.png)

### Extraction Results
![Results](imgs/results.png)
