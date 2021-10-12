# SAR-predict-Hs
Code of internship: Deep learning inversion of sea surface processes from SAR imaging

# Content of the repository:

- Preprocessing data: Convert h5 files to .pt files to load data into RAM to avoid frequent hard disk reading (I/O will limit performance)
- Preliminary experiments: Reproduce of Quach et al.(2020)
- Attention-based network: Training of proposed attention-based networks
- Attention-based network - real part: Training of proposed attention-based network without imaginary part
- Attention-based network - data augmentation: Training of proposed attention-based network with data augmentation
- Increase channels and CNNs: Training of increasing channel of Quach et al.(2020) and the model proposed in this study without any high level features

# Performance of proposed model

- Plot_MSE&STD:

<img src="https://github.com/LANZhengyang/SAR-predict-Hs/blob/main/Image/Plot_MSE%26STD.png" width="200">

- Scatter plots of predictions versus measurements:

<img src="https://github.com/LANZhengyang/SAR-predict-Hs/blob/main/Image/Scatter_plots_of_predictions_versus_measurements.png" width="200">
