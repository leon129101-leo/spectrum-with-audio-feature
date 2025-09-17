# spectrum-with-audio-feature
Stellar Spectral Classification with Multimodal Audio-Inspired Features

This repository contains the official implementation of our paper:

Listening to Stars: Audio-Inspired Multimodal Learning for Star Classification
[Author: Shengwen Zhang, Yanxia Zhang, Chao Liu]

🔑 Overview

Stellar spectral classification is crucial for understanding stellar properties such as temperature, composition, and luminosity. Traditional approaches (template fitting, color-magnitude cuts, and machine learning on raw spectra or images) are limited by degeneracies and the narrow flux–wavelength representation.

In this work, we propose a novel multimodal framework that integrates:

1D spectra (processed with an 8-layer CNN + Coord Attention)

2D spectral images (processed with EPSANet-50)

Audio-inspired features (Mel spectrogram, MFCC, LFCC processed with a 3-layer CNN)

The outputs are mapped into 256-dimensional embeddings, fused via a fully connected layer, and refined with attention mechanisms.

🚀 Key Results

1D spectra (Coord Attention): 89.75 ± 0.28 %

Mel spectrogram only: 90.23 ± 0.36 %

1D + 2D fusion: 91.26 ± 0.35 %

Spectra + audio features: 89.09 ± 0.43 %

Full multimodal (spectra + images + audio): 91.79 ± 0.11 % (best)

These results show that incorporating audio-inspired features provides complementary structural information that improves stellar classification accuracy.

📊 Data Preparation

Spectral data (1D): Provide FITS or preprocessed .spec files.

Spectral images (2D): Convert spectra into image-like representations.

Audio-inspired features: Extracted via Mel spectrogram, MFCC, and LFCC transforms (see utils/feature_extraction.py).
