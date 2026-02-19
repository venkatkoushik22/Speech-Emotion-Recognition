Speech Emotion Recognition using MFCCs and 1D-CNN
1. Project Overview
This project focuses on building an automated Speech Emotion Recognition (SER) system using deep learning. The objective is to classify human emotions-specifically happy, sad, angry, and neutral-based solely on vocal input. The system uses advanced signal processing for feature extraction and a Convolutional Neural Network (CNN) for classification.

2. Dataset
The system is trained and validated on the RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) corpus.

Content: Audio-only samples from 24 professional actors.

Scope: 1,440 files representing a standardized set of emotional vocalizations.

Preprocessing: The audio files are converted into Mel-Frequency Cepstral Coefficients (MFCCs) to represent key acoustic characteristics.

3. Methodology
Data Augmentation
To improve model robustness and prevent overfitting, the training data was tripled using:

Noise Addition: Injecting white noise into the signals.

Pitch Shifting: Adjusting frequency while maintaining duration.

Time Stretching: Altering the speed of the audio.

Model Architecture
The 1D-CNN architecture was designed with the following layers:

Convolutional Layers: To extract spatial-temporal features from the MFCCs.

Batch Normalization: To ensure stable and fast training.

Dropout (0.2): To reduce the risk of overfitting.

Dense Layer: Fully connected layer with Softmax activation for 4-class classification.

4. Performance
The final model achieved a validation accuracy of approximately 87%.

Evaluation Metrics: The project includes ROC curves, Confusion Matrices, and full classification reports (Precision, Recall, and F1-Score).

Strength: The model performs exceptionally well in distinguishing high-energy emotions like "Angry."

5. Repository Contents
Final project.ipynb: Complete Jupyter Notebook containing data processing, model training, and performance plots.

CPE-646_Final Reprt.pdf: Detailed technical report and academic documentation.

.gitignore: Configured to exclude heavy audio datasets and local checkpoints.

6. Installation and Usage
To run this project locally, clone the repository and install the dependencies:

Bash
pip install librosa numpy pandas matplotlib seaborn scikit-learn tensorflow keras
Download the Data: Obtain the RAVDESS dataset from Kaggle or the official repository.

Set Data Path: Update the directory path in the first few cells of Final project.ipynb.

Execute: Run all cells in the notebook to view results and generate performance metrics.
