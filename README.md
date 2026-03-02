#🎙️Deep Learning Based Speech Emotion Recognition with Mel-Spectrogram CNN:
This project implements a Convolutional Neural Network (CNN) to classify human emotions from speech audio using Mel-Spectrogram
features and This model is trained onthe CREMA-D dataset and predicts one of six emotions from raw audio input.


#Problem Statement
Human speech carries emotional information beyond words.  
This project aims to automatically classify emotional states from speech using deep learning techniques.


#Emotions Classified
- ANG – Angry
- DIS – Disgust
- FEA – Fear
- HAP – Happy
- NEU – Neutral
- SAD – Sad


#Tech Stack
- Python
- PyTorch
- Librosa
- NumPy


#Project Structure
```
speech-emotion-recognition/
│── dataset.py
│── preprocess.py
│── model.py
│── train.py
│── test_dataset.py
│── predict.py
```


#Feature Extraction
- Audio is loaded using Librosa
- Converted to Mel-Spectrogram
- Converted to dB scale
- Padded/trimmed to fixed size (128x128)
- Used as CNN input


#Model Architecture
- 3 Convolutional Layers
- ReLU Activation
- MaxPooling
- Fully Connected Layers
- CrossEntropy Loss
- Adam Optimizer


#Training
- Epochs: 50
- Batch Size: 32
- Learning Rate: 0.001
- Train/Test Split: 80/20


#To train:
python train.py


#Testing:
python test_dataset.py


#Predict on Single Audio
Place your audio file as: sample.wav
Then run:python predict.py


#Results
(Add your actual results here)
- Training Accuracy: XX%
- Test Accuracy: XX%


#Important Notes
- Dataset (CREMA-D) is not included due to size
- Trained model file (.pth) is not included
- Update DATA_PATH in train.py if needed


#Future Improvements
- Add Data Augmentation
- Add Dropout to reduce overfitting
- Add Confusion Matrix Visualization
- Deploy using Streamlit
- Real-time microphone emotion detection
