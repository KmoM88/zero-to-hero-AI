
# 🧠 Learning Neural Networks, Deep Learning, and LLMs with Python

This repository contains a practical and progressive roadmap to learn from the fundamentals of neural networks to large language models (LLMs), using **Python**, **TensorFlow**, and **PyTorch**, with executable examples on a laptop and open datasets.

---

## 📌 General Index


### 🔰 Module 1: Fundamentals of Neural Networks
> Objective: Understand and implement basic neural networks from scratch.

- [x] 1.0 Brief review: linear algebra and calculus for neural networks *(optional)*
- [x] 1.1 Perceptron and learning rule (NumPy)
  - Dataset: Iris (UCI)
- [x] 1.2 Simple Multilayer Perceptron (MLP)
  - Dataset: MNIST
- [x] 1.3 Manual backpropagation
- [x] 1.4 Optimization: SGD, Adam, and other optimizers
- [x] 1.5 First MLP with TensorFlow and PyTorch

---



### 🔁 Module 2: Deep Neural Networks (DNN)
> Objective: Train deep networks and apply improvement techniques.

- [x] 2.1 Regularization: Dropout and L2
  - Dataset: Fashion MNIST
- [x] 2.2 Validation and hyperparameter tuning
- [x] 2.3 Techniques to avoid overfitting (BatchNorm, augmentation)
- [x] 2.4 Introduction to MLOps: model and experiment versioning *(for exploration)*
- [x] 2.5 Project: Digit Classifier with GUI (Tkinter)

---



### 📷 Module 3: Convolutional Neural Networks (CNN)
> Objective: Apply deep learning to computer vision.

- [ ] 3.1 CNN Fundamentals (convolutions, pooling, etc.)
  - Dataset: CIFAR-10
- [ ] 3.2 Classic architectures: LeNet, VGG, ResNet
- [ ] 3.3 Transfer Learning (MobileNet/ResNet)
  - Dataset: Oxford Flowers, Dogs vs Cats
- [ ] 3.4 Modern architectures: EfficientNet, Vision Transformers *(for exploration)*
- [ ] 3.5 Project: Real-time classifier with webcam (OpenCV)

---



### 🧾 Module 4: NLP and Recurrent Networks
> Objective: Process text and sequential data.

- [ ] 4.1 Tokenization and embeddings (Word2Vec, GloVe)
  - Dataset: IMDB, Amazon Reviews
- [ ] 4.2 LSTM/RNN for text generation
  - Dataset: Recipes, Shakespeare
- [ ] 4.3 Text classification with RNN/CNN
- [ ] 4.4 Introduction to transformers for NLP *(for exploration)*
- [ ] 4.5 Project: Basic ChatBot (intents + NLP)

---



### 🧠 Module 5: Transformers and Language Models (LLMs)
> Objective: Introduce modern architectures for NLP.

- [ ] 5.1 Attention mechanism and Transformer architecture
- [ ] 5.2 BERT vs GPT comparison and downstream tasks
  - Dataset: SST2, CoNLL-2003
- [ ] 5.3 Fine-tuning with small HuggingFace models
- [ ] 5.4 Introduction to GPT-2 and text generation
- [ ] 5.5 Project: Educational text generator (fine-tune GPT-2)

---



### 🔍 Module 6: Interpretability and Evaluation
> Objective: Understand model decisions.

- [ ] 6.1 Visualization of filters and activations (CNN)
- [ ] 6.2 Interpretability with Grad-CAM and LIME
- [ ] 6.3 SHAP and LIME for text models (NLP)
- [ ] 6.4 Advanced metrics: F1, confusion matrix

---



### 🚀 Module 7: Production and Optimization
> Objective: Take models to production and make them efficient.

- [ ] 7.1 Quantization and pruning (TensorFlow Lite / ONNX)
- [ ] 7.2 Export models to mobile/web (TF.js, CoreML)
- [ ] 7.3 Distributed training, GPU usage (Colab / local)
- [ ] 7.4 Basic MLOps practices *(for exploration)*
- [ ] 7.5 Final project: App with embedded model (classifier or generator)

---


## 📚 Recommended Resources and Datasets

- [HuggingFace Datasets](https://huggingface.co/datasets)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/)
- [OpenML](https://www.openml.org/)

## 🔧 Frameworks and Tools

- `PyTorch`, `TensorFlow`, `Keras`, `scikit-learn`
- `transformers`, `datasets` (HuggingFace)
- `matplotlib`, `seaborn`, `OpenCV`, `Gradio`, `Streamlit`