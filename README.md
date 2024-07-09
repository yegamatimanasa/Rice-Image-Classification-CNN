# Rice-Image-Classification-CNN
This project focuses on the classification of different rice varieties using machine learning models, particularly Convolutional Neural Networks (CNNs). The rice varieties included in the dataset are Arborio, Basmati, Ipsala, Jasmine, and Karacadag. The dataset comprises 75,000 images, with 15,000 images for each rice variety. Additionally, the dataset includes various morphological, shape, and color features.

The classification models implemented in this project include Artificial Neural Networks (ANN), Deep Neural Networks (DNN), and Convolutional Neural Networks (CNN). Among these, the CNN model achieved an impressive accuracy rate of 100% in classifying the rice varieties. The results highlight the efficacy of these models in accurately distinguishing between different types of rice grains.

## Dataset
The dataset used for training and evaluation is available on Kaggle: [Rice Image Dataset](https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset/code). It includes images organized into the following categories:

- Arborio
- Basmati
- Ipsala
- Jasmine
- Karacadag

## Requirements
- Python 3.x
- TensorFlow
- Keras
- OpenCV
- NumPy
- Pandas
- Matplotlib

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/Rice-Classification-using-CNN.git
    ```
2. Navigate to the project directory:
    ```bash
    cd Rice-Classification-using-CNN

## Usage
1. Download the dataset from [Kaggle - Rice Image Dataset](https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset/code) and place it in the project directory.
2. Open the Jupyter notebook:
    ```bash
    jupyter notebook Rice_Dataset.ipynb
    ```
3. Run the cells in the notebook to train and evaluate the CNN model.

## Results
The high accuracy rate of 99.86% achieved on the rice detection dataset is a result of meticulous model development, including data preprocessing, hyperparameter tuning, regularization, transfer learning, data balancing, cross-validation, and ensemble learning. The selection of an optimal CNN architecture, careful hyperparameter tuning, and regularization techniques like dropout and batch normalization were crucial. Transfer learning with pre-trained models and data balancing ensured accurate predictions across all classes. Cross-validation and ensemble learning further enhanced the model's robustness and accuracy.

## Contributing
Contributions are welcome! Please fork the repository and create a pull request with your changes.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
