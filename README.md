# Transfer Learning for Soda Bottle Recognition

This project demonstrates the power of transfer learning in computer vision by applying it to identifying soda bottles. We fine-tune a pre-trained Xception model to achieve high accuracy in recognizing various soda bottle brands.

## Project Overview:

- **Deep Learning in Computer Vision**: Uses advanced techniques to accurately recognize soda bottle images.

- **Transfer Learning Approach**: Applies the Xception model, enhanced through transfer learning, to identify soda bottles.

- **Kaggle Dataset**: Trains and evaluates the model using the [Cola Bottle Identification Dataset from Kaggle](https://www.kaggle.com/datasets/deadskull7/cola-bottle-identification/code).

### Key Components:

- `/app`: Web interface for image upload and prediction.
- `/code/conv_net_comparison.ipynb`: Documentation of neural network comparisons.
- `/data/soda_bottles`: Dataset for model training and validation.
- `/other_files`: Supplementary resources, including weights and test photos.

## Quick Start:

1. Clone the repository:
   `git clone https://github.com/vdrvar/comp_vision_for_soda_bottles.git`

2. Set up the environment with Docker:
cd comp_vision_for_soda_bottles/app
docker build -t flask-image-service .
docker run -p 5000:5000 flask-image-service


3. Access the web service at `http://localhost:5000`.

### How to Use:

- **Home Page**: Upload a photo to see the model's prediction.
- **Results**: Displays predictions with recognized soda bottle types.
- **More Info**: Insights on recognized classes and prediction statistics.

## Make It Your Own:

Tweak, expand, and use this project as a base for further computer vision and transfer learning experiments.

### Please Note:

For development only. Check `conv_net_comparison.ipynb` for model insights.
