# Flask Image Recognition Web Service

This repository houses a Flask web service designed for image recognition, featuring a user-friendly interface, pre-trained Xception model (using a transfer learning approach), and statistics tracking for successful predictions. Beyond the app, it includes resources for model comparison, datasets, and auxiliary files.

### Kaggle Dataset:

For the project, we utilized the [Cola Bottle Identification Dataset on Kaggle](https://www.kaggle.com/datasets/deadskull7/cola-bottle-identification/code), which contains images of various cola bottles for training and evaluation purposes.

### Key Components:

/app Directory: Contains the Flask web service for image recognition. Users can upload images for prediction and view statistics.

/code/conv_net_comparison.ipynb: A Jupyter Notebook where two convolutional neural network architectures are compared, leading to the selection of the model used in the app.

/data/soda_bottles Directory: The dataset used for training and evaluation.

/other_files: Various files supporting the project, e.g. weights and test photos.

### Instructions:

1) Clone the Repository:
git clone https://github.com/vdrvar/comp_vision_for_soda_bottles.git

2) Navigate to the Project Directory:
cd comp_vision_for_soda_bottles/app

3) Build the Docker Image:
docker build -t flask-image-service .

4) Run the Docker Container:
docker run -p 5000:5000 flask-image-service

5) Access the Web Service:
Open http://localhost:5000 in your web browser to interact with the image recognition web service.

6) Use the photos in /other_files/test_photos to test the image recognition functionality of the service.

### Usage:

The app is launched on the Home page, where you can upload an arbitrary photo:
![image](https://github.com/vdrvar/comp_vision_for_soda_bottles/assets/48907543/f6c66c9f-dbd2-450d-b005-c22e80449041)

When you click on "Predict", you get the probabilities that the bottle in the photo belongs to the respective classes:
![image](https://github.com/vdrvar/comp_vision_for_soda_bottles/assets/48907543/229790b4-7f40-4eaa-a0ec-52deb064887f)

You can check what classes are available on the Info page:
![image](https://github.com/vdrvar/comp_vision_for_soda_bottles/assets/48907543/ff669aa0-847e-4b83-93d8-e43b921e798a)

Finally, you can also check the prediction history from the last 10 days on the Statistics page:
![image](https://github.com/vdrvar/comp_vision_for_soda_bottles/assets/48907543/e16b2d40-f2c2-4351-b099-b61a6705bebd)


Customize and extend the project as needed.


### Note:

This is a development server; for production, consider using a production-ready WSGI server.
For more details on the model selection process, refer to conv_net_comparison.ipynb.
Feel free to adjust the description and instructions based on any additional details or features you'd like to highlight.
