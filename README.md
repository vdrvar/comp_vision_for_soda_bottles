# Flask Image Recognition Web Service

This repository houses a Flask web service designed for image recognition, featuring a user-friendly interface, pre-trained Xception model, and statistics tracking for successful predictions. Beyond the app, it includes resources for model comparison, datasets, and auxiliary files.

Key Components:

/app Directory: Contains the Flask web service for image recognition. Users can upload images for prediction and view statistics.

conv_net_comparison.ipynb: A Jupyter Notebook where two convolutional neural network architectures are compared, leading to the selection of the model used in the app.

/Soda Bottles Directory: The dataset used for training and evaluation.

/TVT_split Directory: Dataset split into training, validation, and test sets. Freshly split with one run of the notebook.

Auxiliary Files: Various files supporting the project, including model weights, requirements.txt, and Dockerfile.

1) Clone the Repository:
git clone https://github.com/your-username/flask-image-recognition.git

2) Navigate to the Project Directory:
cd flask-image-recognition

3) Build the Docker Image:
docker build -t flask-image-service .

4) Run the Docker Container:
docker run -p 5000:5000 flask-image-service

5) Access the Web Service:
Open http://localhost:5000 in your web browser to interact with the image recognition web service.

Usage:

Explore the /app directory for the Flask web service.

Review conv_net_comparison.ipynb for insights into the model comparison process and weight training.

The /Soda Bottles directory contains the dataset used for training and evaluation.

The /TVT_split directory provides the dataset split into training, validation, and test sets.

Customize and extend the project as needed.


Note:

This is a development server; for production, consider using a production-ready WSGI server.
For more details on the model selection process, refer to conv_net_comparison.ipynb.
Feel free to adjust the description and instructions based on any additional details or features you'd like to highlight.
