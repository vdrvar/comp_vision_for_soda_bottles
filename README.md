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

1. **Clone the repository**:
```
git clone https://github.com/vdrvar/comp_vision_for_soda_bottles.git
```

2. **Navigate to the Project Directory**:
```
cd comp_vision_for_soda_bottles/app
```

3. **Set Up the Virtual Environment** (Optional but recommended):
- For Windows:
  ```
  python -m venv env
  env\Scripts\activate
  ```
- For macOS and Linux:
  ```
  python3 -m venv env
  source env/bin/activate
  ```

4. **Install Dependencies**:
```
pip install -r requirements.txt
```


5. **Run the Application**:
```
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Replace `app.main:app` with the appropriate module path if your FastAPI app instance is defined elsewhere.

6. **Access the Web Service**:
Open your browser and visit [http://localhost:8000](http://localhost:8000) to interact with the application.

### Note:
- The `--reload` flag in the uvicorn command enables automatic reloading of the server when changes are detected in the code. It is useful for development but should be omitted in a production environment.


### How to Use:

- **Home Page**: Upload a photo to see the model's prediction.
 ![image](https://github.com/vdrvar/comp_vision_for_soda_bottles/assets/48907543/e9df039d-aa78-464e-a473-0304214536ef)

- **Results**: Displays predictions with recognized soda bottle types.
  ![image](https://github.com/vdrvar/comp_vision_for_soda_bottles/assets/48907543/c94e6502-4f7a-4f78-96c3-e9fb96b232a3)

- **More Info**: Insights on recognized classes and prediction statistics.
  ![image](https://github.com/vdrvar/comp_vision_for_soda_bottles/assets/48907543/47f033d1-756e-4025-ab44-4c7c34758ff1)
  ![image](https://github.com/vdrvar/comp_vision_for_soda_bottles/assets/48907543/3b2fdba7-6cb8-4906-acfc-d6161e8e5484)



## Make It Your Own:

Tweak, expand, and use this project as a base for further computer vision and transfer learning experiments.

### Please Note:

For development only. Check `conv_net_comparison.ipynb` for model insights.
