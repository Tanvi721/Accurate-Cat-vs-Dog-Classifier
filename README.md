# 🐱🐶 Cat & Dog Image Classification

## 📌 Project Overview
This project implements a **Cat vs Dog Image Classification system** using **Deep Learning**.  
The model is trained to automatically identify whether a given image contains a **cat** or a **dog**.

This is a classic **binary image classification** problem and a beginner-friendly demonstration of **Convolutional Neural Networks (CNNs)**.

The project is implemented in **Python** using **TensorFlow / Keras**.

---

## 🎯 Objectives
- Build a CNN model for binary image classification  
- Train the model on cat and dog image datasets  
- Evaluate model performance using accuracy  
- Predict class labels for new/unseen images  
- Understand the end-to-end deep learning workflow  

---

## 🧠 Technologies Used
- **Programming Language:** Python  
- **Libraries & Frameworks:**  
  - TensorFlow  
  - Keras  
  - NumPy  
  - Matplotlib  
  - OpenCV / PIL  
- **Platform:** Jupyter Notebook  

---

## 📂 Project Structure

Cat_Dog_Classification/
│
├── Cat_dag_classification.ipynb
├── dataset/
│   ├── train/
│   │   ├── cats/
│   │   └── dogs/
│   └── test/
│       ├── cats/
│       └── dogs/
└── README.md



---

## ⚙️ Workflow
1. **Data Loading & Preprocessing**
   - Image resizing  
   - Normalization  
   - Data augmentation  

2. **Model Building**
   - Convolutional layers  
   - MaxPooling layers  
   - Dense (Fully Connected) layers  
   - Sigmoid activation for binary classification  

3. **Model Training**
   - Loss Function: Binary Crossentropy  
   - Optimizer: Adam  
   - Metric: Accuracy  

4. **Evaluation**
   - Training and validation accuracy  
   - Loss visualization  

5. **Prediction**
   - Predicts whether the input image is a **Cat 🐱** or **Dog 🐶**

---

## 📊 Model Performance
- Achieves **high validation accuracy**
- Data augmentation helps reduce overfitting
- Can be improved further using transfer learning

---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository

git clone <repository-url>
cd Cat_Dog_Classification

### 2️⃣ Install Required Libraries
pip install tensorflow numpy matplotlib opencv-python

### 3️⃣ Run the Notebook
Open Jupyter Notebook and run:
        - Cat_dag_classification.ipynb

### 🖼️ Sample Output

+ **Input**: Image of a cat or dog
<img width="1920" height="1080" alt="Screenshot (55)" src="https://github.com/user-attachments/assets/15b7ca8f-edab-405f-b463-d3fe7338ec6f" />
<img width="1920" height="1080" alt="Screenshot (54)" src="https://github.com/user-attachments/assets/ec6ffa1a-66fe-4305-8be4-1103af2ac53c" />
<img width="1920" height="1080" alt="Screenshot (53)" src="https://github.com/user-attachments/assets/debe3d06-8b1a-41e3-96ad-9e7d5ea634fe" />

+ **Output**:
Cat 🐱 or Dog 🐶 with confidence score

# 🔮 Future Enhancements

+ Use Transfer Learning (VGG16, ResNet, MobileNet)

+ Build a Web App using Streamlit or Flask

+ Increase dataset size

+ Deploy the model on cloud

# 👩‍💻 Author
+ **Tanvi Barve**
+ **Aspiring Data Scientist | Machine Learning Enthusiast**

# 📜 License

This project is created for educational purposes.
You are free to use, modify, and distribute it.

### ⭐ If you like this project, don’t forget to star the repository!



