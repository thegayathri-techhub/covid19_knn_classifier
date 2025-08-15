# 🩺 COVID-19 Chest X-Ray Classification using KNN

A **machine learning project** that classifies chest X-ray images into **COVID-19 Positive**, **Pneumonia**, or **Normal** categories using the **K-Nearest Neighbors (KNN)** algorithm.  
This project leverages the **COVID-19 Radiography Dataset** to aid in rapid and reliable medical screening.

---

## 📌 Features
- Classifies chest X-ray images into:
  - 🦠 COVID-19 Positive
  - 🌬 Pneumonia
  - ✅ Normal
- Uses **KNN** for classification.
- Image preprocessing & feature extraction.
- Detailed performance evaluation with **confusion matrix** and accuracy scores.
- Command-line based execution.

---

## 📂 Dataset
We use the **COVID-19 Radiography Database**:
- **Source:** [Kaggle - COVID-19 Radiography Database](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database)
- **Contents:**
  - `COVID/` → COVID-19 positive cases
  - `PNEUMONIA/` → Pneumonia cases
  - `NORMAL/` → Healthy cases
  - `Lung_Opacity/` → Other lung-related cases
- Each category contains chest X-ray images and corresponding segmentation masks.

📥 **Download Instructions:**
1. Go to the [dataset page](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database).
2. Download the dataset as a ZIP file.
3. Extract it to your project folder, e.g.:
dataset/
COVID/
PNEUMONIA/
NORMAL/
Lung_Opacity/

---

## 🛠 Tech Stack
- **Programming Language:** Python 3.x
- **Libraries:**
- `numpy`
- `opencv-python`
- `scikit-learn`
- `matplotlib`
- `pandas`

---

## 🚀 Installation & Setup

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/<your-username>/covid19_knn_classifier.git
cd covid19_knn_classifier
```

2️⃣ Create Virtual Environment
```python -m venv venv```

3️⃣ Activate Virtual Environment

Windows:
```venv\Scripts\activate```

Mac/Linux:
```source venv/bin/activate```

4️⃣ Install Dependencies
```pip install -r requirements.txt```

5️⃣ Add the Dataset

Download from Kaggle and place in the dataset/ folder.

▶️ Running the Project

Run the classifier script:
```python cnn_knn_covid.py```

📊 Example Output

Accuracy: 93.5%
Confusion Matrix:
[[100   2   1]

 [  3  95   5]
 
 [  1   4  97]]
 
(Example only — actual results may vary depending on dataset split.)

📌 Future Enhancements
Add GUI for easier image upload & classification.

Integrate deep learning (CNN) for feature extraction before KNN.

Deploy as a Flask web app for online access.

📜 License
This project is licensed under the MIT License — free to use and modify.

👨‍💻 Author
Your Name
🔗 GitHub: thegayathri-techhub
