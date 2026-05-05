# 🚀 Real-Time Face Recognition Monitoring System

A real-time face recognition system that detects and identifies suspects using a webcam feed. The system uses deep learning-based embeddings to match faces and logs detections in a database with a live dashboard.

---

## 🧠 Features

- Real-time face detection using OpenCV  
- Face recognition using DeepFace (ArcFace model)  
- Multi-frame verification to reduce false positives  
- MongoDB integration for storing embeddings and logs  
- Alert logging with timestamp, confidence, and image  
- Live dashboard with video stream and logs  

---

## 🛠 Tech Stack

- Python  
- DeepFace (ArcFace)  
- OpenCV  
- Flask  
- MongoDB (PyMongo)  
- NumPy, SciPy  

---

## 📂 Project Structure
face-recognition-monitoring/
│
├── app.py # Main Flask app (real-time detection)
├── generate_embeddings.py # Generate embeddings from dataset
├── match.py # Matching logic (distance, margin)
├── mongo.py # MongoDB connection
├── requirements.txt
├── README.md
├── .gitignore
│
├── templates/
│ └── index.html # Frontend dashboard
│
├── static/
│ ├── style.css # Styling
│ └── script.js # (optional JS)
│
├── dataset/
│ ├── person1/
│ │ ├── img1.jpg
│ │ └── img2.jpg
│ ├── person2/
│ │ ├── img1.jpg
│ │ └── img2.jpg
│ └── .gitkeep
│
└── alerts/ # Saved detection images (auto-created)


---

## ⚙️ Installation & Setup

### 1. Clone the Repository

git clone https://github.com/ashwani2004/Real-Time-Suspect-Monitoring.git

cd face-recognition-monitoring


---

### 2. Install Dependencies

pip install -r requirements.txt


---

### 3. Start MongoDB
Make sure MongoDB is installed and running:

mongod


---

### 4. Prepare Dataset

Create a dataset folder with this structure:


dataset/
person_name/
image1.jpg
image2.jpg


👉 Use multiple images per person for better accuracy.

---

### 5. Generate Embeddings

python generate_embeddings.py


This will:
- Detect faces
- Convert them into embeddings
- Store them in MongoDB

---

### 6. Run the Application

python app.py


---

### 7. Open Dashboard

http://127.0.0.1:5000


---

## 🔁 Workflow

1. Add images to dataset  
2. Generate embeddings  
3. Start webcam  
4. Detect face in real-time  
5. Convert to embedding  
6. Compare with stored embeddings  
7. Verify using consecutive frames  
8. Log detection  
9. Display on dashboard  

---

## 🧪 Matching Logic

- Cosine similarity for comparison  
- Centroid + individual sample embeddings  
- Threshold-based filtering  
- Margin-based validation  
- Consecutive frame verification  

---

## 📊 Output

- Live webcam feed  
- Bounding box around face  
- Name + confidence score  
- Detection logs with image and timestamp  

---

## ⚠️ Challenges

- False positives  
- Lighting and angle variations  
- Threshold tuning  
- Real-time performance optimization  

---

## 🔮 Future Improvements

- MongoDB Atlas (cloud database)  
- Multi-camera support  
- Authentication system (admin login)  
- Real-time alerts (Email/SMS)  
- Mobile app integration  

---

## 📚 References

- DeepFace Documentation  
- OpenCV Documentation  
- MongoDB Documentation  

---

## 👨‍💻 Author

Ashwani Kumar Bhatnagar

---

## ⭐ Notes

- Dataset is not included in the repository  
- Add your own images in the dataset folder  
- Ensure `templates/index.html` exists, otherwise Flask will throw error  
