# LiDAR_Motion_Compensation

LiDAR motion compensation and 3D object reconstruction using timestamped point clouds and bounding box interpolation.

# 🚗 LiDAR Motion Compensation & 3D Reconstruction

This project reconstructs a **clean 3D representation of a moving vehicle** from distorted LiDAR scans by compensating for motion during data acquisition.

It uses **timestamped point clouds and bounding box annotations** to correct motion artifacts and recover the true geometry of the object.

---

## 🚀 Key Features

- ⏱️ Timestamp-based point correction  
- 🔄 Rotation interpolation using SLERP  
- 📦 Bounding box filtering for object isolation  
- 🌍 Transformation into global coordinate system  
- 📊 Accumulation of multi-frame point clouds  
- 🎨 3D visualization with Open3D  

---

## 🧠 Problem Statement

LiDAR sensors scan over time (~0.1s), causing:
- Motion distortion (ego vehicle + object motion)
- Warped object shapes in point clouds

This project solves:
> Reconstructing the **true shape of a moving car** by compensating for motion.

---

## 🛠️ Methodology

### 1. Object Extraction
- Use labeled bounding boxes
- Filter points belonging to `Passenger_Car:1`

### 2. Motion Interpolation
- Linear interpolation for translation
- SLERP (Spherical Linear Interpolation) for rotation

### 3. Deskewing
- Transform points into object-local frame
- Correct motion distortion

### 4. Global Alignment
- Map all points into a common reference frame
- Accumulate across multiple frames

---

## 📂 Project Structure
lidar-motion-compensation/
│
├── main.py # Full pipeline
├── labels.json # Bounding box annotations
├── pcd/ # Input point cloud frames
├── compensated.png # Expected output visualization
├── car_capture.png # Generated output
└── README.md

---

## ▶️ Usage

```bash
python main.py
```
---

📊 Results

✔️ Distorted point cloud → Clean reconstructed car

✔️ Multi-frame accumulation improves shape quality

---
📈 Visualization

-Colored point cloud based on X-axis

  -Interactive Open3D viewer
  
  -Automatic screenshot generation
  
---
⚠️ Assumptions

  -Motion is linear between frames
  
  -Rotation changes smoothly (SLERP valid)
  
  -Accurate bounding box annotations available
  
---
💡 Applications

  -Autonomous driving perception
  
  -SLAM and mapping systems
  
  -Robotics and sensor fusion
  
  -3D object tracking
  
---
🧰 Tech Stack

  -Open3D
  
  -NumPy

  -SciPy (Rotation, SLERP)
  
  -Matplotlib
  
---
👨‍💻 Author

Rushikesh Sonawane
📧 rushikesh.sonawane16598@gmail.com

🔗 LinkedIn: https://www.linkedin.com/in/rushikesh-sonawane2025
