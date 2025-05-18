#  AI agent - LangChain + GPT functions + HuggingFace Spaces + Streamlit

## Overview

AI Document Analyst is a Streamlit web application that enables users to ask questions about their own PDF documents using a Retrieval-Augmented Generation (RAG) pipeline backed by LLMs (via OpenRouter), vector search (Chroma), and embeddings (Hugging Face models).

This project is designed for robust PDF processing, efficient vector search, scalable deployment (via Docker & Hugging Face Spaces), and modularity.

--- https://youtu.be/Oa1ajCfaZE0

## Assumptions & Scope

### What I included:
- Hand tracking using MediaPipe
- ROI extraction based on MCP–PIP alignment
- YOLOv8n-based ring detection
- Real-time detection via webcam
- Streamlit app for easy testing and sharing

### What I excluded (due to time constraints):
- Earrings, clothing, full-body tracking

### Assumptions:
- Rings are most commonly worn on index, middle, and ring fingers
- Input could be from a standard webcam, a phone video, external cam.
- Performance must be real-time (low latency), and work under normal lighting

---

## Architecture & Pipeline

### System Architecture
```plaintext
Video Frame
   ↓
HandTracker (MediaPipe)
   ↓
Landmarks → extract_aligned_roi_finger()
   ↓
Perspective-aligned ROI per finger
   ↓
RingDetectorYOLO → YOLOv8 detection
   ↓
Overlay (OpenCV or Streamlit UI)
```

---

### Pipeline

1. Hand Tracking – hand_tracker.py

    Uses MediaPipe’s 21 landmarks per hand

    Lightweight and very accurate for fingers

2. ROI Extraction – region_finger.py

    Computes the midpoint between MCP and PIP joints

    Uses vector math to build a rectangle aligned with finger direction

    Applies cv2.getPerspectiveTransform() to obtain clean ROIs

3. (First design) Ring Detection by colors – ring_detector_colors.py

    Based on colors (Heuristic Detection)

    Works on the small area and has very low latency

4. (Main design) Ring Detection by YOLO – ring_detector_yolo.py

    Loads a YOLOv8n model trained specifically on ring data

    Works on the small aligned ROI (fast and focused)

    Uses confidence threshold = 0.55 for balance

5. Interfaces

    main_ring_detector_yolo.py: OpenCV-based real-time detection

    app.py: Streamlit web app to test easily in browser

---

### Design Decisions & Alternatives

    | Choise               | Why? / Why not?                                                                            |
    |----------------------|--------------------------------------------------------------------------------------------|
    | Mediapipe Hands      | Precise finger/hand detection, Ideal for mobile first                                      |
    | MCP–PIP based ROI    | Aligns with where rings are worn, better than fixed box                                    |
    | Perspective warping  | Keeps ROI aligned despite hand rotation, improves detection                                |
    | Heuristic Detection  | It depends a lot on the color of the rings, it cannot be a single model for several rings. |
    | Custom ring model    | Pretrained YOLOs don’t detect rings — I trained my own                                     |
    | YOLOv8n Detection    | Accurate, efficient, and easy to train/fine-tune                                           |
    | YOLOv8n Tracking     | Too much latency for this case                                                             |
    | Streamlit            | Easy to test and demo without needing local setup                                          |
    | OpenCV               | Useful for debugging and real-time experimentation                                         |


---

## Iterations & Learnings

Throughout the project, I explored several approaches, learned from failures, and refined the system through multiple iterations. Here's a breakdown of my process:

### 🔹 Hands Tracking 

**Approach:**  

I began with a simple and classic approach using OpenCV:
- I used Mediapipe solutions hands because we needed to track the hands too.
- Fast way to track hand points.

**Why I tried this:**  

This method is fast and doesn’t require training data, just read how to use the landmarks.

**Result:**  
- Worked really well, it didn't take too long

**Learning:**  

https://github.com/user-attachments/assets/e1dc3017-fae5-4062-9adf-3b18c2bc83aa

---

### 🔹 Static Rectangular ROI per Finger

**Approach:**  
I used MediaPipe landmarks to extract a box between MCP and PIP joints of the finger (the typical ring area).  
The ROI was horizontal and fixed in shape.

**Why I tried this:**  
I assumed the region between MCP and PIP would cover most rings, and I wanted a simple pipeline to pass consistent ROIs into detection.

**Result:**  
- Boxes didn’t follow finger rotation.
- When the hand tilted, ROIs missed the actual ring.
- Detections were inconsistent and unreliable.

**Learning:**  
A box without considering finger orientation causes misalignment, especially in motion. ROI must rotate with the finger to stay accurate.

https://github.com/user-attachments/assets/36df7985-65f2-4bbd-bf9c-d04824c7b41e


---

### 🔹 Rotated ROI using getRotationMatrix2D - warpAffine

**Approach:**  
I computed the angle between MCP and PIP to rotate the ROI with `cv2.getRotationMatrix2D` and `warpAffine`.

**Why I tried this:**  
I wanted the ROI to follow the finger's angle for better localization and to improve detection robustness.

**Result:**  
- I didn't understand well the maths.
- The region was not positioned well (like the hand down in front of the camera) the ROI went to the MCP and did not fit where the ring is used.

**Learning:**  
Needed something more precise.
Improve the maths, specially the orientation for vectors and how to draw the rectangle

![vectors](https://github.com/user-attachments/assets/cf6fb72d-815c-4ec9-bfc4-0b0084388ceb)

Then I needed to understand how to get the points of the rectangle, what calculations did I have to make?
Everything had to be based on the direction of the vector.

Top-Left = cx - lx - wx, cy - ly - wy

Top-Right = cx - lx + wx, cy - ly + wy

Bottom-Right = cx + lx + wx, cy + ly + wy

Bottom-Left = cx + lx - wx, cy + ly - wy


![points_box](https://github.com/user-attachments/assets/91bb7299-5746-45ec-b6d2-24d1d684ad8b)

Then I realized I can't send this image to the model because it is not a good perspective so that's why I use warpPerspective

![perspectiva del rectangulo](https://github.com/user-attachments/assets/f7f8c7d3-f940-45c5-a5d8-9261da884ec6)

Then to draw the box in a good perspective I needed to understand how to draw the rectangle based on point just joining the points.

![box perspectiva](https://github.com/user-attachments/assets/8dd45d6f-9912-4a92-958a-98ef60b13d1a)

---

### 🔹 Perspective-Corrected ROI using `warpPerspective`

**Approach:**  
I created a full rectangle aligned with the finger direction using vector math, and extracted a perspective-corrected region with `cv2.getPerspectiveTransform`.

**Why I tried this:**  
To ensure the ROI remains aligned regardless of hand orientation or rotation — keeping it consistent for any detection model.

**Result:**  
- Great improvement in ring localization.
- ROIs were perfectly aligned to fingers.
- Compatible with both heuristics and YOLO input.

**Learning:**  
This was the turning point. Properly aligned ROIs significantly improve consistency and allow more generalizable detection logic.

https://github.com/user-attachments/assets/aac36c9a-137a-4bf7-a9d5-47b978a09608


---

### 🔹 Heuristic Detection (Colors)

**Approach:**  
I began with a simple and classic approach using OpenCV:
- Converted finger regions to HSV.
- Used color thresholds for black and blue.

**Why I tried this:**  
This method is fast and doesn’t require training data and thought it would be good for a quick prototype.

**Result:**  
- Worked only under ideal lighting.
- Many false negatives due to combinations of colors

**Learning:**  
Heuristics are limited when dealing with lighting, and style diversity. I needed a learned model.

https://github.com/user-attachments/assets/819c478f-79e1-41dd-b6ae-6c89d6449cdc

https://github.com/user-attachments/assets/13d5014b-77f7-4dc3-bc93-14a92619fc29


---


### 🔹 YOLOv8n Model Integration

**Approach:**  
I trained a custom YOLOv8n model using annotated ring images via Roboflow.

**Why I tried this:**  
Heuristics and color filters failed in 50% cases. YOLO allows shape, context, and general appearance to be learned from data.

**Result:**  
- Used Yolo v8 NANO, small model for mobile - first.
- Very strong performance on various hand poses and lighting.
- Real-time.
- Low false positives, good generalization.
- I only captured 500 images with and without rings.
- I used Roboflow to label and augmentation so I got like 1300 images to train the model

**Learning:**  
A trained detector is essential for visual tasks involving variable appearance and movement. But it still needs **clean, well-aligned inputs**, which I had prepared in earlier steps.

Dataset is really important, also labeling takes so much time.

https://github.com/user-attachments/assets/ef994c07-72f2-44ec-b25c-0afc6e1e3d75



---

### 🔹 Streamlit Integration

**Approach:**  
I created a Streamlit app (`app.py`) so the system can be tested easily from the browser.

**Why I tried this:**  
To make the system demo-friendly and user-accessible — ideal for a startup or team looking to share results quickly.

**Result:**  
- Easy testing.
- Clear visual feedback.

**Learning:**  
Presentation matters — making your tools usable for others adds serious value to any system.

---

### 🔹 Demo 

- Click on the videos to watch

[![Demo Part 1](https://img.youtube.com/vi/q1fTYXFudIs/hqdefault.jpg)](https://www.youtube.com/watch?v=q1fTYXFudIs)

[![Demo Part 2](https://img.youtube.com/vi/GGSkEnNGisc/hqdefault.jpg)](https://www.youtube.com/watch?v=GGSkEnNGisc)

---

### 🎯 General Learnings

- **Iterative development matters.** Every step taught me something valuable.
- **Don’t over-engineer too early.** My best results came when I simplified, then rebuilt based on feedback.
- **Precision in ROI design is crucial.** Especially in mobile-first applications with gesture input.
- **Combining ML with geometric reasoning** (YOLO + landmark-based ROIs) provides robust, scalable results.



## Failure Cases & Limitations / Technical Trade-offs

- Accuracy depends on the objective. In this case is better if we can detect the rings despite the hand is far from the camera.
- Detection needs to be really fast, that's why I just analyze a region of the finger.
- Need more dataset to improve detection but mostly types of rings.

---

## Future Improvements

### Data Collection

- Focus on collecting more types of rings and perspectives where the model will be use.
- Add detection for earrings and necklaces.
- Improve robustness with larger datasets.

### Use Mediapipe body

- Track the full body with a fast model.

### 3D Modelling

- Integrate with 3D modeling for product visualization.

### Define on which platform it will be launched

- Build a full mobile-first UI.
- Used a smaller o bigger model depends on the platform.

## 🔮 3D & Synthesis Module (In Progress)

To expand the project beyond ring detection, I'm now developing a 3D and video synthesis extension, aligned with Thuli’s product vision.

This includes:
- Generating novel views of hands with rings using NeRF or Gaussian Splatting.
- Rigging a 3D hand in Blender with ring assets and animating it from MediaPipe landmarks.
- Exploring ways to overlay 3D rings onto tracked hands in real time (future AR use case).

All experiments will be documented and included in `/3d_synthesis/`.


---

## Alignment with Thuli Studios Vision

Thuli Studios’ mission to build AI-native, India-first products directly inspired my approach.

Designed for diverse hand tones and dynamic gestures.

Real-time performance makes it ideal for DYLA-style marketplaces.

Can work in influencer video previews, mirror-based try-ons (when the model detected the rings in my screen).

Easily extendable to other jewelry forms.

---

## Summary

I approached this challenge as a chance to show:

That I can think clearly and structure a system

That I can balance creativity with pragmatism

That I enjoy building and iterating under constraints

I’m proud of what I built — not because it’s perfect, but because it’s practical, modular, and real.

Thanks for this opportunity. I hope to continue building with you.


- Wesley




