---
library_name: pytorch
tags:
- sign-language
- computer-vision
- video-classification
- isl
- transformer
- mediapipe
- deep-learning
pipeline_tag: video-classification
license: mit
language:
- en
- hi
metrics:
- accuracy
base_model: mobilenet_v3_large
---

# Model Card for Indian Sign Language Recognition System

## Model Details

### Model Description

This model is a **hierarchical two-stream transformer** designed for real-time **Indian Sign Language (ISL)** recognition. It utilizes a novel gating architecture to decompose the classification problem into semantic groups, improving both accuracy and inference speed. The system integrates computer vision (MediaPipe) with deep learning (PyTorch) to process both visual frames and pose landmarks simultaneously.

- **Developed by:** Abhay Gupta
- **Model type:** Hierarchical Two-Stream Transformer (Visual + Pose)
- **Language(s) (NLP):** English (Output), Hindi/Regional (Translation)
- **License:** mit
- **Resources for more information:**
  - [GitHub Repository](https://github.com/Abs6187/DynamicIndianSignLanguageDetection)

### Model Sources

- **Repository:** https://github.com/Abs6187/DynamicIndianSignLanguageDetection


## Uses

### Direct Use

The model is intended for:
- Real-time ISL-to-text translation.
- Accessibility tools for the deaf and hard-of-hearing community.
- Educational platforms for learning ISL.

### Downstream Use

- Integration into video conferencing tools.
- Public service kiosk interfaces.
- Mobile applications for sign language interpretation.

### Out-of-Scope Use

- Recognition of other sign languages (ASL, BSL, etc.) without retraining.
- High-stakes medical or legal interpretation without human oversight.

## Bias, Risks, and Limitations

- **Lighting Conditions:** Performance improves with good lighting; extreme low light may affect accuracy (though HSV augmentation helps).
- **Occlusions:** Heavy occlusion of hands may degrade performance, despite robust interpolation methods.
- **Vocabulary:** Currently limited to the trained vocabulary (60+ signs generally, specific checkpoints may vary).

## How to Get Started with the Model

Use the `SignLanguageInference` class to load the model and run predictions on video files.

```python
from infer import SignLanguageInference

# Initialize
inference = SignLanguageInference(
    model_path='best_model.pth',
    metadata_path='metadata.json'
)

# Predict
result = inference.predict('video_sample.mp4')
print(f"Predicted Sign: {result['top_prediction']['class']}")
```

## Training Details

### Training Data

- **Source:** Custom captured dataset of 2000+ video samples.
- **Classes:** 80+ ISL signs (hierarchically grouped).
- **Participants:** 10+ diverse signers in indoor/outdoor environments.

### Preprocessing

- **MediaPipe:** Extracts 154-dimensional landmark vectors (Hands + Pose).
- **Augmentation:** 8 strategies including background blur, color shifts, and geometric-preserving spatial transforms.
- **Normalization:** Translation, scale, and Z-score standardization.

### Training Procedure

- **Architecture:** 
    - **Visual Stream:** MobileNetV3 backbone -> Transformer Encoder.
    - **Pose Stream:** 1D Convolutions -> Transformer Encoder.
    - **Fusion:** Gated mechanism combinig visual and pose embeddings.
- **Loss Function:** Focal Loss (to handle class imbalance).
- **Optimization:** Mixed precision training.

## Evaluation

### Testing Data, Factors & Metrics

### Results

| Configuration | Accuracy | Inference Time | Memory |
|--------------|----------|----------------|---------|
| **Monolithic Model** | 88.3% | 150ms | 50MB |
| **Hierarchical (Ours)** | **93.8%** | **95ms** | **30MB** |

- **Group 0 (Pronouns):** 96.2% Accuracy
- **Group 1 (Objects):** 93.4% Accuracy
- **Group 2 (Actions):** 91.8% Accuracy

## Environmental Impact

- **Hardware Type:** Trained on GPUs (Specifics N/A)
- **Inference Efficiency:** Optimized for CPU inference (approx 95ms/video), suitable for edge deployment.

## Technical Specifications

- **Input:** 
    - Video Frames: (T, 3, H, W)
    - Landmarks: (T, 154)
- **Frameworks:** PyTorch 2.0+, TensorFlow 2.x, MediaPipe


---