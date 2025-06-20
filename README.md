# PointSkelNet: A Robust Deep Learning Skeletonization Framework

PointSkelNet is a cutting-edge, universal deep learning framework designed to extract a precise 3D skeleton from a single 2D image of any arbitrary object. By orchestrating a hybrid pipeline that synergizes advanced image processing with a powerful deep neural network, NETISG-XII delivers state-of-the-art skeletonization results suitable for a wide range of applications, including computer graphics, animation, robotics, and biomechanics.


---

## Key Features

-   **Universal Image-to-3D Lifting:** Automatically transforms any 2D image into a 3D point cloud. It uses `rembg` for robust, AI-powered background removal and then applies a distance transform to intelligently infer a third dimension, capturing the object's volumetric shape.
-   **Deep Offset Prediction Network:** At its core, the framework employs a powerful `PointNetOffsetPredictor` model. Trained on the large-scale **RigNet dataset**, this network learns to predict per-point offset vectors, iteratively contracting the entire point cloud towards the object's true medial axis.
-   **Robust Skeleton Topology:** After the deep learning-driven contraction, a final refinement stage applies voxel-based downsampling to identify clean skeleton nodes. A **Minimum Spanning Tree (MST)** algorithm then connects these nodes, forming a coherent and topologically correct skeleton graph.
-   **Interactive 3D Visualization:** The entire pipeline culminates in a dynamic, interactive 3D visualization powered by Open3D. This allows you to inspect the original point cloud, the contracted skeleton points, and the final line set topology from any angle.

---

## Technical Pipeline

The framework operates in a sequential, three-stage pipeline:

1.  **Stage 1: Semantic Lifting**
    -   The input 2D image is processed to remove the background, isolating the foreground object.
    -   A distance transform is calculated on the object's mask. This value becomes the Z-coordinate, lifting the 2D shape into a 3D point cloud.

2.  **Stage 2: Deep Contraction**
    -   The generated point cloud is fed into the pre-trained `PointNetOffsetPredictor`.
    -   The model predicts a 3D offset vector for every point in the cloud.
    -   The points are moved along these vectors in an iterative process, causing the cloud to "contract" and condense around its central skeleton.

3.  **Stage 3: Topology Refinement**
    -   The dense, contracted point cloud is downsampled using voxels to create a sparse set of skeleton nodes.
    -   A Minimum Spanning Tree algorithm connects these nodes based on Euclidean distance, producing the final, clean skeleton structure.

---

## Getting Started

### Prerequisites
-   Python 3.8+
-   PyTorch
-   Open3D
-   A GPU is recommended for training but not required for inference.

### Installation

Clone the repository and install the required dependencies. It is recommended to do this in a virtual environment.
```bash
# Clone the repository
git clone 
cd 

# Install dependencies from setup.py
pip install -e .
```

### Dataset Setup

-   Download the **RigNet Dataset**.
-   Ensure the dataset is placed in a simple, ASCII-only path (e.g., `C:/datasets/RigNet_Kaggle`) to avoid file loading issues.
-   Update the `dataset_path` in `netisg/config.py` to point to your dataset location.

### Training the Model

Run the training script to train the `PointNetOffsetPredictor` on the RigNet dataset. The trained model will be saved in the `/checkpoints` directory.
```bash
python scripts/train.py
```

### Running Inference

Once the model is trained, you can run it on any image to extract its 3D skeleton.
```bash
python scripts/run_netisg.py --input path/to/your/image.jpg
```
An interactive 3D window will appear, displaying the skeletonization result.

---

## Applications

The ability to extract 3D skeletons from 2D images has wide-ranging applications:
-   **3D Animation & Rigging:** Automatically generate a base rig for character animation.
-   **Medical Imaging:** Analyze anatomical structures from scans and images.
-   **Robotics:** Help robots understand object geometry for grasping and manipulation.
-   **Computer Vision:** Serve as a powerful tool for shape analysis and object recognition.

---

## Repository Structure

```
.
├── checkpoints/         # Stores trained model weights (.pth)
├── netisg/              # Core library source code
│   ├── config.py        # All configuration classes
│   ├── data_loader.py   # Robust dataset and dataloader
│   ├── image_processor.py # 2D-to-3D lifting logic
│   ├── model.py         # PointNetOffsetPredictor model definition
│   ├── skeletonizer.py  # Contraction and topology logic
│   └── visualizer.py    # 3D visualization code
├── scripts/             # Executable scripts
│   ├── train.py         # The training script
│   └── run_netisg.py    # The inference script
└── setup.py             # Project setup and dependencies
```
