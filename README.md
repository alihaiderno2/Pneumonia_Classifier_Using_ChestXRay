Chest X-Ray Pneumonia ClassifierAn end-to-end Deep Learning application that classifies chest X-ray images into Normal or Pneumonia categories. Built using a custom Convolutional Neural Network (CNN) in TensorFlow/Keras and served via an interactive web interface using Streamlit.🚀 FeaturesCustom CNN Architecture: Optimized binary classifier utilizing convolutional layers, max-pooling, dropout for regularization, and manual tensor handling.Explainable AI (XAI): Integrated Grad-CAM (Gradient-weighted Class Activation Mapping) using tf.GradientTape to generate visual heatmaps, showing exactly where the model looks to make its decision.Production Ready: Includes a clean Streamlit UI, structured model saving workflow (.h5), and isolated virtual environments.📁 Project StructurePlaintextDeep_Learning_Projects/
├── .devcontainer/           # Dev container configuration
├── venv/                    # Python virtual environment
├── dataset/                 # Train/Test/Val splits (ignored in git)
│   ├── train/
│   └── test/
├── pneumonia_best_model.h5  # Trained Keras model weights
├── explainability.py        # Core XAI / Grad-CAM script
├── app.py                   # Streamlit web application interface
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
🛠️ Tech Stack & DependenciesCore ML: TensorFlow (CPU optimized), NumPy, H5pyComputer Vision & XAI: OpenCV (opencv-python-headless), Matplotlib, PillowWeb Interface: Streamlit💻 Installation & Setup1. Clone the RepositoryBashgit clone https://github.com/alihaiderno2/Deep_Learning_Projects.git
cd Deep_Learning_Projects
2. Set Up Virtual EnvironmentOn Windows:Bashpython -m venv venv
.\venv\Scripts\activate
On Linux/Ubuntu:Bashpython3 -m venv venv
source venv/bin/activate
3. Install DependenciesBashpip install --upgrade pip
pip install -r requirements.txt
🔬 Explainable AI (Grad-CAM)To analyze what the model has learned and verify it isn't "cheating" on background artifacts, run the diagnostic explainability script:Bashpython explainability.py
This script hooks into the final convolutional layer, computes gradients relative to the top predicted class, and overlays a jet-color heatmap over the lung regions.🌐 Running the Web App LocallyLaunch the Streamlit app to test predictions through a graphical user interface:Bashstreamlit run app.py
📊 Model & Evaluation MetricsInput Resolution: $64 \times 64 \times 3$ (RGB)Optimization Strategy: Adam Optimizer with Binary Cross-Entropy Loss.Target Metrics: High priority placed on Sensitivity (Recall) to ensure zero false negatives in medical diagnostics.
