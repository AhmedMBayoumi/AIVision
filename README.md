# AI Vision Assistant

![AI Vision Assistant Logo](https://placehold.co/600x300/1e293b/ffffff?text=AI+Vision+Assistant)

**AI Vision Assistant** is a powerful Android application designed to aid visually impaired individuals by leveraging state-of-the-art machine learning models. It transforms a standard smartphone camera into an intelligent eye, providing real-time auditory feedback about the user's surroundings. The app combines on-device object detection, advanced depth estimation, and scene description powered by the **Gemma 3n** model to create a rich, interactive, and navigable experience.

The entire user interface is driven by intuitive gestures and voice commands, ensuring full accessibility without needing to see the screen.

---

## ✨ Core Features

The application is packed with features designed for seamless and eyes-free operation:

* **Multi-Object Detection:**
    * Dynamically detects key objects like **doors, chairs, and cars**.
    * Provides navigational instructions to guide the user towards the detected object (e.g., "The door is slightly to your left. Move left.").

* **Intelligent Obstacle Avoidance:**
    * Uses a segmentation model to identify and locate various obstacles in the user's path.
    * Warns the user about potential collisions and suggests alternative paths (e.g., "The chair is straight ahead, but there is a potted plant in the way. Move right to avoid it...").
    * Includes a proximity alert with haptic feedback (vibration) for imminent collisions.

* **Deep Scene Discovery (Powered by Gemma 3n):**
    * A sophisticated mode that captures the camera feed and sends it to the **Gemma 3n** local model.
    * Provides a rich, descriptive paragraph of the current scene.
    * Can read and transcribe text found on signs, papers, or posters in the environment.

* **Advanced Gesture Control:**
    * **Swipe Down:** Stops any active detection mode.
    * **Swipe Left/Right:** Cycles through the available detection modes (`door` -> `chair` -> `car` -> `door`...).
    * **Double Tap:** Toggles the camera flash on and off. The app intelligently switches to the main camera lens to activate the flash and switches back to the ultra-wide lens when it's turned off, all with a smooth, animated transition.

* **Comprehensive Voice Commands:**
    * Activate the app by setting it as the default digital assistant (long-press power button).
    * Use simple voice commands like "Detect doors," "Find a chair," or "Describe the scene" to initiate tasks.
    * Stop any process by saying "Stop."

* **Smart UI for Accessibility:**
    * **Low-Light Detection:** Automatically detects dark environments and displays an on-screen prompt to activate the flash.
    * **Haptic Feedback:** Provides physical confirmation for critical warnings.
    * **Auditory Feedback:** Every action, state change, and detection result is communicated clearly through Text-to-Speech (TTS).

---

## 🛠️ Technology Stack

This project integrates a powerful set of modern Android and Machine Learning technologies:

* **Core Language:** [Kotlin](https://kotlinlang.org/)
* **Camera:** [Android CameraX](https://developer.android.com/training/camerax) for robust camera control and lifecycle management.
* **Machine Learning (On-Device):**
    * [TensorFlow Lite](https://www.tensorflow.org/lite) for running models on the device.
    * **YOLOv8 & YOLOv11-seg:** State-of-the-art models for object detection and instance segmentation.
    * **MiDaS:** A powerful model for monocular depth estimation.
    * **Gemma 3n**: A lightweight, on-device generative AI model for advanced image-to-text descriptions.
* **Image Processing:** [OpenCV for Android](https://opencv.org/android/) for classical computer vision tasks like door confirmation and image manipulations.
* **UI & Asynchronous Tasks:**
    * [Coroutines](https://kotlinlang.org/docs/coroutines-overview.html) for managing background tasks without blocking the main thread.
    * Android UI Toolkit for views and animations.
* **Accessibility:** Android Text-to-Speech (TTS) and Vibrator services.

---

## 🚀 Setup and Installation

To get this project running on your own device, follow these steps:

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-username/ai-vision-assistant.git](https://github.com/your-username/ai-vision-assistant.git)
    ```

2.  **Download and Configure Gemma 3n 4B Model:**
    * Open the app on your Android device after installation.
    * Press the "Download Model" button in the app's settings or initial setup screen.
    * Wait for the Gemma 3B model to download and load. Once complete, the app will notify you that the model is ready to use.

3.  **Open in Android Studio:**
    * Open Android Studio and select "Open an existing project."
    * Navigate to the cloned repository folder and open it.

4.  **Build and Run:**
    * Let Android Studio sync the Gradle files.
    * Connect an Android device (with developer options enabled).
    * Click the "Run" button to build and install the app on your device.

---

## 📖 Usage Guide

The application is designed to be used without looking at the screen.

#### Initial Setup (One-Time)
1.  After installing, open your phone's **Settings**.
2.  Go to **Apps > Default apps > Digital assistant app**.
3.  Select **AI Vision Assistant** as your default assistant.

#### Activating the App
* **Long-press the power button** (or use the designated assistant shortcut on your device).
* The app will launch, and you will hear: *"Hello, how can I help you?"*

#### Voice Commands
After activation, you can use the following commands:
* **"Detect doors" / "chair" / "car":** Starts the specific object detection mode.
* **"Deep scene discovery" / "Describe the scene":** Activates the Gemma 3n-powered scene description.
* **"Stop":** Halts any active detection or discovery mode.

#### Gesture Controls
* **Swipe Down:** Instantly stops the current detection mode.
* **Double Tap:** Toggles the camera flash. The app will announce "Flash on" or "Flash off."
* **Swipe Left / Right (Only when a detection is active):** Cycles through the detection modes (`door` <-> `chair` <-> `car`). The app will announce the new mode.

