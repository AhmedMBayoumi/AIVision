# AI Vision Assistant

![AI Vision Assistant Logo](https://placehold.co/600x300/1e293b/ffffff?text=AI+Vision+Assistant)

**AI Vision Assistant** is a powerful Android application designed to aid visually impaired individuals by leveraging state-of-the-art machine learning models. It transforms a standard smartphone camera into an intelligent eye, providing real-time auditory feedback about the user's surroundings. The app combines on-device object detection, advanced depth estimation, and generative AI scene description to create a rich, interactive, and navigable experience.

The entire user interface is driven by intuitive **gestures and voice commands**, ensuring full accessibility without needing to see the screen.

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

* **Deep Scene Discovery (Powered by Google Gemini):**
    * A sophisticated mode that captures the camera feed and sends it to the **Gemini 2.5 Flash** model.
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
* **Machine Learning (Cloud):**
    * [Google Gemini API](https://ai.google.dev/) for the "Deep Scene Discovery" feature, providing advanced image-to-text descriptions.
* **Image Processing:** [OpenCV for Android](https://opencv.org/android/) for classical computer vision tasks like door confirmation and image manipulations.
* **UI & Asynchronous Tasks:**
    * [Coroutines](https://kotlinlang.org/docs/coroutines-overview.html) for managing background tasks and API calls without blocking the main thread.
    * Android UI Toolkit for views and animations.
* **Accessibility:** Android Text-to-Speech (TTS) and Vibrator services.

---

## 🚀 Setup and Installation

To get this project running on your own device, follow these steps:

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-username/ai-vision-assistant.git](https://github.com/your-username/ai-vision-assistant.git)
    ```

2.  **Add Your Gemini API Key:**
    * Obtain an API key from [Google AI Studio](https://ai.google.dev/).
    * In the root of the Android project, create a file named `local.properties`.
    * Add your API key to this file like so:
        ```properties
        GEMINI_API_KEY="YOUR_API_KEY_HERE"
        ```
    * The project is already configured to read this key from the `BuildConfig` field. This file is included in `.gitignore` to keep your key private.

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
* **"Deep scene discovery" / "Describe the scene":** Activates the Gemini-powered scene description.
* **"Stop":** Halts any active detection or discovery mode.

#### Gesture Controls
* **Swipe Down:** Instantly stops the current detection mode.
* **Double Tap:** Toggles the camera flash. The app will announce "Flash on" or "Flash off."
* **Swipe Left / Right (Only when a detection is active):** Cycles through the detection modes (`door` <-> `chair` <-> `car`). The app will announce the new mode.

