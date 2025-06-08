package com.amb.aivision

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.*
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import android.view.View
import android.view.WindowManager
import android.widget.Button
import android.widget.ImageButton
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.view.WindowCompat
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.imgproc.Imgproc
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.*

private const val TAG = "DoorDetection"

class MainActivity : AppCompatActivity(), TextToSpeech.OnInitListener {

    companion object {
        private const val PROCESSING_SIZE = 256         // for seg & depth
        private const val PROXIMITY_THRESHOLD_M = 0.5f  // 0.75 meters for obstacles
        private const val PROXIMITY_THRESHOLD_D = 0.075f // Close to door
        private const val DETECTION_RESOLUTION = 640   // For door detection
        private const val DEPTH_SCALE_FACTOR = 100.0f   // For MiDaS depth to meters
        private const val DETECTION_INTERVAL_MS = 500L // 0.5 second for both
    }

    private val classNames = listOf(
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
        "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
        "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
        "toothbrush"
    )

    private lateinit var previewView: PreviewView
    private lateinit var positionTextView: TextView
    private lateinit var detectButton: Button
    private lateinit var speechRecognizer: SpeechRecognizer

    private var lastDetectionTime = 0L
    private var isSpeaking = false
    private var isVoiceActive = false // Tracks if voice mode is active
    private var isRecognizerListening = false // Tracks if SpeechRecognizer is active
    private var previousMessage: String? = null // Track previous TTS message
    private var consecutiveIdenticalCount = 0  // Count consecutive identical messages

    private lateinit var tts: TextToSpeech
    private var shouldDetectDoors = false
    private var shouldDetectCars = false
    private var shouldDetectChairs = false
    private var shouldDetect = shouldDetectDoors || shouldDetectCars || shouldDetectChairs

    private var canProcess = true
    private var useYolo12s = false // Track current door detection model

    private lateinit var tflite: Interpreter // For YOLO door detection
    private var gpuDelegate: GpuDelegate? = null
    private lateinit var segInterpreter: Interpreter
    private var segGpuDelegate: GpuDelegate? = null
    private lateinit var depthInterpreter: Interpreter
    private var depthGpuDelegate: GpuDelegate? = null

    private lateinit var detectionProcessor: ImageProcessor
    private lateinit var imageProcessor: ImageProcessor
    private lateinit var depthProcessor: ImageProcessor

    private lateinit var cameraExecutor: ExecutorService
    private var initialOfflineWarningSent = false
    private var hasGreeted = false
    private var hasSpokenOfflineWarning = false // New flag
    private var hasSpokenDoorWarning = false // New flag
    private var cameraProvider: ProcessCameraProvider? = null // Store for unbinding
    private var wasDetectingBeforePause = false // Track detection state
    private lateinit var chairButton: ImageButton
    private lateinit var carButton: ImageButton
    private lateinit var doorButton: ImageButton

    // Handler for periodic detection
    private val handler = Handler(Looper.getMainLooper())
    private val detectionRunnable = object : Runnable {
        override fun run() {
            if (shouldDetect) {
                triggerDetection()
                handler.postDelayed(this, DETECTION_INTERVAL_MS)
            }
        }
    }

    @SuppressLint("MissingPermission")
    private fun isInternetAvailable(): Boolean {
        val connectivityManager =
            getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            val network = connectivityManager.activeNetwork ?: return false
            val capabilities = connectivityManager.getNetworkCapabilities(network) ?: return false
            return capabilities.hasCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET) &&
                    capabilities.hasCapability(NetworkCapabilities.NET_CAPABILITY_VALIDATED)
        } else {
            @Suppress("DEPRECATION")
            val networkInfo = connectivityManager.activeNetworkInfo ?: return false
            return networkInfo.isConnected
        }
    }

    // Reusable objects for door detection
    private val reusableBitmap by lazy {
        Bitmap.createBitmap(DETECTION_RESOLUTION, DETECTION_RESOLUTION, Bitmap.Config.ARGB_8888)
    }
    private val reusableCanvas by lazy { Canvas(reusableBitmap) }

    private var numDetections: Int = 0
    private val attributes: Int = 5 // Matches the output shape [1, 5, 8400]
    private var isWaitingForIconSelection = false

    @SuppressLint("SetTextI18n")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        if (OpenCVLoader.initLocal()) {
            Log.d("OpenCV", "OpenCV loaded successfully")
        } else {
            Log.e("OpenCV", "Failed to load OpenCV")
        }
        setContentView(R.layout.activity_main)

        // Initialize UI components
        previewView = findViewById(R.id.previewView)
        positionTextView = findViewById(R.id.positionTextView)
        detectButton = findViewById(R.id.detectButton)
        chairButton = findViewById(R.id.chairButton)
        carButton = findViewById(R.id.carButton)
        doorButton = findViewById(R.id.doorButton)
        detectButton.text = "Start Detection"
        chairButton.visibility = View.GONE
        carButton.visibility = View.GONE
        doorButton.visibility = View.GONE

        cameraExecutor = Executors.newSingleThreadExecutor()

        // Set click listeners
        doorButton.setOnClickListener {
            startDetection("door")
        }
        chairButton.setOnClickListener {
            startDetection("chair")
        }
        carButton.setOnClickListener {
            startDetection("car")
        }
        detectButton.setOnClickListener { toggleDetection() }
        detectButton.setOnLongClickListener {
            toggleDoorModel()
            true
        }

        // Request permissions and initialize if granted
        val permissionsToRequest = mutableListOf(Manifest.permission.CAMERA, Manifest.permission.RECORD_AUDIO)
        if (Build.VERSION.SDK_INT <= Build.VERSION_CODES.P) {
            permissionsToRequest.add(Manifest.permission.WRITE_EXTERNAL_STORAGE)
        }
        val permissionsNeeded = permissionsToRequest.filter {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }
        if (permissionsNeeded.isNotEmpty()) {
            ActivityCompat.requestPermissions(
                this,
                permissionsNeeded.toTypedArray(),
                1001
            )
        } else {
            initializeComponents()
        }
    }

    // Update initializeComponents to respect wasDetectingBeforePause
    @SuppressLint("SetTextI18n")
    private fun initializeComponents() {
        Log.d(TAG, "Initializing components")
        setupFullscreenUI()
        setupProcessors()
        if (!loadModels()) {
            positionTextView.text = "Failed to load models. Please check the app configuration."
            detectButton.isEnabled = false
            return
        }

        // Initialize TTS
        if (::tts.isInitialized) {
            tts.shutdown()
        }
        tts = TextToSpeech(this, this)

        // Initialize SpeechRecognizer
        if (::speechRecognizer.isInitialized) {
            speechRecognizer.destroy()
        }
        speechRecognizer = SpeechRecognizer.createSpeechRecognizer(this)
        speechRecognizer.setRecognitionListener(object : RecognitionListener {
            override fun onReadyForSpeech(params: Bundle?) {
                isRecognizerListening = true
                Log.d(TAG, "SpeechRecognizer ready for speech")
            }

            override fun onBeginningOfSpeech() {
                Log.d(TAG, "SpeechRecognizer began listening")
            }

            override fun onRmsChanged(rmsdB: Float) {}
            override fun onBufferReceived(buffer: ByteArray?) {}
            override fun onEndOfSpeech() {
                isRecognizerListening = false
                Log.d(TAG, "SpeechRecognizer ended speech input")
            }

            override fun onError(error: Int) {
                isRecognizerListening = false
                Log.e(TAG, "Speech recognition error: $error")
                if (!isSpeaking && (isVoiceActive || !shouldDetectDoors)) {
                    handler.postDelayed({ startVoiceRecognition() }, 100L)
                }
            }

            override fun onResults(results: Bundle?) {
                val matches = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                if (matches != null && matches.isNotEmpty()) {
                    val spokenText = matches[0]
                    Log.d(TAG, "Recognized: '$spokenText', isVoiceActive=$isVoiceActive, shouldDetect=$shouldDetectDoors")
                    if (!isSpeaking) { // Process only if not speaking
                        processVoiceCommand(spokenText)
                    }
                    if (!isSpeaking && (isVoiceActive || !shouldDetectDoors)) {
                        handler.postDelayed({ startVoiceRecognition() }, 100L)
                    }
                } else if (!isSpeaking && (isVoiceActive || !shouldDetectDoors)) {
                    handler.postDelayed({ startVoiceRecognition() }, 100L)
                }
            }

            override fun onPartialResults(partialResults: Bundle?) {}
            override fun onEvent(eventType: Int, params: Bundle?) {}
        })

        detectButton.setOnClickListener { toggleDetection() }
        detectButton.setOnLongClickListener {
            toggleDoorModel()
            true
        }

        // Start camera and voice recognition
        if (!isInternetAvailable()) {
            Log.w(TAG, "No internet connection available")
            initialOfflineWarningSent = true
        }
        startCamera()
        startVoiceRecognition()

        // Restore detection only if explicitly active before pause
        if (wasDetectingBeforePause && !shouldDetect) {
            shouldDetect = true
            shouldDetectDoors = true // Default to doors
            detectButton.text = "Stop detecting"
            handler.post(detectionRunnable)
            Log.d(TAG, "Restored continuous detection")
        }
    }


    override fun onPause() {
        super.onPause()
        // Stop detection and speech recognition when app goes to background
        shouldDetect = false
        shouldDetectDoors = false
        shouldDetectCars = false
        shouldDetectChairs = false
        isVoiceActive = false
        hasGreeted = false
        handler.removeCallbacks(detectionRunnable)
        detectButton.text = "Detect"
        runOnUiThread { positionTextView.text = "Detection stopped" }
        Log.d(TAG, "App paused: Stopped detection and reset voice state")

        if (::speechRecognizer.isInitialized && isRecognizerListening) {
            speechRecognizer.stopListening()
            isRecognizerListening = false
            Log.d(TAG, "Stopped SpeechRecognizer on pause")
        }

        // Stop TTS if speaking
        if (::tts.isInitialized && isSpeaking) {
            tts.stop()
            isSpeaking = false
            Log.d(TAG, "Stopped TTS on pause")
        }
    }

    override fun onStop() {
        super.onStop()
        Log.d(TAG, "Stopping activity")

        // Additional cleanup if needed (most handled in onPause)
        if (::speechRecognizer.isInitialized) {
            speechRecognizer.destroy()
            Log.d(TAG, "SpeechRecognizer destroyed during stop")
        }
    }

    override fun onResume() {
        super.onResume()
        // Restart speech recognition to listen for new commands
        if (::speechRecognizer.isInitialized && ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
            startVoiceRecognition()
            Log.d(TAG, "App resumed: Started SpeechRecognizer")
        }
        // Ensure detection remains off until commanded
        shouldDetect = false
        shouldDetectDoors = false
        shouldDetectCars = false
        shouldDetectChairs = false
        detectButton.text = "Detect"
        runOnUiThread { positionTextView.text = "Waiting for voice commands..." }
        Log.d(TAG, "App resumed: Waiting for voice commands")
    }

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            tts.language = Locale.US
            tts.setSpeechRate(1.25f) // Increase speech rate by 25%
            tts.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
                override fun onStart(utteranceId: String?) {
                    isSpeaking = true
                    Log.d(TAG, "TTS started speaking")
                    if (isRecognizerListening) {
                        speechRecognizer.stopListening()
                        isRecognizerListening = false
                        Log.d(TAG, "Stopped SpeechRecognizer during TTS")
                    }
                }
                override fun onDone(utteranceId: String?) {
                    isSpeaking = false
                    Log.d(TAG, "TTS finished speaking")
                    if ((isVoiceActive || !shouldDetectDoors) && !isRecognizerListening) {
                        startVoiceRecognition()
                        Log.d(TAG, "Restarted SpeechRecognizer after TTS")
                    }
                }
                override fun onError(utteranceId: String?) {
                    isSpeaking = false
                    Log.e(TAG, "TTS error occurred")
                    if ((isVoiceActive || !shouldDetectDoors) && !isRecognizerListening) {
                        startVoiceRecognition()
                        Log.d(TAG, "Restarted SpeechRecognizer after TTS error")
                    }
                }
            })
            if (initialOfflineWarningSent && !isInternetAvailable()) {
                speak("No internet connection. Some features like voice recognition may not work.")
            }
        } else {
            Log.e(TAG, "TTS initialization failed")
            runOnUiThread { positionTextView.text = "TTS initialization failed" }
        }
    }

    private fun startVoiceRecognition() {
        if (isRecognizerListening) {
            Log.d(TAG, "SpeechRecognizer already listening, skipping start")
            return
        }
        if (isSpeaking) {
            Log.d(TAG, "TTS is speaking, skipping SpeechRecognizer start")
            return
        }
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            Log.w(TAG, "RECORD_AUDIO permission not granted, cannot start voice recognition")
            runOnUiThread { positionTextView.text = "Audio permission required for voice recognition" }
            return
        }
        if (!isInternetAvailable()) {
            Log.w(TAG, "No internet connection, cannot start voice recognition")
            runOnUiThread {
                detectButton.visibility = View.VISIBLE
            }
            if (!hasSpokenOfflineWarning) {
                speak("No internet connection. Voice recognition is unavailable. Use the button to detect.")
                hasSpokenOfflineWarning = true
            }
            handler.postDelayed({ startVoiceRecognition() }, 5000L)
            return
        }
        // Internet is available, reset offline warning flag
        hasSpokenOfflineWarning = false
        runOnUiThread {
            detectButton.visibility = View.GONE
            // Only update text if SpeechRecognizer is about to start and TTS is not speaking
            if (!hasSpokenDoorWarning) {
                positionTextView.text = "Listening for commands..."
            }
        }
        if (!SpeechRecognizer.isRecognitionAvailable(this)) {
            Log.e(TAG, "Speech recognition is not available on this device")
            runOnUiThread { positionTextView.text = "Speech recognition not available on this device" }
            handler.postDelayed({ startVoiceRecognition() }, 1000L)
            return
        }
        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.getDefault())
            putExtra(RecognizerIntent.EXTRA_PROMPT, "Say 'hello', 'start', 'doors', 'cars', 'chairs', or 'stop'")
        }
        try {
            speechRecognizer.startListening(intent)
            isRecognizerListening = true
            Log.d(TAG, "Started voice recognition")
        } catch (e: Exception) {
            isRecognizerListening = false
            Log.e(TAG, "Failed to start speech recognition: ${e.message}")
            handler.postDelayed({ startVoiceRecognition() }, 1000L)
        }
    }

    private fun processVoiceCommand(command: String) {
        Log.d(TAG, "Processing command: '$command', isVoiceActive=$isVoiceActive, shouldDetect=$shouldDetectDoors")
        if (command.isBlank() || command.lowercase(Locale.getDefault()) in listOf(
                "hello, how can i help you", "starting detection, press on the icon that you want to detect",
                "starting detecting doors", "starting detecting cars", "starting detecting chairs", "stopping detection",
                "no internet connection. voice recognition is unavailable. use the button to detect",
                "no internet connection. some features like voice recognition may not work")) {
            Log.d(TAG, "Ignored command: '$command' (empty or TTS feedback)")
            return
        }
        if (isSpeaking) {
            Log.d(TAG, "Ignored command: '$command' (TTS active, no recognition allowed)")
            return
        }
        when {
            command.lowercase(Locale.getDefault()).contains("hello") || command.lowercase(Locale.getDefault()).contains("vai") -> {
                isVoiceActive = true
                if (!hasGreeted) {
                    speak("Hello, how can I help you?")
                    hasGreeted = true
                }
                runOnUiThread { positionTextView.text = "Voice activated, say 'doors', 'cars', 'chairs', or 'stop'" }
                Log.d(TAG, "Voice command: Activated with 'hello' or 'VAI'")
            }
            command.lowercase(Locale.getDefault()).contains("doors") -> {
                if (!shouldDetectDoors) {
                    startDetection("door")
                    Log.d(TAG, "Voice command: Started door detection")
                } else {
                    Log.d(TAG, "Voice command: Ignored 'doors' as detection is already active")
                }
            }
            command.lowercase(Locale.getDefault()).contains("cars") -> {
                if (!shouldDetectCars) {
                    startDetection("car")
                    Log.d(TAG, "Voice command: Started car detection")
                } else {
                    Log.d(TAG, "Voice command: Ignored 'cars' as detection is already active")
                }
            }
            command.lowercase(Locale.getDefault()).contains("chairs") -> {
                if (!shouldDetectChairs) {
                    startDetection("chair")
                    Log.d(TAG, "Voice command: Started chair detection")
                } else {
                    Log.d(TAG, "Voice command: Ignored 'chairs' as detection is already active")
                }
            }
            command.lowercase(Locale.getDefault()).contains("stop") -> {
                stopDetection()
                Log.d(TAG, "Voice command: Stopped detection")
            }
            else -> {
                Log.d(TAG, "Voice command: Unrecognized command: '$command'")
            }
        }
    }

    private fun setupFullscreenUI() {
        WindowCompat.setDecorFitsSystemWindows(window, false)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            window.decorView.windowInsetsController?.let {
                it.hide(android.view.WindowInsets.Type.statusBars())
                it.systemBarsBehavior =
                    android.view.WindowInsetsController.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
            }
        } else {
            @Suppress("DEPRECATION")
            window.setFlags(
                WindowManager.LayoutParams.FLAG_FULLSCREEN,
                WindowManager.LayoutParams.FLAG_FULLSCREEN
            )
        }
        supportActionBar?.hide()
    }

    private fun setupProcessors() {
        detectionProcessor = ImageProcessor.Builder()
            .add(
                ResizeOp(
                    DETECTION_RESOLUTION,
                    DETECTION_RESOLUTION,
                    ResizeOp.ResizeMethod.BILINEAR
                )
            )
            .add(NormalizeOp(0f, 255f))
            .build()

        imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(PROCESSING_SIZE, PROCESSING_SIZE, ResizeOp.ResizeMethod.BILINEAR))
            .add(NormalizeOp(0f, 255f))
            .build()

        depthProcessor = ImageProcessor.Builder()
            .add(ResizeOp(PROCESSING_SIZE, PROCESSING_SIZE, ResizeOp.ResizeMethod.BILINEAR))
            .add(NormalizeOp(0f, 255f))
            .build()
    }

    @SuppressLint("SetTextI18n")
    private fun loadModels(): Boolean {
        try {
            val compatList = CompatibilityList()
            var useGpu = compatList.isDelegateSupportedOnThisDevice

            // Load door detection model based on useYolo12s
            val modelFile = if (useYolo12s) "yolo12s.tflite" else "best(2)_float32.tflite"
            val model = try {
                FileUtil.loadMappedFile(this, modelFile)
            } catch (e: Exception) {
                Log.e(TAG, "Failed to load YOLO model $modelFile: ${e.message}", e)
                runOnUiThread {
                    positionTextView.text = "Error loading YOLO model: ${e.message}"
                }
                return false
            }

            // Close existing tflite and gpuDelegate
            if (::tflite.isInitialized) {
                tflite.close()
            }
            gpuDelegate?.close()

            if (useGpu) {
                try {
                    val gpuOptions = Interpreter.Options()
                    gpuDelegate = GpuDelegate(compatList.bestOptionsForThisDevice)
                    gpuOptions.addDelegate(gpuDelegate)
                    tflite = Interpreter(model, gpuOptions)
                    Log.d(TAG, "Door detection model $modelFile loaded with GPU delegate")
                } catch (e: Exception) {
                    Log.w(
                        TAG,
                        "GPU delegate failed for YOLO model: ${e.message}. Falling back to CPU.",
                        e
                    )
                    useGpu = false
                }
            }
            if (!useGpu) {
                val cpuOptions = Interpreter.Options().apply {
                    setNumThreads(min(Runtime.getRuntime().availableProcessors(), 4))
                    setUseNNAPI(false)
                }
                tflite = Interpreter(model, cpuOptions)
                Log.d(TAG, "Door detection model $modelFile loaded with CPU")
            }
            numDetections = tflite.getOutputTensor(0).shape()[2]

            // Load segmentation model
            val segModel = try {
                FileUtil.loadMappedFile(this, "yolo11s-seg.tflite")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to load segmentation model: ${e.message}", e)
                runOnUiThread {
                    positionTextView.text = "Error loading segmentation model: ${e.message}"
                }
                return false
            }

            if (useGpu) {
                try {
                    val gpuOptions = Interpreter.Options()
                    segGpuDelegate = GpuDelegate(compatList.bestOptionsForThisDevice)
                    gpuOptions.addDelegate(segGpuDelegate)
                    segInterpreter = Interpreter(segModel, gpuOptions)
                    Log.d(TAG, "Segmentation model loaded with GPU delegate")
                } catch (e: Exception) {
                    Log.w(
                        TAG,
                        "GPU delegate failed for segmentation model: ${e.message}. Falling back to CPU.",
                        e
                    )
                    useGpu = false
                }
            }
            if (!useGpu) {
                val cpuOptions = Interpreter.Options().apply {
                    setNumThreads(min(Runtime.getRuntime().availableProcessors(), 4))
                    setUseNNAPI(false)
                }
                segInterpreter = Interpreter(segModel, cpuOptions)
                Log.d(TAG, "Segmentation model loaded with CPU")
            }

            // Load depth model
            val depthModel = try {
                FileUtil.loadMappedFile(this, "MiDas.tflite")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to load depth model: ${e.message}", e)
                runOnUiThread {
                    positionTextView.text = "Error loading depth model: ${e.message}"
                }
                return false
            }

            if (useGpu) {
                try {
                    val gpuOptions = Interpreter.Options()
                    depthGpuDelegate = GpuDelegate(compatList.bestOptionsForThisDevice)
                    gpuOptions.addDelegate(depthGpuDelegate)
                    depthInterpreter = Interpreter(depthModel, gpuOptions)
                    Log.d(TAG, "Depth model loaded with GPU delegate")
                } catch (e: Exception) {
                    Log.w(
                        TAG,
                        "GPU delegate failed for depth model: ${e.message}. Falling back to CPU.",
                        e
                    )
                    useGpu = false
                }
            }
            if (!useGpu) {
                val cpuOptions = Interpreter.Options().apply {
                    setNumThreads(min(Runtime.getRuntime().availableProcessors(), 4))
                    setUseNNAPI(false)
                }
                depthInterpreter = Interpreter(depthModel, cpuOptions)
                Log.d(TAG, "Depth model loaded with CPU")
            }

            return true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load models: ${e.message}", e)
            runOnUiThread {
                positionTextView.text = "Error loading models: ${e.message}"
            }
            return false
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder().build()
                .also { it.setSurfaceProvider(previewView.surfaceProvider) }

            val analysis = ImageAnalysis.Builder()
                .setTargetResolution(android.util.Size(640, 640))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor, { proxy -> onFrame(proxy) })
                }

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    preview,
                    analysis
                )
            } catch (e: Exception) {
                Log.e(TAG, "Camera binding failed: ${e.message}", e)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    // Update toggleDetection to synchronize shouldDetect
    // Edited function: Toggles between prompting for icon selection and stopping
    @SuppressLint("SetTextI18n")
    private fun toggleDetection() {
        if (shouldDetect || isWaitingForIconSelection) {
            stopDetection()
        } else {
            startDetection() // Prompt for icon selection
        }
    }

    @SuppressLint("SetTextI18n")
    private fun toggleDoorModel() {
        useYolo12s = !useYolo12s
        val modelName = if (useYolo12s) "yolo12s" else "best(2)_float32"
        if (loadModels()) {
            Toast.makeText(this, "Changed model to $modelName", Toast.LENGTH_SHORT).show()
            Log.d(TAG, "Switched door detection model to $modelName")
        } else {
            Toast.makeText(this, "Failed to change model to $modelName", Toast.LENGTH_SHORT).show()
            useYolo12s = !useYolo12s // Revert on failure
            Log.e(TAG, "Failed to switch to $modelName")
        }
    }

    private fun triggerDetection() {
        if (canProcess && shouldDetect) {
            Log.d(TAG, "Triggered detection")
        }
    }

    @SuppressLint("SetTextI18n")
    private fun onFrame(image: ImageProxy) {
        val currentTime = System.currentTimeMillis()
        if (!shouldDetect || !canProcess || currentTime - lastDetectionTime < DETECTION_INTERVAL_MS) {
            image.close()
            Log.d(TAG, "Skipped frame: shouldDetect=$shouldDetect, canProcess=$canProcess, timeSinceLast=$currentTime - $lastDetectionTime")
            return
        }

        canProcess = false
        lastDetectionTime = currentTime
        try {
            val bmp = image.toBitmap()
            image.close()

            // Check if the full image is mostly uniform (e.g., too close to a wall)
            if (isImageMostlyUniform(bmp)) {
                val msg = "You are going to hit something."
                handleMessage(msg)
                canProcess = true
                return
            }

            // Step 1: Detection based on active mode
            val (targetBox, position, depthMeters) = when {
                shouldDetectDoors -> {
                    val (doorBox, pos) = detectDoor(bmp)
                    val fullDepthMap = runDepthEstimation(bmp)
                    val doorDepth = if (doorBox != null) {
                        val rawDoorDepth = avgDepthInBoxFixed(fullDepthMap, doorBox, bmp.width, bmp.height)
                        if (rawDoorDepth.isFinite()) DEPTH_SCALE_FACTOR / rawDoorDepth else Float.MAX_VALUE
                    } else Float.MAX_VALUE
                    Triple(doorBox, pos, doorDepth)
                }
                shouldDetectChairs -> {
                    val (chairBox, pos) = detectChair(bmp)
                    val fullDepthMap = runDepthEstimation(bmp)
                    val chairDepth = if (chairBox != null) {
                        val rawChairDepth = avgDepthInBoxFixed(fullDepthMap, chairBox, bmp.width, bmp.height)
                        if (rawChairDepth.isFinite()) DEPTH_SCALE_FACTOR / rawChairDepth else Float.MAX_VALUE
                    } else Float.MAX_VALUE
                    Triple(chairBox, pos, chairDepth)
                }
                shouldDetectCars -> {
                    val (carBox, pos) = detectCar(bmp)
                    val fullDepthMap = runDepthEstimation(bmp)
                    val carDepth = if (carBox != null) {
                        val rawCarDepth = avgDepthInBoxFixed(fullDepthMap, carBox, bmp.width, bmp.height)
                        if (rawCarDepth.isFinite()) DEPTH_SCALE_FACTOR / rawCarDepth else Float.MAX_VALUE
                    } else Float.MAX_VALUE
                    Triple(carBox, pos, carDepth)
                }
                else -> Triple(null, "", Float.MAX_VALUE)
            }
            Log.d(TAG, "Detected: box=$targetBox, position=$position, depth=$depthMeters meters")

            // Step 2: Obstacle detection on full image
            val fullDepthMap = runDepthEstimation(bmp)
            val obstacles = runSegmentation(bmp)
            val mappedObstacles = obstacles.map { obstacle ->
                val mappedMask = mapMaskToOriginal(obstacle.mask, bmp.width, bmp.height)
                Obstacle(obstacle.box, mappedMask, obstacle.className)
            }.filter {
                // Filter out the target class
                when {
                    shouldDetectDoors -> it.className != "door"
                    shouldDetectChairs -> it.className != "chair"
                    shouldDetectCars -> it.className != "car"
                    else -> true
                }
            }

            // Step 3: Generate message
            val targetClass = when {
                shouldDetectDoors -> "door"
                shouldDetectChairs -> "chair"
                shouldDetectCars -> "car"
                else -> "unknown"
            }
            val message = if (targetBox != null) {
                generateNavigationInstruction(
                    targetBox,
                    position,
                    depthMeters,
                    mappedObstacles,
                    fullDepthMap,
                    bmp,
                    targetClass
                )
            } else {
                // No target detected, check for close obstacles
                val closeObstacles = mappedObstacles.filter { obstacle ->
                    val obstacleDepth = avgMaskDepthFixed(fullDepthMap, obstacle.mask, bmp.width, bmp.height)
                    val obstacleDepthMeters = if (obstacleDepth.isFinite()) DEPTH_SCALE_FACTOR / obstacleDepth else Float.MAX_VALUE
                    obstacleDepthMeters < PROXIMITY_THRESHOLD_M
                }
                if (closeObstacles.isNotEmpty()) {
                    val obstacleNames = closeObstacles.joinToString(" and ") { it.className }
                    "$obstacleNames detected. Move around a little."
                } else {
                    "No $targetClass detected. Move around a little."
                }
            }

            // Output results
            if (message.isNotEmpty()) {
                handleMessage(message)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Frame processing error: ${e.message}", e)
            runOnUiThread { positionTextView.text = "Error: ${e.message}" }
        } finally {
            canProcess = true
        }
    }

    private fun handleMessage(message: String) {
        // Update consecutive message count
        if (message == previousMessage) {
            consecutiveIdenticalCount++
            if (consecutiveIdenticalCount >= 5) {
                // After 7th appearance (count 0 to 5), reset cycle
                consecutiveIdenticalCount = 0
                previousMessage = null
                Log.d(TAG, "Reset cycle after 6 identical messages: '$message'")
            }
        } else {
            // New message, reset count
            consecutiveIdenticalCount = 0
            previousMessage = message
        }

        // Play and display message only on second appearance (count == 1)
        if (consecutiveIdenticalCount == 1) {
            speak(message)
            runOnUiThread { positionTextView.text = message }
            Log.d(TAG, "Played and displayed message '$message' (second appearance)")
        } else {
            Log.d(TAG, "Skipped message '$message' (consecutive count: $consecutiveIdenticalCount)")
        }
    }

    private fun mapMaskToOriginal(
        mask: Array<FloatArray>,
        origWidth: Int,
        origHeight: Int
    ): Array<FloatArray> {
        val origMask = Array(origHeight) { FloatArray(origWidth) }
        val maskH = mask.size
        val maskW = mask[0].size
        for (y in 0 until origHeight) {
            for (x in 0 until origWidth) {
                val maskY = (y * maskH / origHeight).coerceIn(0, maskH - 1)
                val maskX = (x * maskW / origWidth).coerceIn(0, maskW - 1)
                origMask[y][x] = mask[maskY][maskX]
            }
        }
        return origMask
    }

    private fun speak(msg: String) {
        tts.stop() // Stop ongoing speech before speaking new message
        val params = Bundle()
        params.putString(TextToSpeech.Engine.KEY_PARAM_UTTERANCE_ID, "messageId")
        tts.speak(msg, TextToSpeech.QUEUE_FLUSH, params, "messageId")
    }

    private fun generateNavigationInstruction(
        targetBox: RectF,
        position: String,
        depthMeters: Float,
        obstacles: List<Obstacle>,
        depthMap: Array<FloatArray>,
        bitmap: Bitmap,
        targetClass: String
    ): String {
        // Use PROXIMITY_THRESHOLD_D (0.075m) for both doors and chairs
        val proximityThreshold = PROXIMITY_THRESHOLD_D
        if (depthMeters < proximityThreshold) {
            return "You have reached the $targetClass."
        }

        val blockingObstacles = obstacles.filter { obstacle ->
            val obstacleDepth = avgMaskDepthFixed(depthMap, obstacle.mask, bitmap.width, bitmap.height)
            val obstacleDepthMeters = if (obstacleDepth.isFinite()) DEPTH_SCALE_FACTOR / obstacleDepth else Float.MAX_VALUE
            obstacleDepthMeters < depthMeters && obstacleDepthMeters < PROXIMITY_THRESHOLD_M && isObstacleInPath(obstacle.box, targetBox)
        }

        if (blockingObstacles.isEmpty()) {
            return when (position) {
                "left" -> "The $targetClass is slightly to your left. Move left."
                "right" -> "The $targetClass is slightly to your right. Move right."
                else -> "The $targetClass is straight ahead. Move forward."
            }
        }

        val obstacleNames = blockingObstacles.joinToString(" and ") { it.className }
        return when (position) {
            "left" -> {
                val rightHalf = Bitmap.createBitmap(bitmap, 320, 0, 320, 640)
                val rightObstacles = runSegmentation(rightHalf).filter { it.className != targetClass }
                if (rightObstacles.isEmpty()) {
                    "The $targetClass is to your left, but there is $obstacleNames in the way. Move right to avoid it, then turn left."
                } else {
                    val rightObstacleNames = rightObstacles.joinToString(" and ") { it.className }
                    "The $targetClass is to your left, but there is $obstacleNames in the way. The right path is blocked by $rightObstacleNames."
                }
            }
            "right" -> {
                val leftHalf = Bitmap.createBitmap(bitmap, 0, 0, 320, 640)
                val leftObstacles = runSegmentation(leftHalf).filter { it.className != targetClass }
                if (leftObstacles.isEmpty()) {
                    "The $targetClass is to your right, but there is $obstacleNames in the way. Move left to avoid it, then turn right."
                } else {
                    val leftObstacleNames = leftObstacles.joinToString(" and ") { it.className }
                    "The $targetClass is to your right, but there is $obstacleNames in the way. The left path is blocked by $leftObstacleNames."
                }
            }
            else -> {
                val rightThird = Bitmap.createBitmap(bitmap, 426, 0, 214, 640)
                val rightObstacles = runSegmentation(rightThird).filter { it.className != targetClass }
                if (rightObstacles.isEmpty()) {
                    "The $targetClass is straight ahead, but there is $obstacleNames in the way. Move right to avoid it, then continue forward."
                } else {
                    val leftThird = Bitmap.createBitmap(bitmap, 0, 0, 213, 640)
                    val leftObstacles = runSegmentation(leftThird).filter { it.className != targetClass }
                    if (leftObstacles.isEmpty()) {
                        "The $targetClass is straight ahead, but there is $obstacleNames in the way. Move left to avoid it, then continue forward."
                    } else {
                        val leftObstacleNames = leftObstacles.joinToString(" and ") { it.className }
                        "The $targetClass is straight ahead, but there is $obstacleNames in the way. Both paths are blocked by $leftObstacleNames."
                    }
                }
            }
        }
    }

    private fun avgDepthInBoxFixed(
        depthMap: Array<FloatArray>,
        box: RectF,
        imageWidth: Int,
        imageHeight: Int
    ): Float {
        var sum = 0f
        var cnt = 0
        val depthH = depthMap.size
        val depthW = depthMap[0].size
        val startY = ((box.top / imageHeight) * depthH).toInt().coerceIn(0, depthH - 1)
        val endY = ((box.bottom / imageHeight) * depthH).toInt().coerceIn(0, depthH - 1)
        val startX = ((box.left / imageWidth) * depthW).toInt().coerceIn(0, depthW - 1)
        val endX = ((box.right / imageWidth) * depthW).toInt().coerceIn(0, depthW - 1)
        for (y in startY..endY) {
            for (x in startX..endX) {
                val depthValue = depthMap[y][x]
                if (!depthValue.isNaN() && depthValue.isFinite()) {
                    sum += depthValue
                    cnt++
                }
            }
        }
        return if (cnt > 0) sum / cnt else Float.MAX_VALUE
    }

    private fun avgMaskDepthFixed(
        depthMap: Array<FloatArray>,
        mask: Array<FloatArray>,
        imageWidth: Int,
        imageHeight: Int
    ): Float {
        var sum = 0f
        var cnt = 0
        val depthH = depthMap.size
        val depthW = depthMap[0].size
        for (y in 0 until depthH) {
            for (x in 0 until depthW) {
                val maskY = (y * mask.size / depthH).coerceIn(0, mask.size - 1)
                val maskX = (x * mask[0].size / depthW).coerceIn(0, mask[0].size - 1)
                if (mask[maskY][maskX] > 0.1f) {
                    val depthValue = depthMap[y][x]
                    if (!depthValue.isNaN() && depthValue.isFinite()) {
                        sum += depthValue
                        cnt++
                    }
                }
            }
        }
        return if (cnt > 0) sum / cnt else Float.MAX_VALUE
    }

    private fun isObstacleInPath(obstacleBox: RectF, doorBox: RectF): Boolean {
        val obstacleCenter = (obstacleBox.left + obstacleBox.right) / 2
        val doorCenter = (doorBox.left + doorBox.right) / 2
        val pathWidth = doorBox.width() * 1.75f
        return abs(obstacleCenter - doorCenter) < pathWidth / 2
    }

    private fun dilateArray(array: Array<FloatArray>, kernelSize: Int): Array<FloatArray> {
        val result = Array(array.size) { FloatArray(array[0].size) }
        val radius = kernelSize / 2
        for (y in array.indices) {
            for (x in array[0].indices) {
                var value = 0f
                for (ky in -radius..radius) {
                    for (kx in -radius..radius) {
                        val ny = y + ky
                        val nx = x + kx
                        if (ny >= 0 && ny < array.size && nx >= 0 && nx < array[0].size) {
                            value = max(value, array[ny][nx])
                        }
                    }
                }
                result[y][x] = value
            }
        }
        return result
    }
    @SuppressLint("SetTextI18n")
    private fun startDetection(type: String? = null) {
        if (type == null) {
            // Prompt for icon selection
            isWaitingForIconSelection = true
            shouldDetect = false
            shouldDetectDoors = false
            shouldDetectCars = false
            shouldDetectChairs = false
            runOnUiThread {
                detectButton.text = "Stop Detection"
                chairButton.visibility = View.VISIBLE
                carButton.visibility = View.VISIBLE
                doorButton.visibility = View.VISIBLE
                positionTextView.text = "Choose detection type"
            }
            speak("Starting detection, press on the icon that you want to detect")
            Log.d(TAG, "Prompting for icon selection")
        } else {
            // Start specific detection mode
            isWaitingForIconSelection = false
            shouldDetect = true
            shouldDetectDoors = type == "door"
            shouldDetectCars = type == "car"
            shouldDetectChairs = type == "chair"
            runOnUiThread {
                detectButton.text = "Stop Detection"
                chairButton.visibility = View.GONE
                carButton.visibility = View.GONE
                doorButton.visibility = View.GONE
                positionTextView.text = "Starting $type detection"
            }
            speak("Starting detecting ${type}s")
            handler.post(detectionRunnable)
            Log.d(TAG, "Started $type detection")
        }
    }

    // Edited function: Resets state and hides icons
    @SuppressLint("SetTextI18n")
    private fun stopDetection() {
        shouldDetect = false
        shouldDetectDoors = false
        shouldDetectCars = false
        shouldDetectChairs = false
        isWaitingForIconSelection = false
        isVoiceActive = false
        handler.removeCallbacks(detectionRunnable)
        runOnUiThread {
            detectButton.text = "Start Detection"
            chairButton.visibility = View.GONE
            carButton.visibility = View.GONE
            doorButton.visibility = View.GONE
            positionTextView.text = "Detection stopped"
        }
        speak("Stopping detection")
        Log.d(TAG, "Stopped detection")
    }

    private fun detectDoor(bitmap: Bitmap): Pair<RectF?, String> {
        val tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(bitmap)
        val processedImage = detectionProcessor.process(tensorImage)
        val inputBuffer = processedImage.buffer
        inputBuffer.rewind()
        val outputs = Array(1) { Array(5) { FloatArray(8400) } }
        tflite.run(inputBuffer, outputs)
        val threshold = 0.6f
        val iouThresh = 0.4f
        val detections = mutableListOf<Triple<RectF, Float, String>>()

        for (i in 0 until 8400) {
            val x = outputs[0][0][i]
            val y = outputs[0][1][i]
            val w = outputs[0][2][i]
            val h = outputs[0][3][i]
            val confidence = outputs[0][4][i]
            if (confidence > threshold) {
                val centerX = x * bitmap.width
                val centerY = y * bitmap.height
                val widthScaled = w * bitmap.width
                val heightScaled = h * bitmap.height
                val left = centerX - widthScaled / 2
                val top = centerY - heightScaled / 2
                val right = centerX + widthScaled / 2
                val bottom = centerY + heightScaled / 2
                val rect = RectF(left, top, right, bottom)
                val normalizedX = centerX / bitmap.width
                val position = when {
                    normalizedX < 0.33 -> "left"
                    normalizedX < 0.66 -> "mid"
                    else -> "right"
                }
                detections.add(Triple(rect, confidence, position))
                Log.d(
                    TAG,
                    "Detection $i: Confidence=$confidence, Box=[left=$left, top=$top, right=$right, bottom=$bottom], Position=$position"
                )
            }
        }

        val sortedDetections = detections.sortedByDescending { it.second }
        val keep = mutableListOf<Triple<RectF, Float, String>>()
        for (det in sortedDetections) {
            if (keep.size < 2 && keep.none { iou(it.first, det.first) > iouThresh }) {
                keep.add(det)
            }
        }

        if (keep.isNotEmpty()) {
            val best = keep[0]
            val croppedBitmap = cropBitmap(bitmap, best.first)
            if (confirmDoorWithClassicalMethods(croppedBitmap)) {
                Log.d(TAG, "Confirmed door: position=${best.third}, confidence=${best.second}")
                return Pair(best.first, best.third)
            } else if (keep.size > 1) {
                val secondBest = keep[1]
                val secondCroppedBitmap = cropBitmap(bitmap, secondBest.first)
                if (confirmDoorWithClassicalMethods(secondCroppedBitmap)) {
                    Log.d(
                        TAG,
                        "Confirmed second-best door: position=${secondBest.third}, confidence=${secondBest.second}"
                    )
                    return Pair(secondBest.first, secondBest.third)
                }
            }
            Log.d(TAG, "No door confirmed, using best detection: position=${best.third}")
            return Pair(best.first, best.third)
        }
        Log.d(TAG, "No door detected")
        return Pair(null, "")
    }
    private fun detectChair(bitmap: Bitmap): Pair<RectF?, String> {
        val chairs = runSegmentationForChairs(bitmap)
        val sortedChairs = chairs.sortedByDescending { it.box.width() * it.box.height() }

        if (sortedChairs.isNotEmpty()) {
            val bestChair = sortedChairs[0]
            val centerX = (bestChair.box.left + bestChair.box.right) / 2
            val normalizedX = centerX / bitmap.width
            val position = when {
                normalizedX < 0.33 -> "left"
                normalizedX < 0.66 -> "mid"
                else -> "right"
            }
            Log.d(TAG, "Chair detected: position=$position, box=${bestChair.box}")
            return Pair(bestChair.box, position)
        }

        Log.d(TAG, "No chair detected")
        return Pair(null, "")
    }

    // New function: Detect cars using segmentation
    private fun detectCar(bitmap: Bitmap): Pair<RectF?, String> {
        val cars = runSegmentationForCars(bitmap)
        val sortedCars = cars.sortedByDescending { it.box.width() * it.box.height() }

        if (sortedCars.isNotEmpty()) {
            val bestCar = sortedCars[0]
            val centerX = (bestCar.box.left + bestCar.box.right) / 2
            val normalizedX = centerX / bitmap.width
            val position = when {
                normalizedX < 0.33 -> "left"
                normalizedX < 0.66 -> "mid"
                else -> "right"
            }
            Log.d(TAG, "Car detected: position=$position, box=${bestCar.box}")
            return Pair(bestCar.box, position)
        }

        Log.d(TAG, "No car detected")
        return Pair(null, "")
    }

    private fun cropBitmap(bitmap: Bitmap, rect: RectF): Bitmap {
        val left = rect.left.toInt().coerceIn(0, bitmap.width)
        val top = rect.top.toInt().coerceIn(0, bitmap.height)
        val right = rect.right.toInt().coerceIn(0, bitmap.width)
        val bottom = rect.bottom.toInt().coerceIn(0, bitmap.height)
        return Bitmap.createBitmap(bitmap, left, top, right - left, bottom - top)
    }

    private fun confirmDoorWithClassicalMethods(bitmap: Bitmap): Boolean {
        val mat = Mat()
        Utils.bitmapToMat(bitmap, mat)
        val gray = Mat()
        Imgproc.cvtColor(mat, gray, Imgproc.COLOR_BGR2GRAY)
        val edges = Mat()
        Imgproc.Canny(gray, edges, 50.0, 150.0)
        val lines = Mat()
        Imgproc.HoughLinesP(edges, lines, 1.0, Math.PI / 180, 50, 50.0, 10.0)
        var verticalLines = 0
        var horizontalLines = 0
        for (i in 0 until lines.rows()) {
            val line = lines.get(i, 0)
            val x1 = line[0]
            val y1 = line[1]
            val x2 = line[2]
            val y2 = line[3]
            val angle = atan2(y2 - y1, x2 - x1) * 180 / Math.PI
            if (abs(angle) < 10 || abs(angle - 180) < 10) horizontalLines++
            if (abs(angle - 90) < 10 || abs(angle + 90) < 10) verticalLines++
        }
        val contours = mutableListOf<MatOfPoint>()
        val hierarchy = Mat()
        Imgproc.findContours(
            edges,
            contours,
            hierarchy,
            Imgproc.RETR_EXTERNAL,
            Imgproc.CHAIN_APPROX_SIMPLE
        )
        for (contour in contours) {
            val approx = MatOfPoint2f()
            Imgproc.approxPolyDP(
                MatOfPoint2f(*contour.toArray()),
                approx,
                Imgproc.arcLength(MatOfPoint2f(*contour.toArray()), true) * 0.02,
                true
            )
            if (approx.toArray().size == 4) {
                val points = approx.toArray()
                val rect = Imgproc.boundingRect(MatOfPoint(*points))
                val aspectRatio = rect.height.toFloat() / rect.width
                if (aspectRatio in 1.5..3.0) {
                    if (verticalLines >= 2 && horizontalLines >= 2) {
                        return true
                    }
                }
            }
        }
        return false
    }

    data class Obstacle(val box: RectF, val mask: Array<FloatArray>, val className: String)

    private fun runSegmentation(roi: Bitmap): List<Obstacle> {
        try {
            Log.d(TAG, "runSegmentation: ROI dimensions: ${roi.width}x${roi.height}")
            val ti = TensorImage(DataType.FLOAT32)
            ti.load(roi)
            val input = imageProcessor.process(ti).buffer
            val inputArray = FloatArray(input.capacity() / 4).apply { input.asFloatBuffer().get(this) }
            Log.d(
                TAG,
                "runSegmentation: Input pixel range: min=${inputArray.minOrNull()}, max=${inputArray.maxOrNull()}"
            )
            val detShape = segInterpreter.getOutputTensor(0).shape()
            val protoShape = segInterpreter.getOutputTensor(1).shape()
            val detOut = Array(detShape[0]) { Array(detShape[1]) { FloatArray(detShape[2]) } }
            val protoOut = Array(protoShape[0]) {
                Array(protoShape[1]) {
                    Array(protoShape[2]) {
                        FloatArray(protoShape[3])
                    }
                }
            }
            val outputs = mapOf(0 to detOut, 1 to protoOut)
            segInterpreter.runForMultipleInputsOutputs(arrayOf(input), outputs)
            val raw = detOut[0]
            val dets = mutableListOf<Triple<RectF, FloatArray, String>>()
            val threshold = 0.5f
            val numClasses = 80
            val maskCoefsCount = detShape[1] - 4 - numClasses
            val maskCoefsStartIdx = 4 + numClasses
            for (i in 0 until detShape[2]) {
                var maxClassProb = 0f
                var maxClassIdx = -1
                for (c in 0 until numClasses) {
                    val prob = raw[4 + c][i]
                    if (prob > maxClassProb) {
                        maxClassProb = prob
                        maxClassIdx = c
                    }
                }
                if (maxClassProb > threshold) {
                    val cx = raw[0][i] * roi.width
                    val cy = raw[1][i] * roi.height
                    val ww = raw[2][i] * roi.width
                    val hh = raw[3][i] * roi.height
                    val box = RectF(cx - ww / 2, cy - hh / 2, cx + ww / 2, cy + hh / 2)
                    val maskCoefs = FloatArray(maskCoefsCount) { c ->
                        val idx = maskCoefsStartIdx + c
                        if (idx < detShape[1]) raw[idx][i] else 0f
                    }
                    val className = if (maxClassIdx >= 0 && maxClassIdx < classNames.size) {
                        classNames[maxClassIdx]
                    } else {
                        "Unknown"
                    }
                    dets += Triple(box, maskCoefs, className)
                }
            }
            val final = applyNMS(dets, 0.5f)
            val obstacles = mutableListOf<Obstacle>()
            val protoH = protoShape[1]
            val protoW = protoShape[2]
            val protoC = protoShape[3]
            for ((box, coefs, className) in final) {
                try {
                    val mask = Array(256) { FloatArray(256) }
                    val maskValues = mutableListOf<Float>()
                    var activePixels = 0
                    for (dy in 0 until 256) {
                        for (dx in 0 until 256) {
                            val py = (dy * protoH / 256).toInt().coerceIn(0, protoH - 1)
                            val px = (dx * protoW / 256).toInt().coerceIn(0, protoW - 1)
                            var maskValue = 0f
                            for (c in 0 until minOf(coefs.size, protoC)) {
                                maskValue += coefs[c] * protoOut[0][py][px][c]
                            }
                            maskValue = 1.0f / (1.0f + exp(-maskValue))
                            maskValues.add(maskValue)
                            if (maskValue > 0.1f) {
                                mask[dy][dx] = 1f
                                activePixels++
                            }
                        }
                    }
                    if (activePixels >= 50) {
                        if ((shouldDetectCars && className == "car") || (shouldDetectChairs && className == "chair")) {
                            Log.d(TAG, "Obstacle skipped: name=$className, not an obstacle when detecting this class")
                            continue
                        }
                        val dilatedMask = dilateArray(mask, 3)
                        obstacles.add(Obstacle(box, dilatedMask, className))
                        Log.d(TAG, "Obstacle added: name=$className, size=$activePixels pixels")
                    } else {
                        Log.d(TAG, "Obstacle skipped: name=$className, too small, size=$activePixels pixels")
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Error building mask for $className: ${e.message}", e)
                }
            }
            // Print names of all detected obstacles
            obstacles.forEach { obstacle ->
                Log.d(TAG, "Detected obstacle: ${obstacle.className}")
            }
            return obstacles
        } catch (e: Exception) {
            Log.e(TAG, "Segmentation error: ${e.message}", e)
            return emptyList()
        }
    }
    // New function to detect chairs using segmentation
    private fun runSegmentationForChairs(roi: Bitmap): List<Obstacle> {
        try {
            Log.d(TAG, "runSegmentationForChairs: ROI dimensions: ${roi.width}x${roi.height}")
            val ti = TensorImage(DataType.FLOAT32).apply { load(roi) }
            val input = imageProcessor.process(ti).buffer
            val inputArray = FloatArray(input.capacity() / 4).apply { input.asFloatBuffer().get(this) }
            Log.d(TAG, "runSegmentationForChairs: Input pixel range: min=${inputArray.minOrNull()}, max=${inputArray.maxOrNull()}")

            val detShape = segInterpreter.getOutputTensor(0).shape()
            val protoShape = segInterpreter.getOutputTensor(1).shape()
            val detOut = Array(detShape[0]) { Array(detShape[1]) { FloatArray(detShape[2]) } }
            val protoOut = Array(protoShape[0]) { Array(protoShape[1]) { Array(protoShape[2]) { FloatArray(protoShape[3]) } } }
            val outputs = mapOf(0 to detOut, 1 to protoOut)
            segInterpreter.runForMultipleInputsOutputs(arrayOf(input), outputs)

            val raw = detOut[0]
            val dets = mutableListOf<Triple<RectF, FloatArray, String>>()
            val threshold = 0.5f
            val numClasses = 80
            val maskCoefsCount = detShape[1] - 4 - numClasses
            val maskCoefsStartIdx = 4 + numClasses

            for (i in 0 until detShape[2]) {
                var maxClassProb = 0f
                var maxClassIdx = -1
                for (c in 0 until numClasses) {
                    if (classNames[c] == "chair") {
                        val prob = raw[4 + c][i]
                        if (prob > maxClassProb) {
                            maxClassProb = prob
                            maxClassIdx = c
                        }
                    }
                }
                if (maxClassProb > threshold && maxClassIdx >= 0) {
                    val cx = raw[0][i] * roi.width
                    val cy = raw[1][i] * roi.height
                    val ww = raw[2][i] * roi.width
                    val hh = raw[3][i] * roi.height
                    val box = RectF(cx - ww / 2, cy - hh / 2, cx + ww / 2, cy + hh / 2)
                    val maskCoefs = FloatArray(maskCoefsCount) { c ->
                        val idx = maskCoefsStartIdx + c
                        if (idx < detShape[1]) raw[idx][i] else 0f
                    }
                    val className = classNames[maxClassIdx]
                    dets += Triple(box, maskCoefs, className)
                }
            }

            val finalDets = applyNMS(dets, 0.5f)
            val obstacles = mutableListOf<Obstacle>()
            val protoH = protoShape[1]
            val protoW = protoShape[2]
            val protoC = protoShape[3]

            for ((box, coefs, className) in finalDets) {
                try {
                    val mask = Array(256) { FloatArray(256) }
                    val maskValues = mutableListOf<Float>()
                    var activePixels = 0
                    for (dy in 0 until 256) {
                        for (dx in 0 until 256) {
                            val py = (dy * protoH / 256).coerceIn(0, protoH - 1)
                            val px = (dx * protoW / 256).coerceIn(0, protoW - 1)
                            var maskValue = 0f
                            for (c in 0 until minOf(coefs.size, protoC)) {
                                maskValue += coefs[c] * protoOut[0][py][px][c]
                            }
                            maskValue = 1.0f / (1.0f + exp(-maskValue))
                            maskValues.add(maskValue)
                            if (maskValue > 0.1f) {
                                mask[dy][dx] = 1f
                                activePixels++
                            }
                        }
                    }
                    if (activePixels >= 50) {
                        val dilatedMask = dilateArray(mask, 3)
                        obstacles.add(Obstacle(box, dilatedMask, className))
                        Log.d(TAG, "Chair added: name=$className, size=$activePixels pixels")
                    } else {
                        Log.d(TAG, "Chair skipped: name=$className, too small, size=$activePixels pixels")
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Error building mask for $className: ${e.message}", e)
                }
            }
            obstacles.forEach { Log.d(TAG, "Detected chair: ${it.className}") }
            return obstacles
        } catch (e: Exception) {
            Log.e(TAG, "Chair segmentation error: ${e.message}", e)
            return emptyList()
        }
    }

    // New function: Run segmentation specifically for cars
    private fun runSegmentationForCars(roi: Bitmap): List<Obstacle> {
        try {
            Log.d(TAG, "runSegmentationForCars: ROI dimensions: ${roi.width}x${roi.height}")
            val ti = TensorImage(DataType.FLOAT32).apply { load(roi) }
            val input = imageProcessor.process(ti).buffer
            val inputArray = FloatArray(input.capacity() / 4).apply { input.asFloatBuffer().get(this) }
            Log.d(TAG, "runSegmentationForCars: Input pixel range: min=${inputArray.minOrNull()}, max=${inputArray.maxOrNull()}")

            val detShape = segInterpreter.getOutputTensor(0).shape()
            val protoShape = segInterpreter.getOutputTensor(1).shape()
            val detOut = Array(detShape[0]) { Array(detShape[1]) { FloatArray(detShape[2]) } }
            val protoOut = Array(protoShape[0]) { Array(protoShape[1]) { Array(protoShape[2]) { FloatArray(protoShape[3]) } } }
            val outputs = mapOf(0 to detOut, 1 to protoOut)
            segInterpreter.runForMultipleInputsOutputs(arrayOf(input), outputs)

            val raw = detOut[0]
            val dets = mutableListOf<Triple<RectF, FloatArray, String>>()
            val threshold = 0.5f
            val numClasses = 80
            val maskCoefsCount = detShape[1] - 4 - numClasses
            val maskCoefsStartIdx = 4 + numClasses

            for (i in 0 until detShape[2]) {
                var maxClassProb = 0f
                var maxClassIdx = -1
                for (c in 0 until numClasses) {
                    if (classNames[c] == "car") {
                        val prob = raw[4 + c][i]
                        if (prob > maxClassProb) {
                            maxClassProb = prob
                            maxClassIdx = c
                        }
                    }
                }
                if (maxClassProb > threshold && maxClassIdx >= 0) {
                    val cx = raw[0][i] * roi.width
                    val cy = raw[1][i] * roi.height
                    val ww = raw[2][i] * roi.width
                    val hh = raw[3][i] * roi.height
                    val box = RectF(cx - ww / 2, cy - hh / 2, cx + ww / 2, cy + hh / 2)
                    val maskCoefs = FloatArray(maskCoefsCount) { c ->
                        val idx = maskCoefsStartIdx + c
                        if (idx < detShape[1]) raw[idx][i] else 0f
                    }
                    val className = classNames[maxClassIdx]
                    dets += Triple(box, maskCoefs, className)
                }
            }

            val finalDets = applyNMS(dets, 0.5f)
            val obstacles = mutableListOf<Obstacle>()
            val protoH = protoShape[1]
            val protoW = protoShape[2]
            val protoC = protoShape[3]

            for ((box, coefs, className) in finalDets) {
                try {
                    val mask = Array(256) { FloatArray(256) }
                    val maskValues = mutableListOf<Float>()
                    var activePixels = 0
                    for (dy in 0 until 256) {
                        for (dx in 0 until 256) {
                            val py = (dy * protoH / 256).coerceIn(0, protoH - 1)
                            val px = (dx * protoW / 256).coerceIn(0, protoW - 1)
                            var maskValue = 0f
                            for (c in 0 until minOf(coefs.size, protoC)) {
                                maskValue += coefs[c] * protoOut[0][py][px][c]
                            }
                            maskValue = 1.0f / (1.0f + exp(-maskValue))
                            maskValues.add(maskValue)
                            if (maskValue > 0.1f) {
                                mask[dy][dx] = 1f
                                activePixels++
                            }
                        }
                    }
                    if (activePixels >= 50) {
                        val dilatedMask = dilateArray(mask, 3)
                        obstacles.add(Obstacle(box, dilatedMask, className))
                        Log.d(TAG, "Car added: name=$className, size=$activePixels pixels")
                    } else {
                        Log.d(TAG, "Car skipped: name=$className, too small, size=$activePixels pixels")
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Error building mask for $className: ${e.message}", e)
                }
            }
            obstacles.forEach { Log.d(TAG, "Detected car: ${it.className}") }
            return obstacles
        } catch (e: Exception) {
            Log.e(TAG, "Car segmentation error: ${e.message}", e)
            return emptyList()
        }
    }



    private fun applyNMS(
        dets: List<Triple<RectF, FloatArray, String>>,
        iouThresh: Float
    ): List<Triple<RectF, FloatArray, String>> {
        if (dets.isEmpty()) return emptyList()
        val sorted = dets.sortedByDescending { it.second.sumByDouble { v: Float -> v.toDouble() }.toFloat() }
        val keep = mutableListOf<Triple<RectF, FloatArray, String>>()
        for ((box, coefs, className) in sorted) {
            if (keep.none { iou(it.first, box) > iouThresh }) {
                keep += Triple(box, coefs, className)
            }
        }
        return keep
    }

    private fun iou(a: RectF, b: RectF): Float {
        val left = max(a.left, b.left)
        val top = max(a.top, b.top)
        val right = min(a.right, b.right)
        val bottom = min(a.bottom, b.bottom)
        val inter = max(0f, right - left) * max(0f, bottom - top)
        val ua = a.width() * a.height() + b.width() * b.height() - inter
        return if (ua > 0) inter / ua else 0f
    }

    private fun runDepthEstimation(bmp: Bitmap): Array<FloatArray> {
        try {
            Log.d(TAG, "runDepthEstimation: Input bitmap dimensions: ${bmp.width}x${bmp.height}")
            val ti = TensorImage(DataType.FLOAT32)
            ti.load(bmp)
            val input = depthProcessor.process(ti).buffer
            val outShape = depthInterpreter.getOutputTensor(0).shape()
            val depthMap = if (outShape.size == 4) {
                val raw = Array(outShape[0]) {
                    Array(outShape[1]) {
                        Array(outShape[2]) {
                            FloatArray(outShape[3])
                        }
                    }
                }
                depthInterpreter.run(input, raw)
                Array(outShape[1]) { y -> FloatArray(outShape[2]) { x -> raw[0][y][x][0] } }
            } else if (outShape.size == 3) {
                val raw = Array(outShape[0]) { Array(outShape[1]) { FloatArray(outShape[2]) } }
                depthInterpreter.run(input, raw)
                raw[0]
            } else {
                Log.e(TAG, "Unsupported depth output shape: ${outShape.joinToString()}")
                Array(PROCESSING_SIZE) { FloatArray(PROCESSING_SIZE) { Float.MAX_VALUE } }
            }
            var minDepth = Float.MAX_VALUE
            var maxDepth = Float.MIN_VALUE
            for (y in depthMap.indices) {
                for (x in depthMap[0].indices) {
                    minDepth = min(minDepth, depthMap[y][x])
                    maxDepth = max(maxDepth, depthMap[y][x])
                }
            }
            Log.d(TAG, "runDepthEstimation: Depth map range: min=$minDepth, max=$maxDepth")
            return depthMap
        } catch (e: Exception) {
            Log.e(TAG, "Depth estimation error: ${e.message}", e)
            return Array(PROCESSING_SIZE) { FloatArray(PROCESSING_SIZE) { Float.MAX_VALUE } }
        }
    }

    private fun isImageMostlyUniform(bitmap: Bitmap?): Boolean {
        if (bitmap == null) return false
        val width = bitmap.width
        val height = bitmap.height
        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)
        var sum = 0.0
        val intensities = FloatArray(pixels.size)
        for (i in pixels.indices) {
            val pixel = pixels[i]
            val intensity = ((pixel shr 16 and 0xFF) * 0.299f) +
                    ((pixel shr 8 and 0xFF) * 0.587f) +
                    ((pixel and 0xFF) * 0.114f)
            intensities[i] = intensity
            sum += intensity
        }
        val mean = sum / pixels.size
        var variance = 0.0
        for (intensity in intensities) {
            val diff = intensity - mean
            variance += diff * diff
        }
        variance /= pixels.size
        val varianceThreshold = 1000f
        Log.d(TAG, "Image variance: $variance, threshold: $varianceThreshold")
        return variance < varianceThreshold
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<out String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == 1001) {
            if (grantResults.isNotEmpty() && grantResults.all { it == PackageManager.PERMISSION_GRANTED }) {
                startCamera()
                startVoiceRecognition() // Start SpeechRecognizer after permissions granted
                if (!isInternetAvailable()) {
                    Log.w(TAG, "No internet connection available")
                    initialOfflineWarningSent = true // Set flag for TTS warning
                }
            } else {
                val deniedPermissions = permissions.filterIndexed { index, permission ->
                    grantResults[index] != PackageManager.PERMISSION_GRANTED
                }
                val errorMessage = when {
                    deniedPermissions.contains(Manifest.permission.CAMERA) && deniedPermissions.contains(
                        Manifest.permission.RECORD_AUDIO
                    ) ->
                        "Camera and audio permissions denied"

                    deniedPermissions.contains(Manifest.permission.CAMERA) ->
                        "Camera permission denied"

                    deniedPermissions.contains(Manifest.permission.RECORD_AUDIO) ->
                        "Audio permission denied"

                    else -> "Required permissions denied"
                }
                Log.e(TAG, errorMessage)
                runOnUiThread { positionTextView.text = errorMessage }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        Log.d(TAG, "Destroying activity")
        handler.removeCallbacksAndMessages(null)
        if (::speechRecognizer.isInitialized) {
            speechRecognizer.stopListening()
            speechRecognizer.destroy()
            Log.d(TAG, "SpeechRecognizer destroyed")
        }
        if (::tflite.isInitialized) {
            tflite.close()
        }
        gpuDelegate?.close()
        if (::segInterpreter.isInitialized) {
            segInterpreter.close()
        }
        segGpuDelegate?.close()
        if (::depthInterpreter.isInitialized) {
            depthInterpreter.close()
        }
        depthGpuDelegate?.close()
        cameraExecutor.shutdown()
        if (::tts.isInitialized) {
            tts.shutdown()
        }
        cameraProvider?.unbindAll()
    }
}