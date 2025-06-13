package com.amb.aivision

import android.Manifest
import android.animation.Animator
import android.animation.AnimatorListenerAdapter
import android.annotation.SuppressLint
import android.content.Intent
import android.content.pm.PackageManager
import android.os.VibrationEffect
import android.os.Vibrator
import java.util.*
import android.content.res.Configuration
import android.graphics.*
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
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
import android.util.Size
import android.view.*
import android.widget.Button
import android.widget.ImageButton
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.annotation.OptIn
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.camera.core.*
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.constraintlayout.widget.ConstraintLayout
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.graphics.createBitmap
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
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

private const val TAG = "DoorDetection"

@SuppressLint("SetTextI18n")
class MainActivity : AppCompatActivity(), TextToSpeech.OnInitListener {

    companion object {
        private const val PROCESSING_SIZE = 256         // for seg & depth
        private const val PROXIMITY_THRESHOLD_M = 0.5f  // 0.75 meters for obstacles
        private const val PROXIMITY_THRESHOLD_D = 0.1f // Close to door
        private const val DETECTION_RESOLUTION = 640   // For door detection
        private const val DEPTH_SCALE_FACTOR = 100.0f   // For MiDaS depth to meters
        private const val DETECTION_INTERVAL_MS = 333L
        private const val ANIMATION_DURATION = 200L // Animation duration for camera switch
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
    private val detectionModes = listOf("door", "chair", "car")
    private var currentDetectionIndex = 0

    private val previewView: PreviewView by lazy { findViewById(R.id.previewView) }
    val positionTextView: TextView by lazy { findViewById(R.id.positionTextView) }
    private val detectButton: Button by lazy { findViewById(R.id.detectButton) }
    private val cameraSwitchOverlay: View by lazy { findViewById(R.id.cameraSwitchOverlay) }
    internal val swipeInstructionTextView: TextView by lazy { findViewById(R.id.swipeInstructionTextView) }
    private val chairButton: ImageButton by lazy { findViewById(R.id.chairButton) }
    private val carButton: ImageButton by lazy { findViewById(R.id.carButton) }
    private val doorButton: ImageButton by lazy { findViewById(R.id.doorButton) }
    private val lowLightWarningTextView: TextView by lazy { findViewById(R.id.lowLightWarningTextView) }

    private val leftArrowImageView: ImageView by lazy { findViewById(R.id.leftArrowImageView) }
    private val rightArrowImageView: ImageView by lazy { findViewById(R.id.rightArrowImageView) }
    private lateinit var speechRecognizer: SpeechRecognizer
    private lateinit var gestureDetector: GestureDetector
    private lateinit var tts: TextToSpeech

    private var lastDetectionTime = 0L
    private var isSpeaking = false
    private var isVoiceActive = false
    private var isRecognizerListening = false
    private var previousMessage: String? = null
    private var consecutiveIdenticalCount = 0

    private var shouldDetectDoors = false
    private var shouldDetectCars = false
    private var shouldDetectChairs = false
    private var shouldDetect = shouldDetectDoors || shouldDetectCars || shouldDetectChairs
    private var isDeepSceneDiscoveryActive = false

    private var canProcess = true
    private var useYolo12s = false
    private var isFirstLaunch = true

    private lateinit var tflite: Interpreter
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
    private var hasSpokenOfflineWarning = false
    private var hasSpokenDoorWarning = false
    private var wasDetectingBeforePause = false

    private var cameraProvider: ProcessCameraProvider? = null
    private var camera: androidx.camera.core.Camera? = null
    private var mainCameraSelector: CameraSelector? = null
    private var ultraWideCameraSelector: CameraSelector? = null
    private var activeCameraSelector: CameraSelector? = null
    private var isFlashOn = false
    private var isSwitchingCamera = false // Flag to prevent rapid switching



    private lateinit var deepSceneDiscovery: DeepSceneDiscovery

    private val handler = Handler(Looper.getMainLooper())
    private val detectionRunnable = object : Runnable {
        override fun run() {
            if (shouldDetect) {
                handler.postDelayed(this, DETECTION_INTERVAL_MS)
            }
        }
    }

    private var preview: Preview? = null
    private var analysis: ImageAnalysis? = null

    @SuppressLint("MissingPermission")
    private fun isInternetAvailable(): Boolean {
        val connectivityManager =
            getSystemService(CONNECTIVITY_SERVICE) as ConnectivityManager
        val network = connectivityManager.activeNetwork ?: return false
        val capabilities = connectivityManager.getNetworkCapabilities(network) ?: return false
        return capabilities.hasCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET) &&
                capabilities.hasCapability(NetworkCapabilities.NET_CAPABILITY_VALIDATED)
    }

    private var numDetections: Int = 0
    private var isWaitingForIconSelection = false

    @SuppressLint("ClickableViewAccessibility")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        if (OpenCVLoader.initLocal()) {
            org.opencv.core.Core.setNumThreads(1)
            Log.d(TAG, "OpenCV loaded successfully")
        } else {
            Log.e(TAG, "Failed to load OpenCV")
        }
        setContentView(R.layout.activity_main)
        previewView.implementationMode = PreviewView.ImplementationMode.PERFORMANCE

        window.decorView.setOnApplyWindowInsetsListener { _, insets ->
            val topInset = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                insets.getInsets(WindowInsets.Type.statusBars() or WindowInsets.Type.displayCutout()).top
            } else if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                @Suppress("DEPRECATION")
                insets.displayCutout?.safeInsetTop ?: insets.systemWindowInsetTop
            } else {
                (24 * resources.displayMetrics.density).toInt()
            }
            (swipeInstructionTextView.layoutParams as ConstraintLayout.LayoutParams).apply {
                topMargin = topInset + (8 * resources.displayMetrics.density).toInt()
            }
            swipeInstructionTextView.requestLayout()
            insets
        }

        detectButton.text = "Start Detection"
        chairButton.visibility = View.GONE
        carButton.visibility = View.GONE
        doorButton.visibility = View.GONE

        cameraExecutor = Executors.newSingleThreadExecutor()
        deepSceneDiscovery = DeepSceneDiscovery(this)

        setupGestureDetector()

        previewView.setOnTouchListener { _, event ->
            gestureDetector.onTouchEvent(event)
            true
        }

        doorButton.setOnClickListener { startDetection("door") }
        chairButton.setOnClickListener { startDetection("chair") }
        carButton.setOnClickListener { startDetection("car") }
        detectButton.setOnClickListener { toggleDetection() }
        detectButton.setOnLongClickListener {
            toggleDoorModel()
            true
        }

        // --- MODIFICATION: Check if launched as an assistant ---
        if (intent?.action == Intent.ACTION_ASSIST) {
            isVoiceActive = true
        }

        requestPermissions()
    }

    private fun requestPermissions() {
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

    private fun cycleDetection(goForward: Boolean) {
        if (goForward) { // Swipe Right to move to the next item
            currentDetectionIndex++
            if (currentDetectionIndex >= detectionModes.size) {
                currentDetectionIndex = 0 // Wrap around to the start
            }
        } else { // Swipe Left to move to the previous item
            currentDetectionIndex--
            if (currentDetectionIndex < 0) {
                currentDetectionIndex = detectionModes.size - 1 // Wrap around to the end
            }
        }
        val newDetectionType = detectionModes[currentDetectionIndex]
        startDetection(newDetectionType)
    }

    private fun setupGestureDetector() {
        gestureDetector = GestureDetector(this, object : GestureDetector.SimpleOnGestureListener() {
            override fun onFling(
                e1: MotionEvent?, e2: MotionEvent, velocityX: Float, velocityY: Float
            ): Boolean {
                if (e1 == null || isSwitchingCamera) return false

                val deltaX = e2.x - e1.x
                val deltaY = e2.y - e1.y

                if (abs(deltaX) > abs(deltaY)) {
                    // --- HORIZONTAL SWIPE LOGIC ---
                    if (abs(deltaX) > 100 && abs(velocityX) > 100) {
                        if (shouldDetect) {
                            if (deltaX > 0) {
                                // Swipe Right
                                cycleDetection(goForward = true)
                            } else {
                                // Swipe Left
                                cycleDetection(goForward = false)
                            }
                        }
                        return true
                    }
                } else {
                    if (abs(deltaY) > 200 && abs(velocityY) > 100) {
                        if (deltaY > 0) { // Swipe Down
                            if (shouldDetect || isDeepSceneDiscoveryActive) {
                                if (isDeepSceneDiscoveryActive) stopDeepSceneDiscovery() else stopDetection()
                                speak("Stopping detection.")
                            }
                        }
                        return true
                    }
                }
                return false
            }

            override fun onDoubleTap(e: MotionEvent): Boolean {
                if (isSwitchingCamera) return true

                isFlashOn = !isFlashOn

                if (isFlashOn) {
                    val targetSelector = mainCameraSelector
                    if (targetSelector == null) {
                        speak("Flash not available.")
                        isFlashOn = false
                        return true
                    }
                    speak("Flash on")
                    if (activeCameraSelector == targetSelector) {
                        camera?.cameraControl?.enableTorch(true)
                    } else {
                        animateAndSwitchCamera(targetSelector, turnFlashOn = true)
                    }
                } else {
                    val targetSelector = ultraWideCameraSelector
                    if (targetSelector == null) {
                        camera?.cameraControl?.enableTorch(false)
                        speak("Flash off")
                        return true
                    }
                    speak("Flash off")
                    if (activeCameraSelector == targetSelector) {
                        camera?.cameraControl?.enableTorch(false)
                    } else {
                        camera?.cameraControl?.enableTorch(false)
                        animateAndSwitchCamera(targetSelector, turnFlashOn = false)
                    }
                }
                return true
            }
        })
    }

    private fun initializeComponents() {
        Log.d(TAG, "Initializing components")
        setupFullscreenUI()
        setupProcessors()
        runOnUiThread {
            if (!loadModels()) {
                positionTextView.text = "Failed to load models. Please check the app configuration."
                detectButton.isEnabled = false
                return@runOnUiThread
            }
        }

        if (::tts.isInitialized) tts.shutdown()
        tts = TextToSpeech(this, this)
        initSpeechRecognizer()

        if (!isInternetAvailable()) {
            Log.w(TAG, "No internet connection available")
            initialOfflineWarningSent = true
        }
        startCamera()

        handler.postDelayed({
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
                startVoiceRecognition()
            }
        }, 100L)
    }

    private fun isLowLight(bitmap: Bitmap, threshold: Int = 60): Boolean {
        val pixels = IntArray(bitmap.width * bitmap.height)
        bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        var totalLuminance = 0.0
        for (pixel in pixels) {
            val r = Color.red(pixel)
            val g = Color.green(pixel)
            val b = Color.blue(pixel)
            totalLuminance += (0.299 * r + 0.587 * g + 0.114 * b)
        }
        val avgLuminance = totalLuminance / pixels.size
        return avgLuminance < threshold
    }

    private fun triggerHapticFeedback() {
        val vibrator = getSystemService(Vibrator::class.java)
        if (vibrator.hasVibrator()) {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                val vibrationEffect = VibrationEffect.createOneShot(1000, VibrationEffect.DEFAULT_AMPLITUDE)
                vibrator.vibrate(vibrationEffect)
            } else {
                @Suppress("DEPRECATION")
                vibrator.vibrate(1000)
            }
        }
    }

    @OptIn(ExperimentalCamera2Interop::class)
    private fun findCameraSelectors() {
        val cameraProvider = this.cameraProvider ?: return
        val cameraManager = getSystemService(CAMERA_SERVICE) as CameraManager

        val backCameras = cameraProvider.availableCameraInfos.filter {
            it.lensFacing == CameraSelector.LENS_FACING_BACK
        }

        if (backCameras.isEmpty()) {
            Log.e(TAG, "No back cameras found!")
            val defaultSelector = CameraSelector.DEFAULT_BACK_CAMERA
            mainCameraSelector = defaultSelector
            ultraWideCameraSelector = defaultSelector
            return
        }

        val mainCamInfo = backCameras.firstOrNull { it.hasFlashUnit() } ?: backCameras.first()
        mainCameraSelector = CameraSelector.Builder().addCameraFilter {
            it.filter { camInfo -> camInfo == mainCamInfo }
        }.build()

        var ultraWideCamInfo: CameraInfo? = mainCamInfo // Default to main
        var minFocalLength = Float.MAX_VALUE
        for (cameraInfo in backCameras) {
            try {
                val characteristics = cameraManager.getCameraCharacteristics(Camera2CameraInfo.from(cameraInfo).cameraId)
                val focalLengths = characteristics.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)
                val currentMinFocal = focalLengths?.minOrNull() ?: Float.MAX_VALUE
                if (currentMinFocal < minFocalLength) {
                    minFocalLength = currentMinFocal
                    ultraWideCamInfo = cameraInfo
                }
            } catch (e: Exception) {
                Log.e(TAG, "Could not get characteristics for a camera.", e)
            }
        }

        ultraWideCameraSelector = CameraSelector.Builder().addCameraFilter {
            it.filter { camInfo -> camInfo == ultraWideCamInfo }
        }.build()
    }

    private fun animateAndSwitchCamera(selector: CameraSelector, turnFlashOn: Boolean) {
        if (isSwitchingCamera) return
        isSwitchingCamera = true

        cameraSwitchOverlay.visibility = View.VISIBLE
        cameraSwitchOverlay.animate()
            .alpha(1f)
            .setDuration(ANIMATION_DURATION)
            .setListener(object : AnimatorListenerAdapter() {
                override fun onAnimationEnd(animation: Animator) {
                    // Switch camera now that the screen is black
                    bindCameraUseCases(selector, turnFlashOn)
                }
            })
            .start()
    }

    private fun bindCameraUseCases(selector: CameraSelector, turnFlashOn: Boolean) {
        val cameraProvider = this.cameraProvider ?: run {
            isSwitchingCamera = false
            return
        }
        try {
            cameraProvider.unbindAll()

            camera = cameraProvider.bindToLifecycle(
                this,
                selector,
                preview,
                analysis
            )

            this.activeCameraSelector = selector

            if (turnFlashOn) {
                camera?.cameraControl?.enableTorch(true)
            }

            handler.postDelayed({
                cameraSwitchOverlay.animate()
                    .alpha(0f)
                    .setDuration(ANIMATION_DURATION)
                    .setListener(object : AnimatorListenerAdapter() {
                        override fun onAnimationEnd(animation: Animator) {
                            cameraSwitchOverlay.visibility = View.GONE
                            isSwitchingCamera = false // Allow switching again
                        }
                    })
                    .start()
            }, 700)

        } catch (exc: Exception) {
            Log.e(TAG, "Use case binding failed", exc)
            isSwitchingCamera = false
        }
    }

    @OptIn(ExperimentalCamera2Interop::class)
    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            cameraProvider = cameraProviderFuture.get()

            val rotation = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                display?.rotation ?: Surface.ROTATION_0
            } else {
                @Suppress("DEPRECATION")
                windowManager.defaultDisplay.rotation
            }

            preview = Preview.Builder().setTargetRotation(rotation).build()
                .also { it.surfaceProvider = previewView.surfaceProvider }

            analysis = ImageAnalysis.Builder()
                .setTargetRotation(rotation)
                .setResolutionSelector(
                    ResolutionSelector.Builder().setResolutionStrategy(
                        ResolutionStrategy(
                            Size(640, 640),
                            ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER
                        )
                    ).build()
                )
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
                .also { it.setAnalyzer(cameraExecutor, ::onFrame) }

            findCameraSelectors()
            val initialSelector = ultraWideCameraSelector ?: mainCameraSelector ?: CameraSelector.DEFAULT_BACK_CAMERA
            bindCameraUseCases(initialSelector, false)

        }, ContextCompat.getMainExecutor(this))
    }

    //region Unchanged Methods
    private fun startVoiceRecognition() {
        if (isRecognizerListening || isSpeaking || ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            Log.d(TAG, "startVoiceRecognition skipped: isRecognizerListening=$isRecognizerListening, isSpeaking=$isSpeaking, permissionGranted=${ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED}")
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
                runOnUiThread { positionTextView.text = "Audio permission denied. Please grant permission." }
            }
            return
        }

        if (!isInternetAvailable()) {
            Log.w(TAG, "No internet connection for voice recognition")
            runOnUiThread {
                detectButton.visibility = View.VISIBLE
                detectButton.isEnabled = true
                if (!hasSpokenOfflineWarning) {
                    speak("No internet connection. Voice recognition unavailable.")
                    hasSpokenOfflineWarning = true
                }
            }
            return // Exit early instead of retrying
        }

        hasSpokenOfflineWarning = false
        runOnUiThread {
            detectButton.visibility = View.GONE
            detectButton.isEnabled = true
            if (!hasSpokenDoorWarning) {
                positionTextView.text = "Listening..."
            }
        }

        if (!SpeechRecognizer.isRecognitionAvailable(this)) {
            Log.e(TAG, "Speech recognition not available on this device")
            runOnUiThread { positionTextView.text = "Speech recognition not supported on this device" }
            handler.postDelayed({ startVoiceRecognition() }, 1000L)
            return
        }

        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.getDefault().toLanguageTag())
            putExtra(RecognizerIntent.EXTRA_SPEECH_INPUT_MINIMUM_LENGTH_MILLIS, 15000L) // Increase to 15 seconds
            putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
            putExtra(RecognizerIntent.EXTRA_PREFER_OFFLINE, true)
        }
        try {
            Log.d(TAG, "Starting voice recognition with intent: $intent")
            speechRecognizer.startListening(intent)
            isRecognizerListening = true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to start voice recognition: ${e.message}", e)
            isRecognizerListening = false
            handler.postDelayed({ startVoiceRecognition() }, 100L)
        }
    }

    private fun initSpeechRecognizer() {
        if (::speechRecognizer.isInitialized) {
            Log.d(TAG, "Destroying existing SpeechRecognizer")
            speechRecognizer.stopListening()
            speechRecognizer.destroy()
        }
        speechRecognizer = SpeechRecognizer.createSpeechRecognizer(this)
        Log.d(TAG, "Created new SpeechRecognizer: $speechRecognizer")
        speechRecognizer.setRecognitionListener(object : RecognitionListener {
            override fun onReadyForSpeech(params: Bundle?) {
                isRecognizerListening = true
                Log.d(TAG, "SpeechRecognizer ready for speech")
            }

            override fun onBeginningOfSpeech() {
                Log.d(TAG, "SpeechRecognizer detected beginning of speech")
            }

            override fun onRmsChanged(rmsdB: Float) {}
            override fun onBufferReceived(buffer: ByteArray?) {}

            override fun onEndOfSpeech() {
                isRecognizerListening = false
                Log.d(TAG, "SpeechRecognizer detected end of speech")
                if (!isSpeaking && (isVoiceActive || !shouldDetect)) {
                    startVoiceRecognition()
                }
            }

            override fun onError(error: Int) {
                isRecognizerListening = false
                Log.e(TAG, "SpeechRecognizer error: $error")
                when (error) {
                    SpeechRecognizer.ERROR_NO_MATCH, SpeechRecognizer.ERROR_SPEECH_TIMEOUT -> {
                        if (!isSpeaking && (isVoiceActive || !shouldDetect)) {
                            handler.postDelayed({ startVoiceRecognition() }, 1000L) // Add 1s delay
                        }
                    }
                    else -> {
                        if (!isSpeaking && (isVoiceActive || !shouldDetect)) {
                            handler.postDelayed({ startVoiceRecognition() }, 100L)
                        }
                    }
                }
            }

            override fun onResults(results: Bundle?) {
                val matches = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                Log.d(TAG, "SpeechRecognizer results: $matches")
                if (!matches.isNullOrEmpty()) {
                    val spokenText = matches[0]
                    if (!isSpeaking) {
                        processVoiceCommand(spokenText)
                    }
                }
                if (!isSpeaking && (isVoiceActive || !shouldDetect)) {
                    startVoiceRecognition()
                }
            }

            override fun onPartialResults(partialResults: Bundle?) {
                Log.d(TAG, "Partial results: ${partialResults?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)}")
            }

            override fun onEvent(eventType: Int, params: Bundle?) {}
        })
    }

    override fun onResume() {
        super.onResume()
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
            initSpeechRecognizer()
        }

        if (isFirstLaunch) {
            isFirstLaunch = false
            shouldDetect = false
            shouldDetectDoors = false
            shouldDetectCars = false
            shouldDetectChairs = false
            isDeepSceneDiscoveryActive = false
            detectButton.text = "Detect"
            runOnUiThread { positionTextView.text = "Waiting for voice commands..." }
            handler.postDelayed({ startVoiceRecognition() }, 100L)
        } else {
            if (wasDetectingBeforePause) {
                stopDetection()
                stopDeepSceneDiscovery()
                wasDetectingBeforePause = false
            } else {
                handler.postDelayed({ startVoiceRecognition() }, 100L)
            }
        }
    }

    override fun onStop() {
        super.onStop()
        if (::speechRecognizer.isInitialized) {
            Log.d(TAG, "Destroying SpeechRecognizer in onStop")
            speechRecognizer.destroy()
        }
    }

    override fun onPause() {
        super.onPause()
        wasDetectingBeforePause = shouldDetect || isDeepSceneDiscoveryActive
        shouldDetect = false
        handler.removeCallbacks(detectionRunnable)

        if (::speechRecognizer.isInitialized && isRecognizerListening) {
            speechRecognizer.stopListening()
            isRecognizerListening = false
        }

        if (::tts.isInitialized && isSpeaking) {
            tts.stop()
            isSpeaking = false
        }
    }

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            tts.language = Locale.US
            tts.setSpeechRate(1.25f)
            tts.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
                override fun onStart(utteranceId: String?) {
                    isSpeaking = true
                    if (isRecognizerListening) {
                        speechRecognizer.stopListening()
                        isRecognizerListening = false
                    }
                }
                override fun onDone(utteranceId: String?) {
                    isSpeaking = false
                    if (isDeepSceneDiscoveryActive) {
                        deepSceneDiscovery.onSpeechFinished()
                    }
                    if ((isVoiceActive || !shouldDetect) && !isRecognizerListening) {
                        handler.post { startVoiceRecognition() }
                    }
                }
                @Deprecated("Deprecated in Java")
                override fun onError(utteranceId: String?) {
                    isSpeaking = false
                    if (isDeepSceneDiscoveryActive) {
                        deepSceneDiscovery.onSpeechFinished()
                    }
                    if ((isVoiceActive || !shouldDetect) && !isRecognizerListening) {
                        handler.post { startVoiceRecognition() }
                    }
                }
            })

            // --- ADD THIS LOGIC ---
            // Now that TTS is ready, check if we need to perform the initial greeting.
            if (isVoiceActive && !hasGreeted) {
                speak("Hello, how can I help you?")
                hasGreeted = true // Mark as greeted to prevent this from repeating.
                runOnUiThread { positionTextView.text = "Voice activated, say 'doors', 'cars', 'chairs', 'deep scene discovery', or 'stop'" }
            }

            if (initialOfflineWarningSent && !isInternetAvailable()) {
                speak("No internet connection. Some features may not work.")
            }
        } else {
            Log.e(TAG, "TTS initialization failed")
            runOnUiThread { positionTextView.text = "TTS initialization failed" }
        }
    }

    private fun processVoiceCommand(command: String) {
        if (command.isBlank() || command.lowercase(Locale.getDefault()) in listOf(
                "hello, how can i help you", "starting detection, press on the icon that you want to detect",
                "starting detecting doors", "starting detecting cars", "starting detecting chairs", "stopping detection",
                "no internet connection. voice recognition is unavailable. use the button to detect",
                "no internet connection. some features like voice recognition may not work",
                "starting deep scene discovery", "stopping deep scene discovery")) {
            return
        }
        if (isSpeaking) {
            return
        }
        when {
            command.lowercase(Locale.getDefault()).contains("hello") || command.lowercase(Locale.getDefault()).contains("vai") || command.lowercase(Locale.getDefault()).contains("hey") || command.lowercase(Locale.getDefault()).contains("vi") || command.lowercase(Locale.getDefault()).contains("hi") || command.lowercase(Locale.getDefault()).contains("voi") -> {
                isVoiceActive = true
                if (!hasGreeted) {
                    speak("Hello, how can I help you?")
                    hasGreeted = true
                }
                runOnUiThread { positionTextView.text = "Voice activated, say 'doors', 'cars', 'chairs', 'deep scene discovery', or 'stop'" }
            }
            command.lowercase(Locale.getDefault()).contains("door") && !isDeepSceneDiscoveryActive -> {
                if (!shouldDetectDoors) {
                    startDetection("door")
                }
            }
            command.lowercase(Locale.getDefault()).contains("car") && !isDeepSceneDiscoveryActive -> {
                if (!shouldDetectCars) {
                    startDetection("car")
                }
            }
            command.lowercase(Locale.getDefault()).contains("chair") && !isDeepSceneDiscoveryActive -> {
                if (!shouldDetectChairs) {
                    startDetection("chair")
                }
            }
            command.lowercase(Locale.getDefault()).contains("deep") || command.lowercase(Locale.getDefault()).contains("scene") || command.lowercase(Locale.getDefault()).contains("discover") -> {
                if (!isDeepSceneDiscoveryActive) {
                    startDeepSceneDiscovery()
                }
            }
            command.lowercase(Locale.getDefault()).contains("stop") && isDeepSceneDiscoveryActive -> {
                stopDeepSceneDiscovery()
            }
            command.lowercase(Locale.getDefault()).contains("stop") && !isDeepSceneDiscoveryActive -> {
                stopDetection()
            }
        }
    }

    private fun setupFullscreenUI() {
        WindowCompat.setDecorFitsSystemWindows(window, false)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            window.insetsController?.let {
                it.hide(WindowInsets.Type.statusBars() or WindowInsets.Type.displayCutout())
                it.systemBarsBehavior = WindowInsetsController.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
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

    private fun loadModels(): Boolean {
        try {
            val compatList = CompatibilityList()
            var useGpu = compatList.isDelegateSupportedOnThisDevice

            val modelFile = if (useYolo12s) "yolo12s.tflite" else "yolo8n.tflite"
            val model = try {
                FileUtil.loadMappedFile(this, modelFile)
            } catch (e: Exception) {
                Log.e(TAG, "Failed to load YOLO model $modelFile: ${e.message}", e)
                runOnUiThread {
                    positionTextView.text = "Error loading YOLO model: ${e.message}"
                }
                return false
            }

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
                } catch (e: Exception) {
                    Log.w(TAG, "GPU delegate failed for YOLO: ${e.message}. Falling back to CPU.", e)
                    useGpu = false
                }
            }
            if (!useGpu) {
                val cpuOptions = Interpreter.Options().apply {
                    numThreads = min(Runtime.getRuntime().availableProcessors(), 4)
                    useNNAPI = false
                }
                tflite = Interpreter(model, cpuOptions)
            }
            numDetections = tflite.getOutputTensor(0).shape()[2]

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
                } catch (e: Exception) {
                    Log.w(TAG, "GPU delegate failed for seg: ${e.message}. Falling back to CPU.", e)
                    useGpu = false
                }
            }
            if (!useGpu) {
                val cpuOptions = Interpreter.Options().apply {
                    numThreads = min(Runtime.getRuntime().availableProcessors(), 4)
                    useNNAPI = false
                }
                segInterpreter = Interpreter(segModel, cpuOptions)
            }

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
                } catch (e: Exception) {
                    Log.w(TAG, "GPU delegate failed for depth: ${e.message}. Falling back to CPU.", e)
                    useGpu = false
                }
            }
            if (!useGpu) {
                val cpuOptions = Interpreter.Options().apply {
                    numThreads = min(Runtime.getRuntime().availableProcessors(), 4)
                    useNNAPI = false
                }
                depthInterpreter = Interpreter(depthModel, cpuOptions)
            }

            return true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load models: ${e.message}", e)
            runOnUiThread { positionTextView.text = "Error loading models: ${e.message}" }
            return false
        }
    }


    override fun onConfigurationChanged(newConfig: Configuration) {
        super.onConfigurationChanged(newConfig)
        val rotation = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            display?.rotation ?: Surface.ROTATION_0
        } else {
            @Suppress("DEPRECATION")
            windowManager.defaultDisplay.rotation
        }
        preview?.targetRotation = rotation
        analysis?.targetRotation = rotation
    }

    private fun toggleDetection() {
        if (shouldDetect || isWaitingForIconSelection || isDeepSceneDiscoveryActive) {
            if (isDeepSceneDiscoveryActive) {
                stopDeepSceneDiscovery()
            } else {
                stopDetection()
            }
        } else {
            startDetection()
        }
    }

    private fun toggleDoorModel() {
        useYolo12s = !useYolo12s
        val modelName = if (useYolo12s) "yolo12s" else "yolo8n"
        if (loadModels()) {
            Toast.makeText(this, "Changed model to $modelName", Toast.LENGTH_SHORT).show()
        } else {
            Toast.makeText(this, "Failed to change model to $modelName", Toast.LENGTH_SHORT).show()
            useYolo12s = !useYolo12s
        }
    }

    private fun onFrame(image: ImageProxy) {
        val currentTime = System.currentTimeMillis()
        if ((!shouldDetect && !isDeepSceneDiscoveryActive) || !canProcess || currentTime - lastDetectionTime < DETECTION_INTERVAL_MS) {
            image.close()
            return
        }

        canProcess = false
        lastDetectionTime = currentTime
        try {
            val bmp = imageProxyToBitmap(image)
            image.close()

            if (isLowLight(bmp) && !isFlashOn) {
                runOnUiThread {
                    lowLightWarningTextView.visibility = View.VISIBLE
                }
            } else {
                runOnUiThread {
                    lowLightWarningTextView.visibility = View.GONE
                }
            }

            if (isDeepSceneDiscoveryActive) {
                deepSceneDiscovery.processFrame(bmp)
                canProcess = true
                return
            }

            if (isImageMostlyUniform(bmp)) {
                val msg = "You are going to hit something."
                handleMessage(msg)
                triggerHapticFeedback()
                canProcess = true
                return
            }

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

            val fullDepthMap = runDepthEstimation(bmp)
            val obstacles = runSegmentation(bmp)
            val mappedObstacles = obstacles.map { obstacle ->
                val mappedMask = mapMaskToOriginal(obstacle.mask, bmp.width, bmp.height)
                Obstacle(obstacle.box, mappedMask, obstacle.className)
            }.filter {
                when {
                    shouldDetectDoors -> it.className != "door"
                    shouldDetectChairs -> it.className != "chair"
                    shouldDetectCars -> it.className != "car"
                    else -> true
                }
            }

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
                val closeObstacles = mappedObstacles.filter { obstacle ->
                    val obstacleDepth = avgMaskDepthFixed(fullDepthMap, obstacle.mask)
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

    private fun imageProxyToBitmap(image: ImageProxy): Bitmap {
        val plane = image.planes[0]
        val buffer = plane.buffer
        val pixelStride = plane.pixelStride
        val rowStride = plane.rowStride
        val rowPadding = rowStride - pixelStride * image.width

        val bitmap = createBitmap(image.width + rowPadding / pixelStride, image.height)
        bitmap.copyPixelsFromBuffer(buffer)

        val croppedBitmap = Bitmap.createBitmap(bitmap, 0, 0, image.width, image.height)

        val rotationDegrees = image.imageInfo.rotationDegrees
        return if (rotationDegrees != 0) {
            val matrix = Matrix().apply { postRotate(rotationDegrees.toFloat()) }
            Bitmap.createBitmap(croppedBitmap, 0, 0, croppedBitmap.width, croppedBitmap.height, matrix, true)
        } else {
            croppedBitmap
        }
    }

    private fun handleMessage(message: String) {
        if (message == previousMessage) {
            consecutiveIdenticalCount++
            if (consecutiveIdenticalCount >= 5) {
                consecutiveIdenticalCount = 0
                previousMessage = null
            }
        } else {
            consecutiveIdenticalCount = 0
            previousMessage = message
        }

        if (consecutiveIdenticalCount == 1) {
            speak(message)
            // This block no longer hides the low-light warning
            runOnUiThread {
                positionTextView.visibility = View.VISIBLE
                positionTextView.text = message
            }
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

    fun speak(msg: String) {
        tts.stop()
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
        val proximityThreshold = PROXIMITY_THRESHOLD_D
        if (depthMeters < proximityThreshold) {
            return "You have reached the $targetClass."
        }

        val blockingObstacles = obstacles.filter { obstacle ->
            val obstacleDepth = avgMaskDepthFixed(depthMap, obstacle.mask)
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
                val rightHalf = Bitmap.createBitmap(bitmap, bitmap.width / 2, 0, bitmap.width / 2, bitmap.height)
                val rightObstacles = runSegmentation(rightHalf).filter { it.className != targetClass }
                if (rightObstacles.isEmpty()) {
                    "The $targetClass is to your left, but there is $obstacleNames in the way. Move right to avoid it, then turn left."
                } else {
                    val rightObstacleNames = rightObstacles.joinToString(" and ") { it.className }
                    "The $targetClass is to your left, but there is $obstacleNames in the way. The right path is blocked by $rightObstacleNames."
                }
            }
            "right" -> {
                val leftHalf = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width / 2, bitmap.height)
                val leftObstacles = runSegmentation(leftHalf).filter { it.className != targetClass }
                if (leftObstacles.isEmpty()) {
                    "The $targetClass is to your right, but there is $obstacleNames in the way. Move left to avoid it, then turn right."
                } else {
                    val leftObstacleNames = leftObstacles.joinToString(" and ") { it.className }
                    "The $targetClass is to your right, but there is $obstacleNames in the way. The left path is blocked by $leftObstacleNames."
                }
            }
            else -> {
                val rightThird = Bitmap.createBitmap(bitmap, bitmap.width * 2 / 3, 0, bitmap.width / 3, bitmap.height)
                val rightObstacles = runSegmentation(rightThird).filter { it.className != targetClass }
                if (rightObstacles.isEmpty()) {
                    "The $targetClass is straight ahead, but there is $obstacleNames in the way. Move right to avoid it, then continue forward."
                } else {
                    val leftThird = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width / 3, bitmap.height)
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
        mask: Array<FloatArray>
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

    private fun dilateArray(array: Array<FloatArray>): Array<FloatArray> {
        val result = Array(array.size) { FloatArray(array[0].size) }
        val radius = 1 // Since kernelSize is always 3, radius is always 1
        for (y in array.indices) {
            for (x in array[0].indices) {
                var value = 0f
                // The loops now go from -1 to 1
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

    private fun startDetection(type: String? = null) {
        if (type == null) {
            swipeInstructionTextView.visibility = View.VISIBLE
            isWaitingForIconSelection = true
            shouldDetect = false
            // Hide arrows when waiting for icon selection
            runOnUiThread {
                leftArrowImageView.visibility = View.GONE
                rightArrowImageView.visibility = View.GONE
                detectButton.text = "Stop Detection"
                chairButton.visibility = View.VISIBLE
                carButton.visibility = View.VISIBLE
                doorButton.visibility = View.VISIBLE
                positionTextView.text = "Choose detection type"
            }
            speak("Starting detection, press on the icon that you want to detect")
        } else {
            currentDetectionIndex = detectionModes.indexOf(type).coerceAtLeast(0)
            swipeInstructionTextView.visibility = View.VISIBLE
            isWaitingForIconSelection = false
            shouldDetect = true
            shouldDetectDoors = type == "door"
            shouldDetectCars = type == "car"
            shouldDetectChairs = type == "chair"
            runOnUiThread {
                detectButton.text = "Stop Detection"
                // Show arrows now that a specific detection is active
                leftArrowImageView.visibility = View.VISIBLE
                rightArrowImageView.visibility = View.VISIBLE
                chairButton.visibility = View.GONE
                carButton.visibility = View.GONE
                doorButton.visibility = View.GONE
                positionTextView.text = "Detecting ${type}s"
            }
            speak("Detecting ${type}s")
            handler.removeCallbacks(detectionRunnable)
            handler.post(detectionRunnable)
        }
    }

    private fun stopDetection() {
        swipeInstructionTextView.visibility = View.GONE
        shouldDetect = false
        shouldDetectDoors = false
        shouldDetectCars = false
        shouldDetectChairs = false
        isWaitingForIconSelection = false
        isVoiceActive = false
        handler.removeCallbacks(detectionRunnable)
        runOnUiThread {
            detectButton.text = "Start Detection"
            leftArrowImageView.visibility = View.GONE
            rightArrowImageView.visibility = View.GONE
            chairButton.visibility = View.GONE
            carButton.visibility = View.GONE
            doorButton.visibility = View.GONE
            positionTextView.text = "Detection stopped"
        }
    }

    private fun startDeepSceneDiscovery() {
        swipeInstructionTextView.visibility = View.VISIBLE
        stopDetection()
        isDeepSceneDiscoveryActive = true
        runOnUiThread {
            detectButton.text = "Stop Detection"
            positionTextView.text = "Deep Scene Discovery active"
        }
        deepSceneDiscovery.start()
    }

    private fun stopDeepSceneDiscovery() {
        swipeInstructionTextView.visibility = View.GONE
        isDeepSceneDiscoveryActive = false
        deepSceneDiscovery.stop()
        isVoiceActive = false
        hasGreeted = false
        runOnUiThread {
            detectButton.text = "Start Detection"
            positionTextView.text = "Waiting for voice commands..."
        }
    }

    private fun detectDoor(bitmap: Bitmap): Pair<RectF?, String> {
        val tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(bitmap)
        val processedImage = detectionProcessor.process(tensorImage)
        val inputBuffer = processedImage.buffer
        inputBuffer.rewind()
        val outputs = Array(1) { Array(5) { FloatArray(8400) } }
        tflite.run(inputBuffer, outputs)
        val threshold = 0.75f
        val iouThresh = 0.6f
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
                return Pair(best.first, best.third)
            } else if (keep.size > 1) {
                val secondBest = keep[1]
                val secondCroppedBitmap = cropBitmap(bitmap, secondBest.first)
                if (confirmDoorWithClassicalMethods(secondCroppedBitmap)) {
                    return Pair(secondBest.first, secondBest.third)
                }
            }
            return Pair(best.first, best.third)
        }
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
            return Pair(bestChair.box, position)
        }
        return Pair(null, "")
    }

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
            return Pair(bestCar.box, position)
        }
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
        Imgproc.findContours(edges, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
        for (contour in contours) {
            val approx = MatOfPoint2f()
            Imgproc.approxPolyDP(MatOfPoint2f(*contour.toArray()), approx, Imgproc.arcLength(MatOfPoint2f(*contour.toArray()), true) * 0.02, true)
            if (approx.toArray().size == 4) {
                val points = approx.toArray()
                val rect = Imgproc.boundingRect(MatOfPoint(*points))
                val aspectRatio = rect.height.toFloat() / rect.width
                if (aspectRatio in 1.5..3.0 && verticalLines >= 2 && horizontalLines >= 2) {
                    return true
                }
            }
        }
        return false
    }

    data class Obstacle(val box: RectF, val mask: Array<FloatArray>, val className: String) {
        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (javaClass != other?.javaClass) return false

            other as Obstacle

            if (box != other.box) return false
            if (!mask.contentDeepEquals(other.mask)) return false
            if (className != other.className) return false

            return true
        }

        override fun hashCode(): Int {
            var result = box.hashCode()
            result = 31 * result + mask.contentDeepHashCode()
            result = 31 * result + className.hashCode()
            return result
        }
    }

    private fun runSegmentation(roi: Bitmap): List<Obstacle> {
        try {
            val ti = TensorImage(DataType.FLOAT32)
            ti.load(roi)
            val input = imageProcessor.process(ti).buffer
            val detShape = segInterpreter.getOutputTensor(0).shape()
            val protoShape = segInterpreter.getOutputTensor(1).shape()
            val detOut = Array(detShape[0]) { Array(detShape[1]) { FloatArray(detShape[2]) } }
            val protoOut = Array(protoShape[0]) { Array(protoShape[1]) { Array(protoShape[2]) { FloatArray(protoShape[3]) } } }
            val outputs = mapOf(0 to detOut, 1 to protoOut)
            segInterpreter.runForMultipleInputsOutputs(arrayOf(input), outputs)
            val raw = detOut[0]
            val dets = mutableListOf<Triple<RectF, FloatArray, String>>()
            val threshold = 0.75f
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
                    val className = if (maxClassIdx >= 0 && maxClassIdx < classNames.size) classNames[maxClassIdx] else "Unknown"
                    dets += Triple(box, maskCoefs, className)
                }
            }
            val final = applyNMS(dets, 0.6f)
            val obstacles = mutableListOf<Obstacle>()
            val protoH = protoShape[1]
            val protoW = protoShape[2]
            val protoC = protoShape[3]
            for ((box, coefs, className) in final) {
                try {
                    val mask = Array(256) { FloatArray(256) }
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
                            if (maskValue > 0.1f) {
                                mask[dy][dx] = 1f
                                activePixels++
                            }
                        }
                    }
                    if (activePixels >= 50) {
                        if ((shouldDetectCars && className == "car") || (shouldDetectChairs && className == "chair")) {
                            continue
                        }
                        val dilatedMask = dilateArray(mask)
                        obstacles.add(Obstacle(box, dilatedMask, className))
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Error building mask for $className: ${e.message}", e)
                }
            }
            return obstacles
        } catch (e: Exception) {
            Log.e(TAG, "Segmentation error: ${e.message}", e)
            return emptyList()
        }
    }

    private fun runSegmentationForChairs(roi: Bitmap): List<Obstacle> {
        try {
            val ti = TensorImage(DataType.FLOAT32).apply { load(roi) }
            val input = imageProcessor.process(ti).buffer
            val detShape = segInterpreter.getOutputTensor(0).shape()
            val protoShape = segInterpreter.getOutputTensor(1).shape()
            val detOut = Array(detShape[0]) { Array(detShape[1]) { FloatArray(detShape[2]) } }
            val protoOut = Array(protoShape[0]) { Array(protoShape[1]) { Array(protoShape[2]) { FloatArray(protoShape[3]) } } }
            val outputs = mapOf(0 to detOut, 1 to protoOut)
            segInterpreter.runForMultipleInputsOutputs(arrayOf(input), outputs)
            val raw = detOut[0]
            val dets = mutableListOf<Triple<RectF, FloatArray, String>>()
            val threshold = 0.75f
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
                    var activePixels = 0
                    for (dy in 0 until 256) {
                        for (dx in 0 until 256) {
                            val py = (dy * protoH / 256).coerceIn(0, protoH -

                                    1)
                            val px = (dx * protoW / 256).coerceIn(0, protoW - 1)
                            var maskValue = 0f
                            for (c in 0 until minOf(coefs.size, protoC)) {
                                maskValue += coefs[c] * protoOut[0][py][px][c]
                            }
                            maskValue = 1.0f / (1.0f + exp(-maskValue))
                            if (maskValue > 0.1f) {
                                mask[dy][dx] = 1f
                                activePixels++
                            }
                        }
                    }
                    if (activePixels >= 50) {
                        val dilatedMask = dilateArray(mask)
                        obstacles.add(Obstacle(box, dilatedMask, className))
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Error building mask for $className: ${e.message}", e)
                }
            }
            return obstacles
        } catch (e: Exception) {
            Log.e(TAG, "Chair segmentation error: ${e.message}", e)
            return emptyList()
        }
    }

    private fun runSegmentationForCars(roi: Bitmap): List<Obstacle> {
        try {
            val ti = TensorImage(DataType.FLOAT32).apply { load(roi) }
            val input = imageProcessor.process(ti).buffer
            val detShape = segInterpreter.getOutputTensor(0).shape()
            val protoShape = segInterpreter.getOutputTensor(1).shape()
            val detOut = Array(detShape[0]) { Array(detShape[1]) { FloatArray(detShape[2]) } }
            val protoOut = Array(protoShape[0]) { Array(protoShape[1]) { Array(protoShape[2]) { FloatArray(protoShape[3]) } } }
            val outputs = mapOf(0 to detOut, 1 to protoOut)
            segInterpreter.runForMultipleInputsOutputs(arrayOf(input), outputs)
            val raw = detOut[0]
            val dets = mutableListOf<Triple<RectF, FloatArray, String>>()
            val threshold = 0.75f
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
                            if (maskValue > 0.1f) {
                                mask[dy][dx] = 1f
                                activePixels++
                            }
                        }
                    }
                    if (activePixels >= 50) {
                        val dilatedMask = dilateArray(mask)
                        obstacles.add(Obstacle(box, dilatedMask, className))
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Error building mask for $className: ${e.message}", e)
                }
            }
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
        val sorted = dets.sortedByDescending { it.second.sumOf { v: Float -> v.toDouble() }.toFloat() }
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
            val ti = TensorImage(DataType.FLOAT32)
            ti.load(bmp)
            val input = depthProcessor.process(ti).buffer
            val outShape = depthInterpreter.getOutputTensor(0).shape()
            val depthMap = when (outShape.size) {
                4 -> {
                    val raw = Array(outShape[0]) { Array(outShape[1]) { Array(outShape[2]) { FloatArray(outShape[3]) } } }
                    depthInterpreter.run(input, raw)
                    Array(outShape[1]) { y -> FloatArray(outShape[2]) { x -> raw[0][y][x][0] } }
                }
                3 -> {
                    val raw = Array(outShape[0]) { Array(outShape[1]) { FloatArray(outShape[2]) } }
                    depthInterpreter.run(input, raw)
                    raw[0]
                }
                else -> {
                    Log.e(TAG, "Unsupported depth output shape: ${outShape.joinToString()}")
                    Array(PROCESSING_SIZE) { FloatArray(PROCESSING_SIZE) { Float.MAX_VALUE } }
                }
            }
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
        return variance < varianceThreshold
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<out String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == 1001) {
            if (grantResults.isNotEmpty() && grantResults.all { it == PackageManager.PERMISSION_GRANTED }) {
                initializeComponents()
            } else {
                val deniedPermissions = permissions.filterIndexed { index, _ ->
                    grantResults[index] != PackageManager.PERMISSION_GRANTED
                }
                val errorMessage = when {
                    deniedPermissions.contains(Manifest.permission.CAMERA) && deniedPermissions.contains(Manifest.permission.RECORD_AUDIO) ->
                        "Camera and audio permissions denied"
                    deniedPermissions.contains(Manifest.permission.CAMERA) ->
                        "Camera permission denied"
                    deniedPermissions.contains(Manifest.permission.RECORD_AUDIO) ->
                        "Audio permission denied"
                    else -> "Required permissions denied"
                }
                runOnUiThread { positionTextView.text = errorMessage }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        handler.removeCallbacksAndMessages(null)
        if (::speechRecognizer.isInitialized) {
            speechRecognizer.stopListening()
            speechRecognizer.destroy()
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