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
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.abs

private const val TAG = "MainActivity"

@SuppressLint("SetTextI18n")
class MainActivity : AppCompatActivity(), TextToSpeech.OnInitListener {

    companion object {
        private const val PROXIMITY_THRESHOLD_M = 0.2f
        private const val PROXIMITY_THRESHOLD_D = 0.4f
        private const val DETECTION_INTERVAL_MS = 333L
        private const val ANIMATION_DURATION = 200L
        private const val DEPTH_SCALE_FACTOR = 100.0f
    }

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

    var shouldDetectDoors = false
    var shouldDetectCars = false
    var shouldDetectChairs = false
    var shouldDetect = shouldDetectDoors || shouldDetectCars || shouldDetectChairs
    private var isDeepSceneDiscoveryActive = false

    private var canProcess = true
    var useYolo12s = false
    private var isFirstLaunch = true

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
    private var isSwitchingCamera = false
    private var lastDetectionMode: String? = null

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

    private lateinit var detectionLogic: DetectionLogic

    @SuppressLint("MissingPermission")
    private fun isInternetAvailable(): Boolean {
        val connectivityManager = getSystemService(CONNECTIVITY_SERVICE) as ConnectivityManager
        val network = connectivityManager.activeNetwork ?: return false
        val capabilities = connectivityManager.getNetworkCapabilities(network) ?: return false
        return capabilities.hasCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET) &&
                capabilities.hasCapability(NetworkCapabilities.NET_CAPABILITY_VALIDATED)
    }

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
        detectionLogic = DetectionLogic(this)

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
            ActivityCompat.requestPermissions(this, permissionsNeeded.toTypedArray(), 1001)
        } else {
            initializeComponents()
        }
    }

    private fun cycleDetection(goForward: Boolean) {
        if (goForward) {
            currentDetectionIndex++
            if (currentDetectionIndex >= detectionModes.size) currentDetectionIndex = 0
        } else {
            currentDetectionIndex--
            if (currentDetectionIndex < 0) currentDetectionIndex = detectionModes.size - 1
        }
        val newDetectionType = detectionModes[currentDetectionIndex]
        startDetection(newDetectionType)
    }

    private fun setupGestureDetector() {
        gestureDetector = GestureDetector(this, object : GestureDetector.SimpleOnGestureListener() {
            override fun onFling(e1: MotionEvent?, e2: MotionEvent, velocityX: Float, velocityY: Float): Boolean {
                if (e1 == null || isSwitchingCamera) return false
                val deltaX = e2.x - e1.x
                val deltaY = e2.y - e1.y
                if (abs(deltaX) > abs(deltaY)) {
                    if (abs(deltaX) > 100 && abs(velocityX) > 100) {
                        if (shouldDetect) {
                            if (deltaX > 0) cycleDetection(true) else cycleDetection(false)
                        }
                        return true
                    }
                } else {
                    if (abs(deltaY) > 200 && abs(velocityY) > 100) {
                        if (deltaY > 0 && (shouldDetect || isDeepSceneDiscoveryActive)) {
                            if (isDeepSceneDiscoveryActive) stopDeepSceneDiscovery() else stopDetection()
                            speak("Stopping detection.")
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
                        animateAndSwitchCamera(targetSelector, true)
                    }
                    lastDetectionMode?.let { mode ->
                        when (mode) {
                            "deep_scene" -> startDeepSceneDiscovery()
                            "door", "chair", "car" -> startDetection(mode)
                        }
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
                        animateAndSwitchCamera(targetSelector, false)
                    }
                }
                return true
            }
        })
    }

    private fun initializeComponents() {
        Log.d(TAG, "Initializing components")
        setupFullscreenUI()
        detectionLogic.setupProcessors()
        runOnUiThread {
            if (!detectionLogic.loadModels()) {
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
        val backCameras = cameraProvider.availableCameraInfos.filter { it.lensFacing == CameraSelector.LENS_FACING_BACK }
        if (backCameras.isEmpty()) {
            Log.e(TAG, "No back cameras found!")
            val defaultSelector = CameraSelector.DEFAULT_BACK_CAMERA
            mainCameraSelector = defaultSelector
            ultraWideCameraSelector = defaultSelector
            return
        }
        val mainCamInfo = backCameras.firstOrNull { it.hasFlashUnit() } ?: backCameras.first()
        mainCameraSelector = CameraSelector.Builder().addCameraFilter { it.filter { camInfo -> camInfo == mainCamInfo } }.build()
        var ultraWideCamInfo: CameraInfo? = mainCamInfo
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
        ultraWideCameraSelector = CameraSelector.Builder().addCameraFilter { it.filter { camInfo -> camInfo == ultraWideCamInfo } }.build()
    }

    private fun animateAndSwitchCamera(selector: CameraSelector, turnFlashOn: Boolean) {
        if (isSwitchingCamera) return
        isSwitchingCamera = true
        cameraSwitchOverlay.visibility = View.VISIBLE
        cameraSwitchOverlay.animate().alpha(1f).setDuration(ANIMATION_DURATION).setListener(object : AnimatorListenerAdapter() {
            override fun onAnimationEnd(animation: Animator) {
                bindCameraUseCases(selector, turnFlashOn)
            }
        }).start()
    }

    private fun bindCameraUseCases(selector: CameraSelector, turnFlashOn: Boolean) {
        val cameraProvider = this.cameraProvider ?: run {
            isSwitchingCamera = false
            return
        }
        try {
            cameraProvider.unbindAll()
            camera = cameraProvider.bindToLifecycle(this, selector, preview, analysis)
            this.activeCameraSelector = selector
            if (turnFlashOn) camera?.cameraControl?.enableTorch(true)
            handler.postDelayed({
                cameraSwitchOverlay.animate().alpha(0f).setDuration(ANIMATION_DURATION).setListener(object : AnimatorListenerAdapter() {
                    override fun onAnimationEnd(animation: Animator) {
                        cameraSwitchOverlay.visibility = View.GONE
                        isSwitchingCamera = false
                    }
                }).start()
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
            preview = Preview.Builder().setTargetRotation(rotation).build().also { it.surfaceProvider = previewView.surfaceProvider }
            analysis = ImageAnalysis.Builder()
                .setTargetRotation(rotation)
                .setResolutionSelector(ResolutionSelector.Builder().setResolutionStrategy(ResolutionStrategy(Size(640, 640), ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER)).build())
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
                .also { it.setAnalyzer(cameraExecutor, ::onFrame) }
            findCameraSelectors()
            val initialSelector = ultraWideCameraSelector ?: mainCameraSelector ?: CameraSelector.DEFAULT_BACK_CAMERA
            bindCameraUseCases(initialSelector, false)
        }, ContextCompat.getMainExecutor(this))
    }

    private fun startVoiceRecognition() {
        if (isRecognizerListening || isSpeaking || ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            Log.d(TAG, "startVoiceRecognition skipped")
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
            return
        }
        hasSpokenOfflineWarning = false
        runOnUiThread {
            detectButton.visibility = View.GONE
            detectButton.isEnabled = true
            if (!hasSpokenDoorWarning) positionTextView.text = "Listening..."
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
            putExtra(RecognizerIntent.EXTRA_SPEECH_INPUT_MINIMUM_LENGTH_MILLIS, 15000L)
            putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
            putExtra(RecognizerIntent.EXTRA_PREFER_OFFLINE, true)
        }
        try {
            Log.d(TAG, "Starting voice recognition")
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
            speechRecognizer.stopListening()
            speechRecognizer.destroy()
        }
        speechRecognizer = SpeechRecognizer.createSpeechRecognizer(this)
        speechRecognizer.setRecognitionListener(object : RecognitionListener {
            override fun onReadyForSpeech(params: Bundle?) { isRecognizerListening = true }
            override fun onBeginningOfSpeech() {}
            override fun onRmsChanged(rmsdB: Float) {}
            override fun onBufferReceived(buffer: ByteArray?) {}
            override fun onEndOfSpeech() {
                isRecognizerListening = false
                if (!isSpeaking && (isVoiceActive || !shouldDetect)) startVoiceRecognition()
            }
            override fun onError(error: Int) {
                isRecognizerListening = false
                when (error) {
                    SpeechRecognizer.ERROR_NO_MATCH, SpeechRecognizer.ERROR_SPEECH_TIMEOUT ->
                        if (!isSpeaking && (isVoiceActive || !shouldDetect)) handler.postDelayed({ startVoiceRecognition() }, 1000L)
                    else -> if (!isSpeaking && (isVoiceActive || !shouldDetect)) handler.postDelayed({ startVoiceRecognition() }, 100L)
                }
            }
            override fun onResults(results: Bundle?) {
                val matches = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                if (!matches.isNullOrEmpty() && !isSpeaking) processVoiceCommand(matches[0])
                if (!isSpeaking && (isVoiceActive || !shouldDetect)) startVoiceRecognition()
            }
            override fun onPartialResults(partialResults: Bundle?) {}
            override fun onEvent(eventType: Int, params: Bundle?) {}
        })
    }

    override fun onResume() {
        super.onResume()
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) initSpeechRecognizer()
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
        if (::speechRecognizer.isInitialized) speechRecognizer.destroy()
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
                    if (isDeepSceneDiscoveryActive) deepSceneDiscovery.onSpeechFinished()
                    if ((isVoiceActive || !shouldDetect) && !isRecognizerListening) handler.post { startVoiceRecognition() }
                }
                @Deprecated("Deprecated in Java")
                override fun onError(utteranceId: String?) {
                    isSpeaking = false
                    if (isDeepSceneDiscoveryActive) deepSceneDiscovery.onSpeechFinished()
                    if ((isVoiceActive || !shouldDetect) && !isRecognizerListening) handler.post { startVoiceRecognition() }
                }
            })
            if (isVoiceActive && !hasGreeted) {
                speak("Hello, how can I help you?")
                hasGreeted = true
                runOnUiThread { positionTextView.text = "Voice activated, say 'doors', 'cars', 'chairs', 'deep scene discovery', or 'stop'" }
            }
            if (initialOfflineWarningSent && !isInternetAvailable()) speak("No internet connection. Some features may not work.")
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
                "starting deep scene discovery", "stopping deep scene discovery")) return
        if (isSpeaking) return
        when {
            command.lowercase(Locale.getDefault()).contains("hello") || command.lowercase(Locale.getDefault()).contains("vai") ||
                    command.lowercase(Locale.getDefault()).contains("hey") || command.lowercase(Locale.getDefault()).contains("vi") ||
                    command.lowercase(Locale.getDefault()).contains("hi") || command.lowercase(Locale.getDefault()).contains("voi") -> {
                isVoiceActive = true
                if (!hasGreeted) {
                    speak("Hello, how can I help you?")
                    hasGreeted = true
                }
                runOnUiThread { positionTextView.text = "Voice activated, say 'doors', 'cars', 'chairs', 'deep scene discovery', or 'stop'" }
            }
            command.lowercase(Locale.getDefault()).contains("door") && !isDeepSceneDiscoveryActive -> if (!shouldDetectDoors) startDetection("door")
            command.lowercase(Locale.getDefault()).contains("car") && !isDeepSceneDiscoveryActive -> if (!shouldDetectCars) startDetection("car")
            command.lowercase(Locale.getDefault()).contains("chair") && !isDeepSceneDiscoveryActive -> if (!shouldDetectChairs) startDetection("chair")
            command.lowercase(Locale.getDefault()).contains("deep") || command.lowercase(Locale.getDefault()).contains("scene") ||
                    command.lowercase(Locale.getDefault()).contains("discover") -> if (!isDeepSceneDiscoveryActive) startDeepSceneDiscovery()
            command.lowercase(Locale.getDefault()).contains("stop") && isDeepSceneDiscoveryActive -> stopDeepSceneDiscovery()
            command.lowercase(Locale.getDefault()).contains("stop") && !isDeepSceneDiscoveryActive -> stopDetection()
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
            window.setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN, WindowManager.LayoutParams.FLAG_FULLSCREEN)
        }
        supportActionBar?.hide()
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
            if (isDeepSceneDiscoveryActive) stopDeepSceneDiscovery() else stopDetection()
        } else {
            startDetection()
        }
    }

    private fun toggleDoorModel() {
        useYolo12s = !useYolo12s
        val modelName = if (useYolo12s) "yolo12s" else "yolo8n"
        if (detectionLogic.loadModels()) {
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
                    positionTextView.isSingleLine = false
                    positionTextView.maxLines = 3
                    positionTextView.text = "Low lighting. Double tap to turn the flash on"
                }
                if (shouldDetect || isDeepSceneDiscoveryActive) {
                    lastDetectionMode = when {
                        isDeepSceneDiscoveryActive -> "deep_scene"
                        shouldDetectDoors -> "door"
                        shouldDetectChairs -> "chair"
                        shouldDetectCars -> "car"
                        else -> null
                    }
                    stopDetection()
                    stopDeepSceneDiscovery()
                    speak("Low lighting. Double tap to turn the flash on")
                }
                canProcess = true
                return
            } else {
                runOnUiThread { lowLightWarningTextView.visibility = View.GONE }
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
                    val (doorBox, pos) = detectionLogic.detectDoor(bmp)
                    val fullDepthMap = detectionLogic.runDepthEstimation(bmp)
                    val doorDepth = if (doorBox != null) {
                        val rawDoorDepth = detectionLogic.avgDepthInBoxFixed(fullDepthMap, doorBox, bmp.width, bmp.height)
                        if (rawDoorDepth.isFinite()) DEPTH_SCALE_FACTOR / rawDoorDepth else Float.MAX_VALUE
                    } else Float.MAX_VALUE
                    Triple(doorBox, pos, doorDepth)
                }
                shouldDetectChairs -> {
                    val (chairBox, pos) = detectionLogic.detectChair(bmp)
                    val fullDepthMap = detectionLogic.runDepthEstimation(bmp)
                    val chairDepth = if (chairBox != null) {
                        val rawChairDepth = detectionLogic.avgDepthInBoxFixed(fullDepthMap, chairBox, bmp.width, bmp.height)
                        if (rawChairDepth.isFinite()) DEPTH_SCALE_FACTOR / rawChairDepth else Float.MAX_VALUE
                    } else Float.MAX_VALUE
                    Triple(chairBox, pos, chairDepth)
                }
                shouldDetectCars -> {
                    val (carBox, pos) = detectionLogic.detectCar(bmp)
                    val fullDepthMap = detectionLogic.runDepthEstimation(bmp)
                    val carDepth = if (carBox != null) {
                        val rawCarDepth = detectionLogic.avgDepthInBoxFixed(fullDepthMap, carBox, bmp.width, bmp.height)
                        if (rawCarDepth.isFinite()) DEPTH_SCALE_FACTOR / rawCarDepth else Float.MAX_VALUE
                    } else Float.MAX_VALUE
                    Triple(carBox, pos, carDepth)
                }
                else -> Triple(null, "", Float.MAX_VALUE)
            }
            val fullDepthMap = detectionLogic.runDepthEstimation(bmp)
            val obstacles = detectionLogic.runSegmentation(bmp)
            val mappedObstacles = obstacles.map { obstacle ->
                val mappedMask = mapMaskToOriginal(obstacle.mask, bmp.width, bmp.height)
                DetectionLogic.Obstacle(obstacle.box, mappedMask, obstacle.className)
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
                generateNavigationInstruction(targetBox, position, depthMeters, mappedObstacles, fullDepthMap, bmp, targetClass)
            } else {
                val closeObstacles = mappedObstacles.filter { obstacle ->
                    val obstacleDepth = detectionLogic.avgMaskDepthFixed(fullDepthMap, obstacle.mask)
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
            if (message.isNotEmpty()) handleMessage(message)
        } catch (e: Exception) {
            Log.e(TAG, "Frame processing error: ${e.message}", e)
            runOnUiThread {
                positionTextView.isSingleLine = false
                positionTextView.maxLines = 3
                positionTextView.text = "Error: ${e.message}"
            }
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
            runOnUiThread {
                positionTextView.visibility = View.VISIBLE
                positionTextView.isSingleLine = false
                positionTextView.maxLines = 3
                positionTextView.text = message
            }
        }
    }

    private fun mapMaskToOriginal(mask: Array<FloatArray>, origWidth: Int, origHeight: Int): Array<FloatArray> {
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
        obstacles: List<DetectionLogic.Obstacle>,
        depthMap: Array<FloatArray>,
        bitmap: Bitmap,
        targetClass: String
    ): String {
        val proximityThreshold = PROXIMITY_THRESHOLD_D
        if (depthMeters < proximityThreshold) return "You have reached the $targetClass."
        val blockingObstacles = obstacles.filter { obstacle ->
            val obstacleDepth = detectionLogic.avgMaskDepthFixed(depthMap, obstacle.mask)
            val obstacleDepthMeters = if (obstacleDepth.isFinite()) DEPTH_SCALE_FACTOR / obstacleDepth else Float.MAX_VALUE
            obstacleDepthMeters < depthMeters && obstacleDepthMeters < PROXIMITY_THRESHOLD_M && detectionLogic.isObstacleInPath(obstacle.box, targetBox)
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
                val rightObstacles = detectionLogic.runSegmentation(rightHalf).filter { it.className != targetClass }
                if (rightObstacles.isEmpty()) {
                    "The $targetClass is to your left, but there is $obstacleNames in the way. Move right to avoid it, then turn left."
                } else {
                    val rightObstacleNames = rightObstacles.joinToString(" and ") { it.className }
                    "The $targetClass is to your left, but there is $obstacleNames in the way. The right path is blocked by $rightObstacleNames."
                }
            }
            "right" -> {
                val leftHalf = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width / 2, bitmap.height)
                val leftObstacles = detectionLogic.runSegmentation(leftHalf).filter { it.className != targetClass }
                if (leftObstacles.isEmpty()) {
                    "The $targetClass is to your right, but there is $obstacleNames in the way. Move left to avoid it, then turn right."
                } else {
                    val leftObstacleNames = leftObstacles.joinToString(" and ") { it.className }
                    "The $targetClass is to your right, but there is $obstacleNames in the way. The left path is blocked by $leftObstacleNames."
                }
            }
            else -> {
                val rightThird = Bitmap.createBitmap(bitmap, bitmap.width * 2 / 3, 0, bitmap.width / 3, bitmap.height)
                val rightObstacles = detectionLogic.runSegmentation(rightThird).filter { it.className != targetClass }
                if (rightObstacles.isEmpty()) {
                    "The $targetClass is straight ahead, but there is $obstacleNames in the way. Move right to avoid it, then continue forward."
                } else {
                    val leftThird = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width / 3, bitmap.height)
                    val leftObstacles = detectionLogic.runSegmentation(leftThird).filter { it.className != targetClass }
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

    private fun startDetection(type: String? = null) {
        if (type == null) {
            swipeInstructionTextView.visibility = View.VISIBLE
            isWaitingForIconSelection = true
            shouldDetect = false
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
            val intensity = ((pixel shr 16 and 0xFF) * 0.299f) + ((pixel shr 8 and 0xFF) * 0.587f) + ((pixel and 0xFF) * 0.114f)
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
        val varianceThreshold = 2500f
        return variance < varianceThreshold
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == 1001) {
            if (grantResults.isNotEmpty() && grantResults.all { it == PackageManager.PERMISSION_GRANTED }) {
                initializeComponents()
            } else {
                val deniedPermissions = permissions.filterIndexed { index, _ -> grantResults[index] != PackageManager.PERMISSION_GRANTED }
                val errorMessage = when {
                    deniedPermissions.contains(Manifest.permission.CAMERA) && deniedPermissions.contains(Manifest.permission.RECORD_AUDIO) -> "Camera and audio permissions denied"
                    deniedPermissions.contains(Manifest.permission.CAMERA) -> "Camera permission denied"
                    deniedPermissions.contains(Manifest.permission.RECORD_AUDIO) -> "Audio permission denied"
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
        if (::tts.isInitialized) tts.shutdown()
        cameraExecutor.shutdown()
        cameraProvider?.unbindAll()
    }
}