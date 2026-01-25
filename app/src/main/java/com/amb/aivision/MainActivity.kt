package com.amb.aivision

import android.Manifest
import android.animation.Animator
import android.animation.AnimatorListenerAdapter
import android.annotation.SuppressLint
import android.content.Intent
import android.view.animation.Animation
import android.view.animation.AnimationUtils
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
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.provider.Settings
import android.speech.RecognizerIntent
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
import androidx.work.Data
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.WorkInfo
import androidx.work.WorkManager
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import org.opencv.android.OpenCVLoader
import java.io.File
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.abs

private const val TAG = "MainActivity"

@SuppressLint("SetTextI18n")
class MainActivity : AppCompatActivity() {
    companion object {
        private const val PROXIMITY_THRESHOLD_M = 0.2f
        private const val PROXIMITY_THRESHOLD_D = 0.4f
        private const val DETECTION_INTERVAL_MS = 333L
        private const val ANIMATION_DURATION = 200L
        private const val DEPTH_SCALE_FACTOR = 100.0f
        private const val MODEL_FILE_NAME = "gemma-3n-E4B-it-int4.task"
        private const val MODEL_URL_HF = "https://huggingface.co/google/gemma-3n-E4B-it-litert-preview/resolve/main/gemma-3n-E4B-it-int4.task?download=true"
        private const val MODEL_URL_KAGGLE = "https://www.kaggle.com/models/google/gemma-3n/tfLite/download?file=gemma-3n-E4B-it-int4.task.zip"
        private const val HF = BuildConfig.HF_TOKEN
        private const val REQUEST_PERMISSIONS_CODE = 1001
//        private const val MODEL_TOTAL_BYTES = 4_733_321_216L // 4.41 GB for Hugging Face
        private const val MODEL_TOTAL_BYTES = 4_402_141_478L // 4.10 GB for Hugging Face
        private const val MODEL_TOTAL_BYTES_KAGGLE = 2_684_354_560L // ~2.5 GB for Kaggle ZIP
    }

    private val detectionModes = listOf("door", "chair", "car")
    private var currentDetectionIndex = 0

    private val topLeftGlow: ImageView by lazy { findViewById(R.id.topLeftGlow) }
    private val topRightGlow: ImageView by lazy { findViewById(R.id.topRightGlow) }
    private val bottomLeftGlow: ImageView by lazy { findViewById(R.id.bottomLeftGlow) }
    private val bottomRightGlow: ImageView by lazy { findViewById(R.id.bottomRightGlow) }
    private lateinit var glowViews: List<ImageView>
    private val previewView: PreviewView by lazy { findViewById(R.id.previewView) }
    val positionTextView: TextView by lazy { findViewById(R.id.positionTextView) }
    val detectButton: Button by lazy { findViewById(R.id.detectButton) }
    private val cameraSwitchOverlay: View by lazy { findViewById(R.id.cameraSwitchOverlay) }
    internal val swipeInstructionTextView: TextView by lazy { findViewById(R.id.swipeInstructionTextView) }
    private val chairButton: ImageButton by lazy { findViewById(R.id.chairButton) }
    private val carButton: ImageButton by lazy { findViewById(R.id.carButton) }
    private val doorButton: ImageButton by lazy { findViewById(R.id.doorButton) }
    private val lowLightWarningTextView: TextView by lazy { findViewById(R.id.lowLightWarningTextView) }
    private val leftArrowImageView: ImageView by lazy { findViewById(R.id.leftArrowImageView) }
    private val rightArrowImageView: ImageView by lazy { findViewById(R.id.rightArrowImageView) }
    val downloadModelButton: Button by lazy { findViewById(R.id.downloadModelButton) }
    lateinit var voiceManager: VoiceManager
    private lateinit var gestureDetector: GestureDetector
    private lateinit var model: Model
    private var lastDetectionTime = 0L
    private var isVoiceActive = false
    private var previousMessage: String? = null
    private var consecutiveIdenticalCount = 0
    var shouldDetectDoors = false
    var shouldDetectCars = false
    var shouldDetectChairs = false
    var shouldDetect = false
    private var isDeepSceneDiscoveryActive = false
    private var canProcess = true
    private var isFirstLaunch = true
    private var initialOfflineWarningSent = false
    private var hasGreeted = false
    private var hasSpokenOfflineWarning = false
    private var hasSpokenDoorWarning = false
    private var wasDetectingBeforePause = false
    private lateinit var cameraExecutor: ExecutorService
    private var cameraProvider: ProcessCameraProvider? = null
    private var camera: androidx.camera.core.Camera? = null
    private var mainCameraSelector: CameraSelector? = null
    private var ultraWideCameraSelector: CameraSelector? = null
    private var activeCameraSelector: CameraSelector? = null
    private var isFlashOn = false
    private var isSwitchingCamera = false
    private var lastDetectionMode: String? = null
    private val modelDIR = "models"
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
    private var deviceOrientationDegrees = 0
    private lateinit var orientationEventListener: OrientationEventListener

    @SuppressLint("MissingPermission")
    private fun isInternetAvailable(): Boolean {
        val connectivityManager = getSystemService(CONNECTIVITY_SERVICE) as ConnectivityManager
        val network = connectivityManager.activeNetwork ?: return false
        val capabilities = connectivityManager.getNetworkCapabilities(network) ?: return false
        return capabilities.hasCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET) &&
                capabilities.hasCapability(NetworkCapabilities.NET_CAPABILITY_VALIDATED)
    }

    private var isWaitingForIconSelection = false

    @SuppressLint("ClickableViewAccessibility", "NewApi")
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
        glowViews = listOf(topLeftGlow, topRightGlow, bottomLeftGlow, bottomRightGlow)

        window.decorView.setOnApplyWindowInsetsListener { _, insets ->
            val topInset =
                insets.getInsets(WindowInsets.Type.statusBars() or WindowInsets.Type.displayCutout()).top
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
        deepSceneDiscovery.setMainActivity(this)
        model = Model(
            name = "gemma-3n",
            url = MODEL_URL_HF,
            totalBytes = MODEL_TOTAL_BYTES,
            downloadFileName = MODEL_FILE_NAME,
            normalizedName = modelDIR,
            accessToken = HF
        )

        // Initialize Gemma model immediately on app start
        CoroutineScope(Dispatchers.Main).launch {
            try {
                deepSceneDiscovery.initialize()
                runOnUiThread {
                    if (deepSceneDiscovery.initializationComplete) {
                        positionTextView.text = "Models ready. Waiting for voice commands..."
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "LLM initialization failed: ${e.message}", e)
                // LLM not available - YOLO detection still works
            }
        }

        checkModelAvailability()

        downloadModelButton.setOnClickListener {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
                positionTextView.text = "Camera permission required to proceed."
                Toast.makeText(this, "Please grant camera permission.", Toast.LENGTH_LONG).show()
                requestPermissions()
            } else if (ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED) {
                positionTextView.text = "Notification permission required for download updates."
                Toast.makeText(this, "Please grant notification permission.", Toast.LENGTH_LONG).show()
                ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.POST_NOTIFICATIONS), REQUEST_PERMISSIONS_CODE)
            } else {
                downloadModelButton.isEnabled = false
                positionTextView.text = "Starting model download..."
                startModelDownload()
            }
        }
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
            startDeepSceneDiscovery()
            true
        }

        if (intent?.action == Intent.ACTION_ASSIST) {
            isVoiceActive = true
        }

        requestPermissions()
        
        orientationEventListener = object : OrientationEventListener(this) {
            override fun onOrientationChanged(orientation: Int) {
                if (orientation == OrientationEventListener.ORIENTATION_UNKNOWN) return
                deviceOrientationDegrees = orientation
            }
        }
        if (orientationEventListener.canDetectOrientation()) {
            orientationEventListener.enable()
        }
    }

    private fun checkModelAvailability(): Boolean {
        val modelFile = File(getExternalFilesDir(null), "$modelDIR/$MODEL_FILE_NAME")
        val expectedSize = MODEL_TOTAL_BYTES
        if (modelFile.exists() && modelFile.length() >= expectedSize) {
            runOnUiThread {
                downloadModelButton.visibility = View.GONE
                // detectButton is enabled by YOLO loading in initializeComponents, not LLM
                positionTextView.text = "LLM model ready."
            }
            return true
        } else if (modelFile.exists() && modelFile.length() < expectedSize) {
            runOnUiThread {
                downloadModelButton.visibility = View.VISIBLE
                positionTextView.text = "Partial LLM model detected. Resume download for scene description."
            }
            checkExistingDownloads()
            return false
        } else {
            runOnUiThread {
                downloadModelButton.visibility = View.VISIBLE
                positionTextView.text = "Download LLM model for scene description, or use object detection."
            }
            checkExistingDownloads()
            return false
        }
    }

    @SuppressLint("NewApi")
    private fun startModelDownload() {
        val workManager = WorkManager.getInstance(this)
        val isKaggle = model.url == MODEL_URL_KAGGLE
        val data = Data.Builder()
            .putString(DownloadWorker.KEY_MODEL_URL, model.url)
            .putString(DownloadWorker.KEY_MODEL_NAME, model.name)
            .putString(DownloadWorker.KEY_MODEL_VERSION, "1.0")
            .putString(DownloadWorker.KEY_MODEL_DOWNLOAD_FILE_NAME, if (isKaggle) "gemma-3n-E4B-it-int4.task.zip" else MODEL_FILE_NAME)
            .putString(DownloadWorker.KEY_MODEL_DOWNLOAD_MODEL_DIR, modelDIR)
            .putBoolean(DownloadWorker.KEY_MODEL_IS_ZIP, isKaggle)
            .putString(DownloadWorker.KEY_MODEL_UNZIPPED_DIR, if (isKaggle) modelDIR else null)
            .putLong(DownloadWorker.KEY_MODEL_TOTAL_BYTES, if (isKaggle) MODEL_TOTAL_BYTES_KAGGLE else MODEL_TOTAL_BYTES)
            .putString(DownloadWorker.KEY_MODEL_DOWNLOAD_ACCESS_TOKEN, model.accessToken)
            .build()

        val downloadRequest = OneTimeWorkRequestBuilder<DownloadWorker>()
            .setInputData(data)
            .addTag(model.name)
            .build()

        workManager.enqueue(downloadRequest)
        observeDownloadProgress(downloadRequest.id)
    }

    private fun checkExistingDownloads() {
        val workManager = WorkManager.getInstance(this)
        workManager.getWorkInfosByTagLiveData(model.name).observe(this) { workInfos ->
            val workInfo = workInfos.find { it.tags.contains(model.name) }
            if (workInfo != null && (workInfo.state == WorkInfo.State.ENQUEUED || workInfo.state == WorkInfo.State.RUNNING)) {
                downloadModelButton.isEnabled = false
                positionTextView.text = "Download already in progress..."
                observeDownloadProgress(workInfo.id)
            }
        }
    }

    @SuppressLint("NewApi")
    private fun observeDownloadProgress(workId: UUID) {
        WorkManager.getInstance(this).getWorkInfoByIdLiveData(workId).observe(this) { workInfo ->
            if (workInfo != null) {
                when (workInfo.state) {
                    WorkInfo.State.RUNNING -> {
                        val receivedBytes = workInfo.progress.getLong(DownloadWorker.KEY_MODEL_DOWNLOAD_RECEIVED_BYTES, 0L)
                        val isUnzipping = workInfo.progress.getBoolean(DownloadWorker.KEY_MODEL_START_UNZIPPING, false)
                        val progress = minOf((receivedBytes * 100.0 / MODEL_TOTAL_BYTES.toDouble()).toInt(), 100)
                        runOnUiThread {
                            positionTextView.text = if (isUnzipping) "Unzipping model..." else "Downloading model: $progress%"
                        }
                    }
                    WorkInfo.State.SUCCEEDED -> {
                        runOnUiThread {
                            positionTextView.text = "Model downloaded successfully."
                            downloadModelButton.visibility = View.GONE
                            downloadModelButton.isEnabled = true
                            downloadModelButton.text = "Download Model"
                            detectButton.isEnabled = true
                        }
                        CoroutineScope(Dispatchers.Main).launch {
                            deepSceneDiscovery.initialize()
                        }
                    }
                    WorkInfo.State.FAILED, WorkInfo.State.CANCELLED -> {
                        val errorMessage = workInfo.outputData.getString(DownloadWorker.KEY_MODEL_DOWNLOAD_ERROR_MESSAGE) ?: "Unknown error"
                        runOnUiThread {
                            positionTextView.text = "Failed to download model: $errorMessage"
                            downloadModelButton.isEnabled = true
                            downloadModelButton.text = "Download Model"
                        }
                    }
                    else -> {}
                }
            }
        }
    }

    private fun requestPermissions() {
        val permissionsToRequest = mutableListOf(
            Manifest.permission.CAMERA,
            Manifest.permission.RECORD_AUDIO,
            Manifest.permission.POST_NOTIFICATIONS
        ).filter {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }
        if (permissionsToRequest.isNotEmpty()) {
            ActivityCompat.requestPermissions(this, permissionsToRequest.toTypedArray(), REQUEST_PERMISSIONS_CODE)
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
                            voiceManager.speak("Stopping detection.")
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
                        voiceManager.speak("Flash not available.")
                        isFlashOn = false
                        return true
                    }
                    voiceManager.speak("Flash on")
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
                        voiceManager.speak("Flash off")
                        return true
                    }
                    voiceManager.speak("Flash off")
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
        
        voiceManager = VoiceManager(
            this,
            onCommandRecognized = { command -> processVoiceCommand(command) },
            onTtsStart = { stopDetectionIfListening() }, // Helper to handle listening state 
            onTtsDone = { 
                if (isDeepSceneDiscoveryActive) deepSceneDiscovery.onSpeechFinished()
            },
            onTtsError = { error ->
                Log.e(TAG, "TTS Error: $error")
                if (isDeepSceneDiscoveryActive) deepSceneDiscovery.onSpeechFinished()
            }
        )
        
        updateUiState()
        
        if (!isInternetAvailable()) {
            Log.w(TAG, "No internet connection available")
            voiceManager.speak("No internet connection. Some features may not work.")
            initialOfflineWarningSent = true
        }
        startCamera()
        handler.postDelayed({
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
               voiceManager.startListening()
            }
        }, 100L)
    }

    private fun stopDetectionIfListening() {
        if (::voiceManager.isInitialized) voiceManager.stopListening()
    }

    private fun startGlowEffect() {
        val fadeIn = AnimationUtils.loadAnimation(this, R.anim.fade_in)
        glowViews.forEach {
            it.visibility = View.VISIBLE
            it.startAnimation(fadeIn)
        }
    }

    private fun stopGlowEffect() {
        val fadeOut = AnimationUtils.loadAnimation(this, R.anim.fade_out)
        fadeOut.setAnimationListener(object : Animation.AnimationListener {
            override fun onAnimationStart(animation: Animation?) {}
            override fun onAnimationEnd(animation: Animation?) {
                glowViews.forEach { it.visibility = View.GONE }
            }
            override fun onAnimationRepeat(animation: Animation?) {}
        })
        glowViews.forEach {
            it.startAnimation(fadeOut)
        }
    }

    private fun isLowLight(bitmap: Bitmap, threshold: Int = 60): Boolean {
        val avgLuminance = computeAmbientLightLevel(bitmap)
        return avgLuminance < threshold
    }

    /**
     * Compute ambient light level from bitmap (0-255 scale)
     */
    private fun computeAmbientLightLevel(bitmap: Bitmap): Float {
        val pixels = IntArray(bitmap.width * bitmap.height)
        bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        var totalLuminance = 0.0
        for (pixel in pixels) {
            val r = Color.red(pixel)
            val g = Color.green(pixel)
            val b = Color.blue(pixel)
            totalLuminance += (0.299 * r + 0.587 * g + 0.114 * b)
        }
        return (totalLuminance / pixels.size).toFloat()
    }

    private fun triggerHapticFeedback() {
        val vibrator = getSystemService(Vibrator::class.java)
        if (vibrator.hasVibrator()) {
            val vibrationEffect = VibrationEffect.createOneShot(1000, VibrationEffect.DEFAULT_AMPLITUDE)
            vibrator.vibrate(vibrationEffect)
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
            val rotation =
                display?.rotation ?: Surface.ROTATION_0
            preview = Preview.Builder().setTargetRotation(rotation).build().also { it.surfaceProvider = previewView.surfaceProvider }
            analysis = ImageAnalysis.Builder()
                .setTargetRotation(rotation)
                .setResolutionSelector(ResolutionSelector.Builder().setResolutionStrategy(ResolutionStrategy(
                    Size(640, 640), ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER)).build())
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
                .also { it.setAnalyzer(cameraExecutor, ::onFrame) }
            findCameraSelectors()
            val initialSelector = ultraWideCameraSelector ?: mainCameraSelector ?: CameraSelector.DEFAULT_BACK_CAMERA
            bindCameraUseCases(initialSelector, false)
        }, ContextCompat.getMainExecutor(this))
    }

    // startVoiceRecognition logic moved to VoiceManager and MainActivity initialization
    private fun startVoiceLogic() {
         if (::voiceManager.isInitialized) {
             voiceManager.shouldListen = true
             voiceManager.startListening()
         }
    }

    private fun updateUiState() {
        val isOnline = isInternetAvailable()
        runOnUiThread {
            if (isOnline) {
                detectButton.visibility = View.GONE
                // Offline buttons should be hidden initially or managed by startDetection logic
                chairButton.visibility = View.GONE
                carButton.visibility = View.GONE
                doorButton.visibility = View.GONE
                if (!hasGreeted && isFirstLaunch) {
                    positionTextView.text = "Listening..."
                }
            } else {
                detectButton.visibility = View.VISIBLE
                detectButton.isEnabled = true
                detectButton.text = "Start Detection"
                positionTextView.text = "Offline Mode. Use Manual Detection."
            }
        }
    }

    // initSpeechRecognizer removed - replaced by VoiceManager

    @SuppressLint("NewApi")
    override fun onResume() {
        super.onResume()
        // VoiceManager is initialized in initializeComponents
        updateUiState()

        if (isFirstLaunch) {
            isFirstLaunch = false
            shouldDetect = false
            shouldDetectDoors = false
            shouldDetectCars = false
            shouldDetectChairs = false
            isDeepSceneDiscoveryActive = false
            detectButton.text = "Start Detection"
            val modelFile = File(getExternalFilesDir(null), "$modelDIR/$MODEL_FILE_NAME")
            if (!modelFile.exists()) {
                downloadModelButton.visibility = View.VISIBLE
                // detectButton managed by YOLO loading, not LLM
                positionTextView.text = "Download LLM for scene description, or use object detection."
            } else {
                positionTextView.text = "Waiting for voice commands..."
            }
            handler.postDelayed({ startVoiceLogic() }, 100L)
        } else {
            if (wasDetectingBeforePause) {
                stopDetection()
                stopDeepSceneDiscovery()
                wasDetectingBeforePause = false
            } else {
                handler.postDelayed({ startVoiceLogic() }, 100L)
            }
        }
    }

    override fun onStop() {
        super.onStop()
        // VoiceManager shutdown moved to onDestroy
    }

    override fun onPause() {
        super.onPause()
        wasDetectingBeforePause = shouldDetect || isDeepSceneDiscoveryActive
        shouldDetect = false
        handler.removeCallbacks(detectionRunnable)
        if (::voiceManager.isInitialized) {
            voiceManager.stopListening()
            voiceManager.stopTts()
        }
    }

    private fun processVoiceCommand(command: String) {
        if (command.isBlank() || command.lowercase(Locale.getDefault()) in listOf(
                "hello, how can i help you", "starting detection, press on the icon that you want to detect",
                "starting detecting doors", "starting detecting cars", "starting detecting chairs", "stopping detection",
                "no internet connection. voice recognition is unavailable. use the button to detect",
                "no internet connection. some features like voice recognition may not work",
                "starting deep scene discovery", "stopping deep scene discovery")) return
        if (voiceManager.isSpeaking) return
        when {
            command.lowercase(Locale.getDefault()).contains("hello") || command.lowercase(Locale.getDefault()).contains("vai") ||
                    command.lowercase(Locale.getDefault()).contains("hey") || command.lowercase(Locale.getDefault()).contains("vi") ||
                    command.lowercase(Locale.getDefault()).contains("hi") || command.lowercase(Locale.getDefault()).contains("voi") -> {
                isVoiceActive = true
                if (!hasGreeted) {
                    voiceManager.speak("Hello, how can I help you?")
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
        window.insetsController?.let {
            it.hide(WindowInsets.Type.statusBars() or WindowInsets.Type.displayCutout())
            it.systemBarsBehavior = WindowInsetsController.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
        }
        supportActionBar?.hide()
    }

    override fun onConfigurationChanged(newConfig: Configuration) {
        super.onConfigurationChanged(newConfig)
        val rotation =
            display?.rotation ?: Surface.ROTATION_0
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

    private fun onFrame(image: ImageProxy) {
        val currentTime = System.currentTimeMillis()
        if ((!shouldDetect && !isDeepSceneDiscoveryActive) || !canProcess || currentTime - lastDetectionTime < DETECTION_INTERVAL_MS) {
            image.close()
            return
        }
        canProcess = false
        lastDetectionTime = currentTime
        try {
            // Determine if we need extra rotation. 
            // If display is Portrait (0) but device is Landscape (90/270), rotate to match world horizon.
            val uiRotation = display?.rotation ?: Surface.ROTATION_0
            var extraRotation = 0f
            if (uiRotation == Surface.ROTATION_0) {
                 if (deviceOrientationDegrees in 45..135) { // Landscape Right (90)
                     extraRotation = 90f  // Fixed: was 270f, swapped direction
                 } else if (deviceOrientationDegrees in 225..315) { // Landscape Left (270)
                     extraRotation = 270f  // Fixed: was 90f, swapped direction
                 }
            }
            
            val bmp = imageProxyToBitmap(image, extraRotation)
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
                    voiceManager.speak("Low lighting. Double tap to turn the flash on")
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
            // Compute depth map once for all detection types
            val depthStart = System.currentTimeMillis()
            val fullDepthMap = detectionLogic.runDepthEstimation(bmp)
            val depthTime = System.currentTimeMillis() - depthStart
            
            val detectionStart = System.currentTimeMillis()
            val (targetBox, position, depthMeters) = when {
                shouldDetectDoors -> {
                    val (doorBox, pos) = detectionLogic.detectDoor(bmp)
                    val doorDepth = if (doorBox != null) {
                        val rawDoorDepth = detectionLogic.avgDepthInBoxFixed(fullDepthMap, doorBox, bmp.width, bmp.height)
                        if (rawDoorDepth.isFinite()) DEPTH_SCALE_FACTOR / rawDoorDepth else Float.MAX_VALUE
                    } else Float.MAX_VALUE
                    Triple(doorBox, pos, doorDepth)
                }
                shouldDetectChairs -> {
                    val (chairBox, pos) = detectionLogic.detectChair(bmp)
                    val chairDepth = if (chairBox != null) {
                        val rawChairDepth = detectionLogic.avgDepthInBoxFixed(fullDepthMap, chairBox, bmp.width, bmp.height)
                        if (rawChairDepth.isFinite()) DEPTH_SCALE_FACTOR / rawChairDepth else Float.MAX_VALUE
                    } else Float.MAX_VALUE
                    Triple(chairBox, pos, chairDepth)
                }
                shouldDetectCars -> {
                    val (carBox, pos) = detectionLogic.detectCar(bmp)
                    val carDepth = if (carBox != null) {
                        val rawCarDepth = detectionLogic.avgDepthInBoxFixed(fullDepthMap, carBox, bmp.width, bmp.height)
                        if (rawCarDepth.isFinite()) DEPTH_SCALE_FACTOR / rawCarDepth else Float.MAX_VALUE
                    } else Float.MAX_VALUE
                    Triple(carBox, pos, carDepth)
                }
                else -> Triple(null, "", Float.MAX_VALUE)
            }
            val detectionTime = System.currentTimeMillis() - detectionStart
            
            // Compute ambient light and update adaptive parameters
            val ambientLight = computeAmbientLightLevel(bmp) / 255f  // Normalize to 0-1
            detectionLogic.updateAmbientLight(ambientLight)
            
            // Use YOLO26 segmentation with NPU acceleration
            // Limit to 5 obstacles max to avoid OOM and improve performance
            val segStart = System.currentTimeMillis()
            val obstacles = detectionLogic.runSmartSegmentation(bmp).take(5)
            val segTime = System.currentTimeMillis() - segStart
            
            val totalTime = (System.currentTimeMillis() - currentTime) / 1000.0
            Log.d(TAG, "Frame timing: Depth=${depthTime}ms, Detection=${detectionTime}ms, Seg=${segTime}ms, Total=%.2fs".format(totalTime))
            
            // Skip expensive mask mapping - use original masks directly, depth functions handle scaling
            val mappedObstacles = obstacles.filter { obstacle ->
                when {
                    shouldDetectDoors -> obstacle.className != "door"
                    shouldDetectChairs -> obstacle.className != "chair"
                    shouldDetectCars -> obstacle.className != "car"
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

    private fun imageProxyToBitmap(image: ImageProxy, extraRotation: Float = 0f): Bitmap {
        val plane = image.planes[0]
        val buffer = plane.buffer
        val pixelStride = plane.pixelStride
        val rowStride = plane.rowStride
        val rowPadding = rowStride - pixelStride * image.width
        val bitmap = createBitmap(image.width + rowPadding / pixelStride, image.height)
        bitmap.copyPixelsFromBuffer(buffer)
        val croppedBitmap = Bitmap.createBitmap(bitmap, 0, 0, image.width, image.height)
        val rotationDegrees = image.imageInfo.rotationDegrees.toFloat() + extraRotation
        return if (rotationDegrees != 0f) {
            val matrix = Matrix().apply { postRotate(rotationDegrees) }
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
            voiceManager.speak(message)
            runOnUiThread {
                positionTextView.visibility = View.VISIBLE
                positionTextView.isSingleLine = false
                positionTextView.maxLines = 3
                positionTextView.text = message // Fixed: Changed from 'model' to 'message'
            }
        }
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
        // Filter obstacles that are blocking the path to the target
        // Increase proximity threshold to detect obstacles further away (1.5m radius)
        val obstacleProximityThreshold = 1.5f
        val blockingObstacles = obstacles.filter { obstacle ->
            val obstacleDepth = detectionLogic.avgMaskDepthFixed(depthMap, obstacle.mask)
            val obstacleDepthMeters = if (obstacleDepth.isFinite()) DEPTH_SCALE_FACTOR / obstacleDepth else Float.MAX_VALUE
            // Obstacle must be closer than target and within proximity threshold
            obstacleDepthMeters < depthMeters && obstacleDepthMeters < obstacleProximityThreshold && 
                detectionLogic.isObstacleInPath(obstacle.box, targetBox, bitmap.width.toFloat())
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
                val rightObstacles = detectionLogic.runSmartSegmentation(rightHalf).filter { it.className != targetClass }
                if (rightObstacles.isEmpty()) {
                    "The $targetClass is to your left, but there is $obstacleNames in the way. Move right to avoid it, then turn left."
                } else {
                    val rightObstacleNames = rightObstacles.joinToString(" and ") { it.className }
                    "The $targetClass is to your left, but there is $obstacleNames in the way. The right path is blocked by $rightObstacleNames."
                }
            }
            "right" -> {
                val leftHalf = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width / 2, bitmap.height)
                val leftObstacles = detectionLogic.runSmartSegmentation(leftHalf).filter { it.className != targetClass }
                if (leftObstacles.isEmpty()) {
                    "The $targetClass is to your right, but there is $obstacleNames in the way. Move left to avoid it, then turn right."
                } else {
                    val leftObstacleNames = leftObstacles.joinToString(" and ") { it.className }
                    "The $targetClass is to your right, but there is $obstacleNames in the way. The left path is blocked by $leftObstacleNames."
                }
            }
            else -> {
                val rightThird = Bitmap.createBitmap(bitmap, bitmap.width * 2 / 3, 0, bitmap.width / 3, bitmap.height)
                val rightObstacles = detectionLogic.runSmartSegmentation(rightThird).filter { it.className != targetClass }
                if (rightObstacles.isEmpty()) {
                    "The $targetClass is straight ahead, but there is $obstacleNames in the way. Move right to avoid it, then continue forward."
                } else {
                    val leftThird = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width / 3, bitmap.height)
                    val leftObstacles = detectionLogic.runSmartSegmentation(leftThird).filter { it.className != targetClass }
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
            voiceManager.speak("Starting detection, press on the icon that you want to detect")
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
            voiceManager.speak("Detecting ${type}s")
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
        if (!deepSceneDiscovery.initializationComplete) {
            voiceManager.speak("Model is not ready yet")
            runOnUiThread {
                positionTextView.text = "Model is not ready yet"
                detectButton.text = "Start Detection"
            }
            return
        }
        startGlowEffect()
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
        stopGlowEffect()
        swipeInstructionTextView.visibility = View.GONE
        isDeepSceneDiscoveryActive = false
        deepSceneDiscovery.stop()
        // Stop any ongoing TTS immediately
        if (::voiceManager.isInitialized) {
            voiceManager.stopTts()
        }
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
        val varianceThreshold = 800f
        return variance < varianceThreshold
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_PERMISSIONS_CODE) {
            if (permissions.isEmpty() || grantResults.isEmpty()) {
                runOnUiThread {
                    positionTextView.text = "Permission request canceled. Camera and notifications required."
                    downloadModelButton.visibility = View.VISIBLE
                    detectButton.isEnabled = false
                }
                return
            }

            val cameraIndex = permissions.indexOf(Manifest.permission.CAMERA)
            val notificationIndex = permissions.indexOf(Manifest.permission.POST_NOTIFICATIONS)
            val allPermissionsGranted = permissions.indices.all { grantResults[it] == PackageManager.PERMISSION_GRANTED }

            if (allPermissionsGranted) {
                initializeComponents()
                val modelFile = File(getExternalFilesDir(null), "$modelDIR/$MODEL_FILE_NAME")
                if (!modelFile.exists()) {
                    runOnUiThread {
                        downloadModelButton.visibility = View.VISIBLE
                        // detectButton enabled by YOLO loading, not LLM
                        positionTextView.text = "Download LLM for scene description, or use object detection."
                        checkExistingDownloads()
                    }
                }
            } else {
                if (cameraIndex != -1 && grantResults[cameraIndex] != PackageManager.PERMISSION_GRANTED) {
                    if (!ActivityCompat.shouldShowRequestPermissionRationale(this, Manifest.permission.CAMERA)) {
                        runOnUiThread {
                            positionTextView.text = "Camera permission permanently denied. Please enable in settings."
                            Toast.makeText(this, "Camera permission required. Go to app settings to enable.", Toast.LENGTH_LONG).show()
                            val intent = Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS)
                            intent.data = Uri.fromParts("package", packageName, null)
                            startActivity(intent)
                        }
                    } else {
                        runOnUiThread {
                            positionTextView.text = "Camera permission denied. Please grant to proceed."
                            Toast.makeText(this, "Please grant camera permission.", Toast.LENGTH_LONG).show()
                        }
                    }
                }
                if (notificationIndex != -1 && grantResults[notificationIndex] != PackageManager.PERMISSION_GRANTED) {
                    runOnUiThread {
                        positionTextView.text = "Notification permission denied. Download updates may not be shown."
                        Toast.makeText(this, "Please grant notification permission for download updates.", Toast.LENGTH_LONG).show()
                    }
                }
                if (cameraIndex != -1 && grantResults[cameraIndex] == PackageManager.PERMISSION_GRANTED) {
                    initializeComponents()
                    if (!File(getExternalFilesDir(null), "$modelDIR/$MODEL_FILE_NAME").exists()) {
                        runOnUiThread {
                            downloadModelButton.visibility = View.VISIBLE
                            // detectButton enabled by YOLO loading, not LLM
                            positionTextView.text = "Download LLM for scene description, or use object detection."
                            checkExistingDownloads()
                        }
                    }
                }
            }
        }
    }


    
    override fun onDestroy() {
        super.onDestroy()
        handler.removeCallbacksAndMessages(null)
        if (::voiceManager.isInitialized) voiceManager.shutdown()
        cameraExecutor.shutdown()
        cameraProvider?.unbindAll()
        if (::orientationEventListener.isInitialized) orientationEventListener.disable()
    }
}