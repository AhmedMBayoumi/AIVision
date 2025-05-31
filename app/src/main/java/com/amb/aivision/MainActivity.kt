package com.amb.aivision

import android.Manifest
import android.annotation.SuppressLint
import android.content.ContentValues
import android.content.pm.PackageManager
import android.graphics.*
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.os.Handler
import android.os.Looper
import android.provider.MediaStore
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import android.view.WindowManager
import android.widget.Button
import android.widget.TextView
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
import java.io.File
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.*

private const val TAG = "DoorDetection"

class MainActivity : AppCompatActivity(), TextToSpeech.OnInitListener {

    companion object {
        private const val CAMERA_PERMISSION_CODE = 1001
        private const val STORAGE_PERMISSION_CODE = 1002
        private const val PROCESSING_SIZE = 256           // for both seg & depth
        private const val PROXIMITY_THRESHOLD_M = 2.0f    // 2 meters
        private const val PROXIMITY_THRESHOLD_D = 0.075f
        private const val DETECTION_RESOLUTION = 640      // For door detection
        private const val DEPTH_SCALE_FACTOR = 69.375f    // For MiDaS depth to meters
        private const val DETECTION_INTERVAL_MS = 3000L   // 3 seconds
    }

    private lateinit var previewView: PreviewView
    private lateinit var positionTextView: TextView
    private lateinit var detectButton: Button

    private var lastDetectionTime = 0L
    private var isSpeaking = false

    private lateinit var tts: TextToSpeech
    private var shouldDetect = false
    private var canProcess = true

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

    // Reusable objects for door detection
    private val reusableBitmap by lazy {
        Bitmap.createBitmap(DETECTION_RESOLUTION, DETECTION_RESOLUTION, Bitmap.Config.ARGB_8888)
    }
    private val reusableCanvas by lazy { Canvas(reusableBitmap) }

    private var numDetections: Int = 0
    private val attributes: Int = 5 // Matches the output shape [1, 5, 8400]

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        if (OpenCVLoader.initLocal()) {
            Log.d("OpenCV", "OpenCV loaded successfully")
        } else {
            Log.e("OpenCV", "Failed to load OpenCV")
        }
        setContentView(R.layout.activity_main)

        previewView = findViewById(R.id.previewView)
        positionTextView = findViewById(R.id.positionTextView)
        detectButton = findViewById(R.id.detectButton)

        setupFullscreenUI()
        setupProcessors()
        if (!loadModels()) {
            positionTextView.text = "Failed to load models. Please check the app configuration."
            detectButton.isEnabled = false // Disable detection until models are loaded
            return
        }

        tts = TextToSpeech(this, this)
        detectButton.setOnClickListener { toggleDetection() }

        cameraExecutor = Executors.newSingleThreadExecutor()
        val permissionsToRequest = mutableListOf(Manifest.permission.CAMERA)
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
                CAMERA_PERMISSION_CODE
            )
        } else {
            startCamera()
        }
    }

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            tts.language = Locale.US
            tts.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
                override fun onStart(utteranceId: String?) {
                    isSpeaking = true
                    Log.d(TAG, "TTS started speaking")
                }

                override fun onDone(utteranceId: String?) {
                    isSpeaking = false
                    Log.d(TAG, "TTS finished speaking")
                }

                override fun onError(utteranceId: String?) {
                    isSpeaking = false
                    Log.e(TAG, "TTS error occurred")
                }
            })
        } else {
            Log.e(TAG, "TTS initialization failed")
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
            .add(ResizeOp(DETECTION_RESOLUTION, DETECTION_RESOLUTION, ResizeOp.ResizeMethod.BILINEAR))
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

            // Door detection model
            val model = try {
                FileUtil.loadMappedFile(this, "best(2)_float32.tflite")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to load YOLO model: ${e.message}", e)
                runOnUiThread {
                    positionTextView.text = "Error loading YOLO model: ${e.message}"
                }
                return false
            }

            if (useGpu) {
                try {
                    val gpuOptions = Interpreter.Options()
                    gpuDelegate = GpuDelegate(compatList.bestOptionsForThisDevice)
                    gpuOptions.addDelegate(gpuDelegate)
                    tflite = Interpreter(model, gpuOptions)
                    Log.d(TAG, "Door detection model loaded with GPU delegate")
                } catch (e: Exception) {
                    Log.w(TAG, "GPU delegate failed for YOLO model: ${e.message}. Falling back to CPU.", e)
                    useGpu = false
                }
            }
            if (!useGpu) {
                val cpuOptions = Interpreter.Options().apply {
                    setNumThreads(min(Runtime.getRuntime().availableProcessors(), 4))
                    setUseNNAPI(false)
                }
                tflite = Interpreter(model, cpuOptions)
                Log.d(TAG, "Door detection model loaded with CPU")
            }
            numDetections = tflite.getOutputTensor(0).shape()[2]

            // Segmentation model
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
                    Log.w(TAG, "GPU delegate failed for segmentation model: ${e.message}. Falling back to CPU.", e)
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

            // Depth model
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
                    Log.w(TAG, "GPU delegate failed for depth model: ${e.message}. Falling back to CPU.", e)
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

    @SuppressLint("SetTextI18n")
    private fun toggleDetection() {
        shouldDetect = !shouldDetect
        if (shouldDetect) {
            detectButton.text = "Stop detecting doors"
            Log.d(TAG, "Started continuous detection")
        } else {
            detectButton.text = "Detect Door"
            runOnUiThread { positionTextView.text = "Detection stopped" }
            Log.d(TAG, "Stopped continuous detection")
        }
    }

    private fun triggerDetection() {
        if (canProcess && shouldDetect) {
            shouldDetect = true
            Log.d(TAG, "Triggered detection")
        }
    }

    @SuppressLint("SetTextI18n")
    private fun onFrame(image: ImageProxy) {
        val currentTime = System.currentTimeMillis()
        if (!shouldDetect || !canProcess) {
            image.close()
            return
        }
        if (currentTime - lastDetectionTime < DETECTION_INTERVAL_MS || isSpeaking) {
            Log.d(TAG, "Skipping frame processing - Time since last: ${currentTime - lastDetectionTime}ms, TTS speaking: $isSpeaking")
            image.close()
            return
        }
        lastDetectionTime = currentTime
        canProcess = false
        try {
            val bmp = image.toBitmap() // Changed from image.toBitmapFromRGBA(reusableBitmap, reusableCanvas)
            image.close()


            // Check if the full image is mostly uniform
            if (isImageMostlyUniform(bmp)) {
                val msg = "You are going to hit something."
                speak(msg)
                runOnUiThread { positionTextView.text = msg }
                canProcess = true
                return
            }

            val (doorBox, position) = detectDoor(bmp)
            if (doorBox == null) {
                speak("No door detected, please move around")
                runOnUiThread { positionTextView.text = "No door" }
                canProcess = true
                return
            }
            val fullDepthMap = runDepthEstimation(bmp)
            val rawDoorDepth = avgDepthInBoxFixed(fullDepthMap, doorBox, bmp.width, bmp.height)
            val doorDepthMeters = if (rawDoorDepth.isFinite()) DEPTH_SCALE_FACTOR / rawDoorDepth else Float.MAX_VALUE
            Log.d(TAG, "onFrame: Door depth: $doorDepthMeters meters (raw: $rawDoorDepth)")
            val obstacles = runSegmentation(bmp)
            val blockingObstacles = obstacles.filter { obstacle ->
                val obstacleDepth = avgMaskDepthFixed(fullDepthMap, obstacle.mask, bmp.width, bmp.height)
                val obstacleDepthMeters = if (obstacleDepth.isFinite()) DEPTH_SCALE_FACTOR / obstacleDepth else Float.MAX_VALUE
                Log.d(TAG, "onFrame: Obstacle depth calculated: $obstacleDepthMeters meters (raw: $obstacleDepth)")
                obstacleDepthMeters < doorDepthMeters &&
                        obstacleDepthMeters < PROXIMITY_THRESHOLD_M &&
                        isObstacleInPath(obstacle.box, doorBox)
            }
            val msg = generateNavigationInstruction(doorBox, bmp.width, bmp.height, blockingObstacles, doorDepthMeters, position)
            speak(msg)
            runOnUiThread { positionTextView.text = msg }
        } catch (e: Exception) {
            Log.e(TAG, "Frame processing error: ${e.message}", e)
            runOnUiThread { positionTextView.text = "Error: ${e.message}" }
        } finally {
            canProcess = true
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
                if (mask[maskY][maskX] > 0.01f) {
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
        val pathWidth = doorBox.width() * 1.5f
        return abs(obstacleCenter - doorCenter) < pathWidth / 2
    }

    private fun generateNavigationInstruction(
        doorBox: RectF,
        imageWidth: Int,
        imageHeight: Int,
        blockingObstacles: List<Obstacle>,
        doorDepthMeters: Float,
        position: String
    ): String {
        return when {
            doorDepthMeters < PROXIMITY_THRESHOLD_D -> {
                "You reached the door."
            }
            blockingObstacles.isNotEmpty() -> {
                when (position) {
                    "mid" -> {
                        val hasLeftSpace = checkPathClear(doorBox, imageWidth, "left", blockingObstacles)
                        val hasRightSpace = checkPathClear(doorBox, imageWidth, "right", blockingObstacles)
                        when {
                            hasLeftSpace && hasRightSpace ->
                                "The door is in front of you, but there is an obstacle. You can move left or right to avoid it."
                            hasLeftSpace ->
                                "The door is in front of you, but there is an obstacle. Move slightly left to avoid it, then continue forward."
                            hasRightSpace ->
                                "The door is in front of you, but there is an obstacle. Move slightly right to avoid it, then continue forward."
                            else ->
                                "The door is in front of you, but there is no clear path to it."
                        }
                    }
                    "left" -> "The door is on the left, but there is an obstacle. Move slightly right first, then take left."
                    "right" -> "The door is on the right, but there is an obstacle. Move slightly left first, then take right."
                    else -> "The door is ahead, but there is an obstacle. Move around it."
                }
            }
            else -> {
                when (position) {
                    "left" -> "The door is slightly on the left."
                    "right" -> "The door is slightly on the right."
                    else -> "The door is in front of you, move forward."
                }
            }
        }
    }

    private fun checkPathClear(
        doorBox: RectF,
        imageWidth: Int,
        direction: String,
        obstacles: List<Obstacle>
    ): Boolean {
        val checkZone = when (direction) {
            "left" -> RectF(0f, doorBox.top, imageWidth * 0.4f, doorBox.bottom)
            "right" -> RectF(imageWidth * 0.6f, doorBox.top, imageWidth.toFloat(), doorBox.bottom)
            else -> return false
        }
        return obstacles.none { obstacle ->
            RectF.intersects(obstacle.box, checkZone)
        }
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

    private fun detectDoor(bitmap: Bitmap): Pair<RectF?, String> {
        val tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(bitmap)
        val processedImage = detectionProcessor.process(tensorImage)
        val inputBuffer = processedImage.buffer
        inputBuffer.rewind()
        val outputs = Array(1) { Array(5) { FloatArray(8400) } }
        tflite.run(inputBuffer, outputs)
        val threshold = 0.6f
        var bestDetection: Pair<RectF, Float>? = null
        var maxConfidence = -1f

        // Log all detections above threshold
        Log.d(TAG, "YOLO Model Output: Printing detections with confidence > $threshold")
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
                Log.d(TAG, "Detection $i: Confidence=$confidence, Box=[left=$left, top=$top, right=$right, bottom=$bottom]")
            }
        }

        // Select the best detection
        for (i in 0 until 8400) {
            val x = outputs[0][0][i]
            val y = outputs[0][1][i]
            val w = outputs[0][2][i]
            val h = outputs[0][3][i]
            val confidence = outputs[0][4][i]
            if (confidence > threshold && confidence > maxConfidence) {
                val centerX = x * bitmap.width
                val centerY = y * bitmap.height
                val widthScaled = w * bitmap.width
                val heightScaled = h * bitmap.height
                val left = centerX - widthScaled / 2
                val top = centerY - heightScaled / 2
                val right = centerX + widthScaled / 2
                val bottom = centerY + heightScaled / 2
                val rect = RectF(left, top, right, bottom)
                bestDetection = Pair(rect, confidence)
                maxConfidence = confidence
            }
        }
        if (bestDetection != null) {
            val rect = bestDetection.first
            val croppedBitmap = cropBitmap(bitmap, rect)
            if (confirmDoorWithClassicalMethods(croppedBitmap)) {
                val centerX = (rect.left + rect.right) / 2
                val normalizedX = centerX / bitmap.width
                val position = when {
                    normalizedX < 0.33 -> "left"
                    normalizedX < 0.66 -> "mid"
                    else -> "right"
                }
                Log.d(TAG, "Door confirmed at position: $position, confidence: ${bestDetection.second}, box: $rect")
                return Pair(rect, position)
            } else {
                Log.d(TAG, "Detected object is not a door")
            }
        }
        Log.d(TAG, "No door detected")
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
                if (aspectRatio >= 1.5 && aspectRatio <= 3.0) {
                    if (verticalLines >= 2 && horizontalLines >= 2) {
                        return true
                    }
                }
            }
        }
        return false
    }

    data class Obstacle(val box: RectF, val mask: Array<FloatArray>)

    private fun runSegmentation(roi: Bitmap): List<Obstacle> {
        try {
            Log.d(TAG, "runSegmentation: ROI dimensions: ${roi.width}x${roi.height}")
            val ti = TensorImage(DataType.FLOAT32)
            ti.load(roi)
            val input = imageProcessor.process(ti).buffer
            val inputArray = FloatArray(input.capacity() / 4).apply { input.asFloatBuffer().get(this) }
            Log.d(TAG, "runSegmentation: Input pixel range: min=${inputArray.minOrNull()}, max=${inputArray.maxOrNull()}")
            val detShape = segInterpreter.getOutputTensor(0).shape()
            val protoShape = segInterpreter.getOutputTensor(1).shape()
            Log.d(TAG, "runSegmentation: Detection output shape: ${detShape.joinToString()}")
            Log.d(TAG, "runSegmentation: Proto output shape: ${protoShape.joinToString()}")
            val detOut = Array(detShape[0]) { Array(detShape[1]) { FloatArray(detShape[2]) } }
            val protoOut = Array(protoShape[0]) { Array(protoShape[1]) { Array(protoShape[2]) { FloatArray(protoShape[3]) } } }
            val outputs = mapOf(0 to detOut, 1 to protoOut)
            segInterpreter.runForMultipleInputsOutputs(arrayOf(input), outputs)
            Log.d(TAG, "runSegmentation: Model inference completed")
            val raw = detOut[0]
            Log.d(TAG, "runSegmentation: Raw detections attributes: ${raw.size}, detections: ${raw[0].size}")
            val dets = mutableListOf<Pair<RectF, FloatArray>>()
            val threshold = 0.1f
            val numClasses = 80
            val maskCoefsCount = detShape[1] - 4 - numClasses
            val maskCoefsStartIdx = 4 + numClasses
            Log.d(TAG, "runSegmentation: numClasses=$numClasses, maskCoefsStartIdx=$maskCoefsStartIdx, maskCoefsCount=$maskCoefsCount")
            if (maskCoefsStartIdx + maskCoefsCount > detShape[1]) {
                Log.e(TAG, "Invalid mask coefficients range: start=$maskCoefsStartIdx, count=$maskCoefsCount, max=${detShape[1]}")
                return emptyList()
            }
            val confidences = mutableListOf<Float>()
            for (i in 0 until detShape[2]) {
                var maxClassProb = 0f
                for (c in 0 until numClasses) {
                    val prob = raw[4 + c][i]
                    if (prob > maxClassProb) maxClassProb = prob
                }
                confidences.add(maxClassProb)
            }
            val topConfidences = confidences.sortedDescending().take(10)
            Log.d(TAG, "runSegmentation: Top 10 confidences: ${topConfidences.joinToString()}")
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
                    val box = RectF(cx - ww/2, cy - hh/2, cx + ww/2, cy + hh/2)
                    val maskCoefs = FloatArray(maskCoefsCount) { c ->
                        val idx = maskCoefsStartIdx + c
                        if (idx < detShape[1]) raw[idx][i] else 0f
                    }
                    Log.d(TAG, "Detection $i: class=$maxClassIdx, conf=$maxClassProb, box=$box, coefs min=${maskCoefs.minOrNull()}, max=${maskCoefs.maxOrNull()}")
                    dets += box to maskCoefs
                }
            }
            Log.d(TAG, "runSegmentation: Detections after confidence threshold ($threshold): ${dets.size}")
            val final = applyNMS(dets, 0.4f)
            Log.d(TAG, "runSegmentation: Detections after NMS: ${final.size}")
            val obstacles = mutableListOf<Obstacle>()
            val protoH = protoShape[1]
            val protoW = protoShape[2]
            val protoC = protoShape[3]
            for ((box, coefs) in final) {
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
                            if (maskValue > 0.01f) {
                                mask[dy][dx] = 1f
                                activePixels++
                            }
                        }
                    }
                    Log.d(TAG, "Mask for box $box: values min=${maskValues.minOrNull()}, max=${maskValues.maxOrNull()}, active pixels=$activePixels")
                    val dilatedMask = dilateArray(mask, 3)
                    val activeAfter = dilatedMask.sumBy { row -> row.count { it > 0f } }
                    Log.d(TAG, "Mask pixels: before=$activePixels, after dilation=$activeAfter")
                    obstacles.add(Obstacle(box, dilatedMask))
                } catch (e: Exception) {
                    Log.e(TAG, "Error building mask: ${e.message}", e)
                }
            }
            return obstacles
        } catch (e: Exception) {
            Log.e(TAG, "Segmentation error: ${e.message}", e)
            return emptyList()
        }
    }

    private fun applyNMS(
        dets: List<Pair<RectF, FloatArray>>,
        iouThresh: Float
    ): List<Pair<RectF, FloatArray>> {
        if (dets.isEmpty()) return emptyList()
        val sorted = dets.sortedByDescending { it.second.sumByDouble { v: Float -> v.toDouble() }.toFloat() }
        val keep = mutableListOf<Pair<RectF, FloatArray>>()
        for ((box, coefs) in sorted) {
            if (keep.none { iou(it.first, box) > iouThresh }) {
                keep += box to coefs
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
            Log.d(TAG, "runDepthEstimation: Output shape: ${outShape.joinToString()}")
            val depthMap = if (outShape.size == 4) {
                val raw = Array(outShape[0]) { Array(outShape[1]) { Array(outShape[2]) { FloatArray(outShape[3]) } } }
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
        Log.d("ImageUniformity", "Image variance: $variance, threshold: $varianceThreshold")
        return variance < varianceThreshold
    }

    private fun speak(msg: String) {
        val params = Bundle()
        params.putString(TextToSpeech.Engine.KEY_PARAM_UTTERANCE_ID, "messageId")
        tts.speak(msg, TextToSpeech.QUEUE_FLUSH, params, "messageId")
    }



    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<out String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == CAMERA_PERMISSION_CODE) {
            if (grantResults.isNotEmpty() && grantResults.all { it == PackageManager.PERMISSION_GRANTED }) {
                startCamera()
            } else {
                Log.e(TAG, "Required permissions denied")
                runOnUiThread { positionTextView.text = "Camera and/or storage permission denied" }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        handler.removeCallbacks(detectionRunnable)
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
    }
}