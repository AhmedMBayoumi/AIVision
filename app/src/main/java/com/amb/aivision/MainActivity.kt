package com.amb.aivision

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.*
import android.speech.tts.UtteranceProgressListener
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.speech.tts.TextToSpeech
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
class MainActivity : AppCompatActivity(), TextToSpeech.OnInitListener {

    companion object {
        private const val CAMERA_PERMISSION_CODE = 1001
        private const val PROCESSING_SIZE = 256           // for both seg & depth
        private const val PROXIMITY_THRESHOLD_M = 2.0f    // 2 meters
        private const val PROXIMITY_THRESHOLD_D = 0.075f
        private const val DETECTION_RESOLUTION = 640      // For door detection
        private const val TAG = "SegmentationDebug"       // For logging
        private const val DEPTH_SCALE_FACTOR = 69.375f    // For MiDaS depth to meters
        private const val DETECTION_INTERVAL_MS = 3000L   // 2 seconds

        private var lastDetectionTime = 0L // Add this new variable
    }

    private lateinit var previewView: PreviewView
    private lateinit var positionTextView: TextView
    private lateinit var detectButton: Button

    private var lastDetectionTime = 0L // Add this new variable
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
        setContentView(R.layout.activity_main)

        previewView = findViewById(R.id.previewView)
        positionTextView = findViewById(R.id.positionTextView)
        detectButton = findViewById(R.id.detectButton)

        setupFullscreenUI()
        setupProcessors()
        if (!loadModels()) {
            finish()
            return
        }

        tts = TextToSpeech(this, this)
        detectButton.setOnClickListener { toggleDetection() }

        cameraExecutor = Executors.newSingleThreadExecutor()
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.CAMERA),
                CAMERA_PERMISSION_CODE
            )
        }
    }


    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            tts.language = Locale.US

            // Add a UtteranceProgressListener to track speaking status
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
            // Door detection model
            val compatList = CompatibilityList()
            val options = Interpreter.Options()
            if (compatList.isDelegateSupportedOnThisDevice) {
                val delegateOptions = compatList.bestOptionsForThisDevice
                gpuDelegate = GpuDelegate(delegateOptions)
                options.addDelegate(gpuDelegate)
            } else {
                options.setNumThreads(min(Runtime.getRuntime().availableProcessors(), 4))
                options.setUseNNAPI(false)
            }
            val model = FileUtil.loadMappedFile(this, "yolo12s.tflite")
            tflite = Interpreter(model, options)
            numDetections = tflite.getOutputTensor(0).shape()[2]

            // Segmentation model
            FileUtil.loadMappedFile(this, "yolo11s-seg.tflite").let { mapped ->
                val opt = Interpreter.Options().apply {
                    val gpu = GpuDelegate()
                    addDelegate(gpu).also { segGpuDelegate = gpu }
                }
                segInterpreter = Interpreter(mapped, opt)
            }

            // Depth model
            FileUtil.loadMappedFile(this, "MiDas.tflite").let { mapped ->
                val opt = Interpreter.Options().apply {
                    val gpu = GpuDelegate()
                    addDelegate(gpu).also { depthGpuDelegate = gpu }
                }
                depthInterpreter = Interpreter(mapped, opt)
            }
            return true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load models: ${e.message}", e)
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

    private fun cropROI(bitmap: Bitmap, position: String): Bitmap {
        val w = bitmap.width
        val h = bitmap.height
        val roiWidth = w / 3
        val roiHeight = h / 3
        val leftX = when (position) {
            "left" -> 0
            "mid"  -> (w - roiWidth) / 2
            else   -> w - roiWidth
        }
        val topY = (h - roiHeight) / 2
        Log.d(TAG, "cropROI: position=$position, leftX=$leftX, topY=$topY, roiWidth=$roiWidth, roiHeight=$roiHeight")
        return Bitmap.createBitmap(bitmap, leftX, topY, roiWidth, roiHeight)
    }

    private fun toggleDetection() {
        shouldDetect = !shouldDetect

        if (shouldDetect) {
            detectButton.text = "Stop detecting doors"
            // No need for handler.post here - detection will happen through onFrame
            Log.d(TAG, "Started continuous detection")
        } else {
            detectButton.text = "Detect Door"
            runOnUiThread { positionTextView.text = "Detection stopped" }
            Log.d(TAG, "Stopped continuous detection")
        }
    }

    private fun triggerDetection() {
        if (canProcess && shouldDetect) {
            shouldDetect = true // Ensure onFrame processes the frame
            Log.d(TAG, "Triggered detection")
        }
    }

    private fun onFrame(image: ImageProxy) {
        val currentTime = System.currentTimeMillis()

        // Check if we should process this frame
        if (!shouldDetect || !canProcess) {
            image.close()
            return
        }

        // Check if enough time has passed since the last detection
        // AND check that TTS is not currently speaking
        if (currentTime - lastDetectionTime < DETECTION_INTERVAL_MS || isSpeaking) {
            Log.d(TAG, "Skipping frame processing - Time since last: ${currentTime - lastDetectionTime}ms, TTS speaking: $isSpeaking")
            image.close()
            return
        }

        // Update the last detection time
        lastDetectionTime = currentTime
        canProcess = false
        try {
            // 1) Convert to bitmap & prepare input
            val bmp = image.toBitmapFromRGBA(reusableBitmap, reusableCanvas)
            image.close()

            // 2) Run door detection to get `doorBox: RectF?` and `position: String`
            val (doorBox, position) = detectDoor(bmp)
            if (doorBox == null) {
                speak("No door detected, please move around")
                runOnUiThread { positionTextView.text = "No door" }
                canProcess = true
                return
            }

            // 3) Compute ROI coordinates
            val w = bmp.width
            val h = bmp.height
            val roiWidth = w / 3
            val roiHeight = h / 3
            val leftX = when (position) {
                "left" -> 0
                "mid"  -> (w - roiWidth) / 2
                else   -> w - roiWidth
            }
            val topY = (h - roiHeight) / 2

            // 4) Crop ROI
            val roi = Bitmap.createBitmap(bmp, leftX, topY, roiWidth, roiHeight)

            // 5) Run segmentation on ROI
            val obstacles = runSegmentation(roi)
            Log.d(TAG, "onFrame: Number of obstacles detected: ${obstacles.size}")

            // 6) Run MiDaS depth on ROI
            val depthMap = runDepthEstimation(roi)

            // 7) Compute door depth
            val rawDoorDepth = avgDepthInBox(depthMap, doorBox, leftX, topY, roiWidth, roiHeight)
            val doorDepth = if (rawDoorDepth.isFinite()) DEPTH_SCALE_FACTOR / rawDoorDepth else Float.MAX_VALUE
            Log.d(TAG, "onFrame: Door depth: $doorDepth meters (raw: $rawDoorDepth)")

            // 8) Compute each obstacle depth, store them, and check proximity
            val obstacleDepths = mutableListOf<Float>()
            val blocking = obstacles.any { obs ->
                val rawD = avgMaskDepth(depthMap, obs.mask)
                val d = if (rawD.isFinite()) DEPTH_SCALE_FACTOR / rawD else Float.MAX_VALUE
                Log.d(TAG, "onFrame: Obstacle depth calculated: $d meters (raw: $rawD)")
                obstacleDepths.add(d)
                d < doorDepth && d < PROXIMITY_THRESHOLD_M
            }

            // 9) Choose TTS guidance and append depth values
            val msg = if (blocking) {
                if (position == "mid") {
                    // Check left region
                    val leftRoi = cropROI(bmp, "left")
                    val leftObstacles = runSegmentation(leftRoi)
                    Log.d(TAG, "onFrame: Left region obstacles: ${leftObstacles.size}")
                    if (leftObstacles.isEmpty()) {
                        "The door is in front of you, but there is an obstacle. Move slightly left to avoid it, then continue forward."
                    } else {
                        // Check right region
                        val rightRoi = cropROI(bmp, "right")
                        val rightObstacles = runSegmentation(rightRoi)
                        Log.d(TAG, "onFrame: Right region obstacles: ${rightObstacles.size}")
                        if (rightObstacles.isEmpty()) {
                            "The door is in front of you, but there is an obstacle. Move slightly right to avoid it, then continue forward."
                        } else {
                            "The door is in front of you, but there is no clear path to it."
                        }
                    }
                } else {
                    when (position) {
                        "left"  -> "The door is on the left, but there is an obstacle. Move slightly right first, then take left."
                        "right" -> "The door is on the right, but there is an obstacle. Move slightly left first, then take right."
                        else    -> "The door is in front of you, but there is an obstacle. Move slightly left or right to avoid it, then continue forward."
                    }
                }
            } else if (doorDepth < PROXIMITY_THRESHOLD_D) {
                "You reached the door."
            } else {
                when (position) {
                    "left"  -> "The door is slightly on the left."
                    "right" -> "The door is slightly on the right."
                    else    -> "The door is in front of you, move forward."
                }
            }

            speak(msg)
            runOnUiThread { positionTextView.text = msg }
        } catch (e: Exception) {
            Log.e(TAG, "Frame processing error: ${e.message}", e)
            runOnUiThread { positionTextView.text = "Error: ${e.message}" }
        } finally {
            canProcess = true
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

    // ---- Door detection ----
    private fun detectDoor(bitmap: Bitmap): Pair<RectF?, String> {
        val tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(bitmap)
        val processedImage = detectionProcessor.process(tensorImage)

        val inputBuffer = processedImage.buffer
        inputBuffer.rewind()

        // Run model
        val outputs = Array(1) { Array(5) { FloatArray(8400) } }
        tflite.run(inputBuffer, outputs)

        // Detection logic for highest probability
        val threshold = 0.6f
        var bestDetection: Pair<RectF, Float>? = null
        var maxConfidence = -1f
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
            val centerX = (rect.left + rect.right) / 2
            val normalizedX = centerX / bitmap.width
            val position = when {
                normalizedX < 0.33 -> "left"
                normalizedX < 0.66 -> "mid"
                else -> "right"
            }
            Log.d(TAG, "Door detected at position: $position, confidence: ${bestDetection.second}, box: $rect")
            return Pair(rect, position)
        }
        Log.d(TAG, "No door detected")
        return Pair(null, "")
    }

    // ---- Segmentation ----
    data class Obstacle(val box: RectF, val mask: Array<FloatArray>)

    private fun runSegmentation(roi: Bitmap): List<Obstacle> {
        try {
            // 1) Preprocess
            Log.d(TAG, "runSegmentation: ROI dimensions: ${roi.width}x${roi.height}")
            val ti = TensorImage(DataType.FLOAT32)
            ti.load(roi)
            val input = imageProcessor.process(ti).buffer
            val inputArray = FloatArray(input.capacity() / 4).apply { input.asFloatBuffer().get(this) }
            Log.d(TAG, "runSegmentation: Input pixel range: min=${inputArray.minOrNull()}, max=${inputArray.maxOrNull()}")

            // 2) Allocate outputs based on model shapes
            val detShape = segInterpreter.getOutputTensor(0).shape()
            val protoShape = segInterpreter.getOutputTensor(1).shape()
            Log.d(TAG, "runSegmentation: Detection output shape: ${detShape.contentToString()}")
            Log.d(TAG, "runSegmentation: Proto output shape: ${protoShape.contentToString()}")

            val detOut = Array(detShape[0]) { Array(detShape[1]) { FloatArray(detShape[2]) } }
            val protoOut = Array(protoShape[0]) { Array(protoShape[1]) { Array(protoShape[2]) { FloatArray(protoShape[3]) } } }
            val outputs = mapOf(0 to detOut, 1 to protoOut)

            // 3) Run inference
            segInterpreter.runForMultipleInputsOutputs(arrayOf(input), outputs)
            Log.d(TAG, "runSegmentation: Model inference completed")

            // 4) Parse detections
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

            // Collect all confidences for top 10
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

            // 5) Build masks at 256x256 resolution
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

    // ---- Depth estimation ----
    private fun runDepthEstimation(bmp: Bitmap): Array<FloatArray> {
        try {
            Log.d(TAG, "runDepthEstimation: Input bitmap dimensions: ${bmp.width}x${bmp.height}")
            val ti = TensorImage(DataType.FLOAT32)
            ti.load(bmp)
            val input = depthProcessor.process(ti).buffer
            val outShape = depthInterpreter.getOutputTensor(0).shape()
            Log.d(TAG, "runDepthEstimation: Output shape: ${outShape.contentToString()}")
            val depthMap = if (outShape.size == 4) {
                val raw = Array(outShape[0]) { Array(outShape[1]) { Array(outShape[2]) { FloatArray(outShape[3]) } } }
                depthInterpreter.run(input, raw)
                Array(outShape[1]) { y -> FloatArray(outShape[2]) { x -> raw[0][y][x][0] } }
            } else if (outShape.size == 3) {
                val raw = Array(outShape[0]) { Array(outShape[1]) { FloatArray(outShape[2]) } }
                depthInterpreter.run(input, raw)
                raw[0]
            } else {
                Log.e(TAG, "Unsupported depth output shape: ${outShape.contentToString()}")
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

    // Compute avg depth inside door box
    private fun avgDepthInBox(
        depth: Array<FloatArray>,
        box: RectF,
        roiLeft: Int,
        roiTop: Int,
        roiWidth: Int,
        roiHeight: Int
    ): Float {
        var sum = 0f
        var cnt = 0
        val depthH = depth.size
        val depthW = depth[0].size
        for (y in 0 until depthH) {
            for (x in 0 until depthW) {
                val fullX = (x.toFloat() / (depthW - 1)) * (roiWidth - 1) + roiLeft
                val fullY = (y.toFloat() / (depthH - 1)) * (roiHeight - 1) + roiTop
                if (box.contains(fullX, fullY)) {
                    val depthValue = depth[y][x]
                    if (!depthValue.isNaN() && depthValue.isFinite()) {
                        sum += depthValue
                        cnt++
                    }
                }
            }
        }
        return if (cnt > 0) sum / cnt else Float.MAX_VALUE
    }

    // Compute avg depth in obstacle mask
    private fun avgMaskDepth(
        depth: Array<FloatArray>,
        mask: Array<FloatArray>
    ): Float {
        var sum = 0f
        var cnt = 0
        for (y in 0 until depth.size) {
            for (x in 0 until depth[0].size) {
                if (mask[y][x] > 0.01f) {
                    val depthValue = depth[y][x]
                    if (!depthValue.isNaN() && depthValue.isFinite()) {
                        sum += depthValue
                        cnt++
                    }
                }
            }
        }
        return if (cnt > 0) sum / cnt else Float.MAX_VALUE
    }

    private fun speak(msg: String) {
        val params = Bundle()
        params.putString(TextToSpeech.Engine.KEY_PARAM_UTTERANCE_ID, "messageId")
        tts.speak(msg, TextToSpeech.QUEUE_FLUSH, params, "messageId")
    }

    // Extension: ImageProxy → Bitmap for door detection
    private fun ImageProxy.toBitmapFromRGBA(reusableBitmap: Bitmap, canvas: Canvas): Bitmap {
        val buffer = planes[0].buffer
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        bitmap.copyPixelsFromBuffer(buffer)
        canvas.drawBitmap(
            bitmap,
            Rect(0, 0, bitmap.width, bitmap.height),
            Rect(0, 0, reusableBitmap.width, reusableBitmap.height),
            null
        )
        if (bitmap != reusableBitmap) {
            bitmap.recycle()
        }
        return reusableBitmap
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<out String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == CAMERA_PERMISSION_CODE &&
            grantResults.firstOrNull() == PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        handler.removeCallbacks(detectionRunnable)
        tflite.close()
        gpuDelegate?.close()
        segInterpreter.close()
        segGpuDelegate?.close()
        depthInterpreter.close()
        depthGpuDelegate?.close()
        cameraExecutor.shutdown()
        tts.shutdown()
    }
}