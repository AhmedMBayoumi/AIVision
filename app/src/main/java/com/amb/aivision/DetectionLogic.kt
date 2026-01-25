package com.amb.aivision

import android.annotation.SuppressLint
import android.graphics.Bitmap
import android.graphics.RectF
import android.util.Log
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
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

private const val TAG = "DetectionLogic"
@SuppressLint("SetTextI18n")
class DetectionLogic(private val context: MainActivity) {

    companion object {
        private const val PROCESSING_SIZE = 256
        private const val DETECTION_RESOLUTION = 640
        private const val DEPTH_RESOLUTION = 518  // Depth-Anything-V2 requires exactly 518x518
        private const val YOLO26_MAX_DETECTIONS = 300
        private const val YOLO26_FEATURES = 38  // 4 bbox + 2 confidence + 32 mask coefs
        private const val YOLO26_PROTO_SIZE = 160
        private const val YOLO26_MASK_COEFS = 32
        @Suppress("unused") private const val TEMPORAL_WINDOW_SIZE = 3
    }

    // Adaptive parameters that adjust based on environment
    data class AdaptiveParams(
        var detectionThreshold: Float = 0.65f,
        var proximityThresholdClose: Float = 0.25f,  // meters for "you have reached"
        var proximityThresholdWarn: Float = 0.5f,    // meters for obstacle warning
        var iouThreshold: Float = 0.5f,
        var depthScaleFactor: Float = 100.0f,
        var maskThreshold: Float = 0.6f
    )

    private val params = AdaptiveParams()
    private var currentAmbientLight: Float = 1.0f  // 0.0 = dark, 1.0 = bright

    // Temporal smoothing - track recent detections
    data class DetectionHistory(
        val className: String,
        val position: String,
        val confidence: Float,
        val timestamp: Long
    )
    private val recentDetections = mutableListOf<DetectionHistory>()
    private val historyMaxAge = 1000L  // 1 second window

    @Suppress("unused") // Used for class ID mapping from YOLO26
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

    private lateinit var yolo26DoorInterpreter: Interpreter  // YOLO26n-fp16 for door detection (NMS baked in)
    private var yolo26DoorGpuDelegate: GpuDelegate? = null
    private lateinit var yolo26SegInterpreter: Interpreter  // YOLO26 NMS-free segmentation (NPU accelerated)
    private lateinit var depthInterpreter: Interpreter
    private var depthGpuDelegate: GpuDelegate? = null

    private lateinit var detectionProcessor: ImageProcessor
    private lateinit var imageProcessor: ImageProcessor
    private lateinit var depthProcessor: ImageProcessor
    private lateinit var yolo26Processor: ImageProcessor  // For YOLO26's 640x640 input

    // YOLO26 door detection constants
    private val yolo26DoorMaxDetections = 300
    private val yolo26DoorFeatures = 6  // 4 bbox + 1 confidence + 1 class_id
    private val maskBuffer = Array(PROCESSING_SIZE) { FloatArray(PROCESSING_SIZE) }
    private val reusableMaskCoefs = FloatArray(YOLO26_MASK_COEFS)


    fun loadModels(): Boolean {
        try {
            val compatList = CompatibilityList()

            // Load YOLO26n-fp16 model for door detection (NMS baked in, output: [1, 300, 6])
            val yolo26DoorModel = try {
                FileUtil.loadMappedFile(context, "YOLO26n-fp16.tflite")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to load YOLO26n-fp16 model: ${e.message}", e)
                context.runOnUiThread {
                    context.positionTextView.text = "Error loading YOLO26n door model: ${e.message}"
                }
                return false
            }

            if (this::yolo26DoorInterpreter.isInitialized) {
                yolo26DoorInterpreter.close()
            }
            yolo26DoorGpuDelegate?.close()

            // Try GPU delegate first for YOLO26 door detection
            var yolo26DoorLoaded = false
            if (compatList.isDelegateSupportedOnThisDevice) {
                try {
                    val gpuOptions = Interpreter.Options()
                    yolo26DoorGpuDelegate = GpuDelegate(compatList.bestOptionsForThisDevice)
                    gpuOptions.addDelegate(yolo26DoorGpuDelegate)
                    yolo26DoorInterpreter = Interpreter(yolo26DoorModel, gpuOptions)
                    yolo26DoorLoaded = true
                    Log.i(TAG, "✓ YOLO26n door detection loaded with GPU delegate")
                } catch (e: Exception) {
                    Log.w(TAG, "GPU delegate failed for YOLO26 door: ${e.message}. Trying NNAPI...")
                }
            }
            
            // Fallback to NNAPI
            if (!yolo26DoorLoaded) {
                try {
                    val nnapiOptions = Interpreter.Options().apply {
                        numThreads = min(Runtime.getRuntime().availableProcessors(), 4)
                        useNNAPI = true
                    }
                    yolo26DoorInterpreter = Interpreter(yolo26DoorModel, nnapiOptions)
                    yolo26DoorLoaded = true
                    Log.i(TAG, "✓ YOLO26n door detection loaded with NNAPI (NPU)")
                } catch (e: Exception) {
                    Log.w(TAG, "NNAPI failed for YOLO26 door: ${e.message}. Falling back to CPU.")
                }
            }
            
            // Final fallback to multi-threaded CPU
            if (!yolo26DoorLoaded) {
                val cpuOptions = Interpreter.Options().apply {
                    numThreads = min(Runtime.getRuntime().availableProcessors(), 4)
                    useNNAPI = false
                }
                yolo26DoorInterpreter = Interpreter(yolo26DoorModel, cpuOptions)
                Log.i(TAG, "✓ YOLO26n door detection loaded with CPU (multi-threaded)")
            }


            // Load YOLO26 segmentation model (int8 quantized - optimized for NPU via NNAPI)
            val yolo26Model = try {
                FileUtil.loadMappedFile(context, "yolo26n-seg_int8.tflite")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to load YOLO26 segmentation model: ${e.message}", e)
                context.runOnUiThread {
                    context.positionTextView.text = "Error loading YOLO26 model: ${e.message}"
                }
                return false
            }

            // Try GPU delegate first (generally faster for float ops, works with int8 on many devices)
            var yolo26Loaded = false
            if (compatList.isDelegateSupportedOnThisDevice) {
                try {
                    val gpuOptions = Interpreter.Options()
                    val gpuDelegate = GpuDelegate(compatList.bestOptionsForThisDevice)
                    gpuOptions.addDelegate(gpuDelegate)
                    yolo26SegInterpreter = Interpreter(yolo26Model, gpuOptions)
                    yolo26Loaded = true
                    Log.i(TAG, "✓ YOLO26 loaded with GPU delegate")
                } catch (e: Exception) {
                    Log.w(TAG, "GPU delegate failed for YOLO26: ${e.message}. Trying NNAPI...")
                }
            }
            
            // Fallback to NNAPI (leverages NPU on supported devices)
            if (!yolo26Loaded) {
                try {
                    val nnapiOptions = Interpreter.Options().apply {
                        numThreads = min(Runtime.getRuntime().availableProcessors(), 4)
                        useNNAPI = true
                    }
                    yolo26SegInterpreter = Interpreter(yolo26Model, nnapiOptions)
                    yolo26Loaded = true
                    Log.i(TAG, "✓ YOLO26 loaded with NNAPI (NPU/DSP)")
                } catch (e: Exception) {
                    Log.w(TAG, "NNAPI failed for YOLO26: ${e.message}. Falling back to CPU.")
                }
            }
            
            // Final fallback to multi-threaded CPU
            if (!yolo26Loaded) {
                val cpuOptions = Interpreter.Options().apply {
                    numThreads = min(Runtime.getRuntime().availableProcessors(), 4)
                    useNNAPI = false
                }
                yolo26SegInterpreter = Interpreter(yolo26Model, cpuOptions)
                Log.i(TAG, "✓ YOLO26 loaded with CPU (multi-threaded)")
            }

            // Load Depth-Anything-V2 model (optimized for NPU via NNAPI)
            val depthModel = try {
                FileUtil.loadMappedFile(context, "Depth-Anything-V2.tflite")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to load Depth-Anything-V2 model: ${e.message}", e)
                context.runOnUiThread {
                    context.positionTextView.text = "Error loading depth model: ${e.message}"
                }
                return false
            }

            // Try GPU delegate first for Depth model (generally faster)
            var depthLoaded = false
            if (compatList.isDelegateSupportedOnThisDevice) {
                try {
                    val gpuOptions = Interpreter.Options()
                    depthGpuDelegate = GpuDelegate(compatList.bestOptionsForThisDevice)
                    gpuOptions.addDelegate(depthGpuDelegate)
                    depthInterpreter = Interpreter(depthModel, gpuOptions)
                    depthLoaded = true
                    Log.i(TAG, "✓ Depth-Anything-V2 loaded with GPU delegate")
                } catch (e: Exception) {
                    Log.w(TAG, "GPU delegate failed for depth: ${e.message}. Trying NNAPI...")
                }
            }
            
            // Fallback to NNAPI
            if (!depthLoaded) {
                try {
                    val nnapiOptions = Interpreter.Options().apply {
                        numThreads = min(Runtime.getRuntime().availableProcessors(), 4)
                        useNNAPI = true
                    }
                    depthInterpreter = Interpreter(depthModel, nnapiOptions)
                    depthLoaded = true
                    Log.i(TAG, "✓ Depth-Anything-V2 loaded with NNAPI (NPU)")
                } catch (e: Exception) {
                    Log.w(TAG, "NNAPI failed for depth: ${e.message}. Falling back to CPU.")
                }
            }
            
            // Final fallback to CPU
            if (!depthLoaded) {
                val cpuOptions = Interpreter.Options().apply {
                    numThreads = min(Runtime.getRuntime().availableProcessors(), 4)
                    useNNAPI = false
                }
                depthInterpreter = Interpreter(depthModel, cpuOptions)
                Log.i(TAG, "✓ Depth-Anything-V2 loaded with CPU (multi-threaded)")
            }

            return true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load models: ${e.message}", e)
            context.runOnUiThread { context.positionTextView.text = "Error loading models: ${e.message}" }
            return false
        }
    }

    fun setupProcessors() {
        detectionProcessor = ImageProcessor.Builder()
            .add(ResizeOp(DETECTION_RESOLUTION, DETECTION_RESOLUTION, ResizeOp.ResizeMethod.BILINEAR))
            .add(NormalizeOp(0f, 255f))
            .build()

        imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(PROCESSING_SIZE, PROCESSING_SIZE, ResizeOp.ResizeMethod.BILINEAR))
            .add(NormalizeOp(0f, 255f))
            .build()

        // Depth-Anything-V2 processor: 518x518 input, simple /255 normalization
        depthProcessor = ImageProcessor.Builder()
            .add(ResizeOp(DEPTH_RESOLUTION, DEPTH_RESOLUTION, ResizeOp.ResizeMethod.BILINEAR))
            .add(NormalizeOp(0f, 255f))  // Scales to [0, 1] range
            .build()

        // YOLO26 processor - int8 model expects pixel values normalized differently
        yolo26Processor = ImageProcessor.Builder()
            .add(ResizeOp(DETECTION_RESOLUTION, DETECTION_RESOLUTION, ResizeOp.ResizeMethod.BILINEAR))
            .add(NormalizeOp(0f, 255f))
            .build()
    }

    /**
     * Update adaptive parameters based on ambient light level (0.0 = dark, 1.0 = bright)
     */
    fun updateAmbientLight(lightLevel: Float) {
        currentAmbientLight = lightLevel.coerceIn(0f, 1f)
        // Lower detection threshold in low light (more sensitive, but accept lower confidence)
        params.detectionThreshold = if (lightLevel < 0.3f) 0.35f else 0.45f
        // Increase proximity warning in low light (be more cautious)
        params.proximityThresholdWarn = if (lightLevel < 0.3f) 0.7f else 0.5f
    }

    /**
     * Add detection to temporal history and check if it should be announced
     */
    @Suppress("unused") // Used for temporal smoothing of detection announcements
    private fun shouldAnnounceDetection(className: String, position: String, confidence: Float): Boolean {
        val now = System.currentTimeMillis()
        
        // Prune old entries
        recentDetections.removeAll { now - it.timestamp > historyMaxAge }
        
        // Add current detection
        recentDetections.add(DetectionHistory(className, position, confidence, now))
        
        // Count matching detections in recent history
        val matchingCount = recentDetections.count { it.className == className && it.position == position }
        
        // Require at least 2 consistent detections to announce
        return matchingCount >= 2
    }

    fun detectDoor(bitmap: Bitmap): Pair<RectF?, String> {
        if (!::yolo26DoorInterpreter.isInitialized) {
            Log.e(TAG, "YOLO26 door interpreter not initialized")
            return Pair(null, "")
        }
        
        try {
            val tensorImage = TensorImage(DataType.FLOAT32)
            tensorImage.load(bitmap)
            val processedImage = detectionProcessor.process(tensorImage)
            val inputBuffer = processedImage.buffer
            inputBuffer.rewind()
            
            // YOLO26n output: [1, 300, 6] - NMS is baked in
            // Format: [x, y, w, h, confidence, class_id] per detection
            val outputs = Array(1) { Array(yolo26DoorMaxDetections) { FloatArray(yolo26DoorFeatures) } }
            yolo26DoorInterpreter.run(inputBuffer, outputs)
            
            val threshold = 0.5f
            val detections = mutableListOf<Triple<RectF, Float, String>>()

            for (i in 0 until yolo26DoorMaxDetections) {
                val row = outputs[0][i]
                val x = row[0]  // center x (normalized 0-1)
                val y = row[1]  // center y (normalized 0-1)
                val w = row[2]  // width (normalized)
                val h = row[3]  // height (normalized)
                val confidence = row[4]
                // row[5] is class_id - YOLO26n detects general objects, we'll use heuristics + classical confirmation
                
                if (confidence <= threshold || confidence.isNaN() || !confidence.isFinite()) continue
                
                // Skip tiny or invalid boxes
                if (w <= 0.01f || h <= 0.01f) continue
                
                val centerX = x * bitmap.width
                val centerY = y * bitmap.height
                val widthScaled = w * bitmap.width
                val heightScaled = h * bitmap.height
                val left = centerX - widthScaled / 2
                val top = centerY - heightScaled / 2
                val right = centerX + widthScaled / 2
                val bottom = centerY + heightScaled / 2
                
                val rect = RectF(left, top, right, bottom)
                
                // Validate box bounds
                if (rect.width() < 20 || rect.height() < 20) continue
                
                // Clip to image bounds
                rect.left = max(0f, rect.left)
                rect.top = max(0f, rect.top)
                rect.right = min(bitmap.width.toFloat(), rect.right)
                rect.bottom = min(bitmap.height.toFloat(), rect.bottom)
                
                // Determine position from normalized x
                val position = when {
                    x < 0.33f -> "left"
                    x < 0.66f -> "mid"
                    else -> "right"
                }
                detections.add(Triple(rect, confidence, position))
            }

            // NMS is already applied in model, but sort by confidence for best selection
            val sortedDetections = detections.sortedByDescending { it.second }

            // Try to find a door using classical confirmation
            for (det in sortedDetections.take(5)) {
                val croppedBitmap = cropBitmap(bitmap, det.first)
                if (confirmDoorWithClassicalMethods(croppedBitmap)) {
                    return Pair(det.first, det.third)
                }
            }
            
            // If no classical confirmation, return best detection if aspect ratio suggests a door
            if (sortedDetections.isNotEmpty()) {
                val best = sortedDetections[0]
                val aspectRatio = best.first.height() / max(1f, best.first.width())
                // Doors are typically tall rectangles (aspect ratio 1.5 to 3.5)
                if (aspectRatio > 1.2f && aspectRatio < 4.0f) {
                    return Pair(best.first, best.third)
                }
            }
            
            return Pair(null, "")
        } catch (e: Exception) {
            Log.e(TAG, "YOLO26 door detection error: ${e.message}", e)
            return Pair(null, "")
        }
    }

    fun detectChair(bitmap: Bitmap): Pair<RectF?, String> {
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

    fun detectCar(bitmap: Bitmap): Pair<RectF?, String> {
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

    /**
     * Run segmentation using YOLO26 model
     * This is the primary segmentation method - always uses YOLO26 with NPU acceleration
     */
    @Suppress("unused") // Public API for obstacle segmentation
    fun runSegmentation(roi: Bitmap): List<Obstacle> {
        return runYolo26Segmentation(roi)
    }

    /**
     * Run YOLO26 NMS-free segmentation - newer architecture with end-to-end detection
     * Output format: (1, 300, 38) where 38 = 4 bbox + 2 confidence + 32 mask coefficients
     * No NMS required as YOLO26 uses a one-to-one head for end-to-end predictions
     */
    fun runYolo26Segmentation(roi: Bitmap, filterClasses: List<String>? = null): List<Obstacle> {
        if (!::yolo26SegInterpreter.isInitialized) {
            Log.e(TAG, "YOLO26 segmentation interpreter not initialized")
            return emptyList()
        }
        
        try {
            val ti = TensorImage(DataType.FLOAT32)
            ti.load(roi)
            val processedImage = yolo26Processor.process(ti)
            val inputBuffer = processedImage.buffer
            inputBuffer.rewind()
            
            // Get output tensor info for dequantization
            val detTensor = yolo26SegInterpreter.getOutputTensor(0)
            val protoTensor = yolo26SegInterpreter.getOutputTensor(1)
            
            // Output shapes: (1, 300, 38) and (1, 32, 160, 160)
            val detShape = detTensor.shape()
            val protoShape = protoTensor.shape()
            
            Log.d(TAG, "YOLO26 det shape: ${detShape.joinToString()}, proto shape: ${protoShape.joinToString()}")
            
            // Allocate output buffers based on model output type
            val numSlots = if (detShape.size == 3) detShape[1] else YOLO26_MAX_DETECTIONS
            val numFeatures = if (detShape.size == 3) detShape[2] else YOLO26_FEATURES
            
            // For int8 output, use ByteBuffer then dequantize
            val detOut: Array<FloatArray>
            val protoOut: Array<Array<FloatArray>>
            
            if (detTensor.dataType() == DataType.INT8 || detTensor.dataType() == DataType.UINT8) {
                // Int8 quantized output - need to dequantize
                val detParams = detTensor.quantizationParams()
                val protoParams = protoTensor.quantizationParams()
                
                val detByteBuffer = ByteBuffer.allocateDirect(numSlots * numFeatures)
                    .order(ByteOrder.nativeOrder())
                val protoByteBuffer = ByteBuffer.allocateDirect(YOLO26_MASK_COEFS * YOLO26_PROTO_SIZE * YOLO26_PROTO_SIZE)
                    .order(ByteOrder.nativeOrder())
                
                val outputs = mapOf(0 to detByteBuffer, 1 to protoByteBuffer)
                yolo26SegInterpreter.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputs)
                
                // Dequantize det output
                detByteBuffer.rewind()
                detOut = Array(numSlots) { FloatArray(numFeatures) }
                for (i in 0 until numSlots) {
                    for (j in 0 until numFeatures) {
                        val quantizedValue = (detByteBuffer.get().toInt() and 0xFF)  // uint8 to int
                        detOut[i][j] = detParams.scale * (quantizedValue - detParams.zeroPoint)
                    }
                }
                
                // Dequantize proto output
                protoByteBuffer.rewind()
                protoOut = Array(YOLO26_MASK_COEFS) { Array(YOLO26_PROTO_SIZE) { FloatArray(YOLO26_PROTO_SIZE) } }
                for (c in 0 until YOLO26_MASK_COEFS) {
                    for (y in 0 until YOLO26_PROTO_SIZE) {
                        for (x in 0 until YOLO26_PROTO_SIZE) {
                            val quantizedValue = (protoByteBuffer.get().toInt() and 0xFF)
                            protoOut[c][y][x] = protoParams.scale * (quantizedValue - protoParams.zeroPoint)
                        }
                    }
                }
            } else {
                // Float32 output - Model outputs NHWC format [1, 160, 160, 32]
                val detFloatOut = Array(1) { Array(numSlots) { FloatArray(numFeatures) } }
                val protoFloatOut = Array(1) { Array(YOLO26_PROTO_SIZE) { Array(YOLO26_PROTO_SIZE) { FloatArray(YOLO26_MASK_COEFS) } } }
                val outputs = mapOf(0 to detFloatOut, 1 to protoFloatOut)
                yolo26SegInterpreter.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputs)
                detOut = detFloatOut[0]
                // Transpose from NHWC [160, 160, 32] to internal format [32, 160, 160] for mask processing
                protoOut = Array(YOLO26_MASK_COEFS) { c ->
                    Array(YOLO26_PROTO_SIZE) { y ->
                        FloatArray(YOLO26_PROTO_SIZE) { x ->
                            protoFloatOut[0][y][x][c]
                        }
                    }
                }
            }
            
            val obstacles = mutableListOf<Obstacle>()
            
            // YOLO26 output: each row is [x, y, w, h, conf1, conf2, mask_coef_0...mask_coef_31]
            // conf1 and conf2 are objectness scores from dual heads, we use max
            for (i in 0 until numSlots) {
                val row = detOut[i]
                if (row.size < 6) continue
                
                val x = row[0]  // center x (normalized 0-1)
                val y = row[1]  // center y (normalized 0-1)
                val w = row[2]  // width (normalized)
                val h = row[3]  // height (normalized)
                val conf1 = row[4]
                val conf2 = row[5]
                val confidence = max(conf1, conf2)
                
                if (confidence <= 0.6f || confidence.isNaN() || !confidence.isFinite()) continue
                
                // Get mask coefficients (32 values) using reusable array
                for (c in 0 until YOLO26_MASK_COEFS) {
                    reusableMaskCoefs[c] = if (6 + c < row.size) row[6 + c] else 0f
                }
                
                // Scale box to image coordinates
                val cx = x * roi.width
                val cy = y * roi.height
                val ww = w * roi.width
                val hh = h * roi.height
                val box = RectF(cx - ww / 2, cy - hh / 2, cx + ww / 2, cy + hh / 2)
                
                // Validate box
                if (box.width() < 10 || box.height() < 10) continue
                if (box.left < 0 || box.top < 0 || box.right > roi.width || box.bottom > roi.height) {
                    // Clip to image bounds
                    box.left = max(0f, box.left)
                    box.top = max(0f, box.top)
                    box.right = min(roi.width.toFloat(), box.right)
                    box.bottom = min(roi.height.toFloat(), box.bottom)
                }
                
                // Keep only top 8 confident detections to avoid processing too many masks
                // This limit is crucial for performance - processing 100+ masks is too slow
                if (obstacles.size >= 8) break

                // Generate mask from proto and coefficients
                try {
                    // Reuse the class-level buffer instead of allocating new array every time
                    // Reset buffer first
                    for (i in 0 until PROCESSING_SIZE) {
                        for (j in 0 until PROCESSING_SIZE) {
                            maskBuffer[i][j] = 0f
                        }
                    }
                    var activePixels = 0
                    
                    for (dy in 0 until PROCESSING_SIZE) {
                        for (dx in 0 until PROCESSING_SIZE) {
                            val py = (dy * YOLO26_PROTO_SIZE / PROCESSING_SIZE).coerceIn(0, YOLO26_PROTO_SIZE - 1)
                            val px = (dx * YOLO26_PROTO_SIZE / PROCESSING_SIZE).coerceIn(0, YOLO26_PROTO_SIZE - 1)
                            
                            var maskValue = 0f
                            for (c in 0 until YOLO26_MASK_COEFS) {
                                maskValue += reusableMaskCoefs[c] * protoOut[c][py][px]
                            }
                            // Apply sigmoid
                            maskValue = 1.0f / (1.0f + exp(-maskValue))
                            
                            if (maskValue > params.maskThreshold) {
                                maskBuffer[dy][dx] = 1f
                                activePixels++
                            }
                        }
                    }
                    
                    if (activePixels >= 50) {
                        // YOLO26 doesn't provide class info in this format, infer from mask shape
                        val aspectRatio = box.height() / max(1f, box.width())
                        val className = inferClassFromAspectRatio(aspectRatio, box, roi)
                        
                        // Apply filter if specified
                        if (filterClasses != null && className !in filterClasses) continue
                        
                        // Skip if detecting this class
                        if ((context.shouldDetectCars && className == "car") || 
                            (context.shouldDetectChairs && className == "chair")) {
                            continue
                        }
                        
                        val dilatedMask = dilateArray(maskBuffer)
                        obstacles.add(Obstacle(box, dilatedMask, className))
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Error building YOLO26 mask: ${e.message}", e)
                }
            }
            
            Log.d(TAG, "YOLO26 segmentation found ${obstacles.size} obstacles")
            return obstacles
            
        } catch (e: Exception) {
            Log.e(TAG, "YOLO26 segmentation error: ${e.message}", e)
            return emptyList()
        }
    }
    
    /**
     * Infer object class from aspect ratio and size (heuristic for YOLO26)
     */
    private fun inferClassFromAspectRatio(aspectRatio: Float, box: RectF, image: Bitmap): String {
        val relativeWidth = box.width() / image.width
        val relativeHeight = box.height() / image.height
        val area = relativeWidth * relativeHeight
        
        return when {
            // Doors are typically tall and rectangular
            aspectRatio > 1.5f && aspectRatio < 3.5f && area > 0.05f -> "door"
            // Chairs are roughly square to slightly tall
            aspectRatio > 0.7f && aspectRatio < 1.8f && area < 0.2f -> "chair"
            // Cars are wide and not very tall
            aspectRatio < 0.8f && area > 0.1f -> "car"
            // People are tall
            aspectRatio > 1.5f && relativeHeight > 0.3f -> "person"
            // Default to generic obstacle
            else -> "obstacle"
        }
    }
    
    /**
     * Smart segmentation - always uses YOLO26 with NPU acceleration
     */
    fun runSmartSegmentation(roi: Bitmap): List<Obstacle> {
        return runYolo26Segmentation(roi)
    }

    /**
     * Detect chairs using YOLO26 segmentation with class filtering
     */
    private fun runSegmentationForChairs(roi: Bitmap): List<Obstacle> {
        return runYolo26Segmentation(roi, listOf("chair"))
    }

    /**
     * Detect cars using YOLO26 segmentation with class filtering
     */
    private fun runSegmentationForCars(roi: Bitmap): List<Obstacle> {
        return runYolo26Segmentation(roi, listOf("car"))
    }

    /**
     * Run depth estimation using Depth-Anything-V2
     * Input: 518x518x3, Output: 518x518 relative depth (higher values = farther)
     * We invert the output so higher values = closer (for proximity calculations)
     */
    fun runDepthEstimation(bmp: Bitmap): Array<FloatArray> {
        try {
            val ti = TensorImage(DataType.FLOAT32)
            ti.load(bmp)
            val input = depthProcessor.process(ti).buffer
            input.rewind()
            
            val outShape = depthInterpreter.getOutputTensor(0).shape()
            Log.d(TAG, "Depth output shape: ${outShape.joinToString()}")
            
            val depthMap = when (outShape.size) {
                4 -> {
                    // Shape: (1, H, W, 1)
                    val raw = Array(outShape[0]) { Array(outShape[1]) { Array(outShape[2]) { FloatArray(outShape[3]) } } }
                    depthInterpreter.run(input, raw)
                    Array(outShape[1]) { y -> FloatArray(outShape[2]) { x -> 1.0f - raw[0][y][x][0] } }  // Invert: higher=closer
                }
                3 -> {
                    // Shape: (1, 518, 518) - expected for Depth-Anything-V2
                    val raw = Array(outShape[0]) { Array(outShape[1]) { FloatArray(outShape[2]) } }
                    depthInterpreter.run(input, raw)
                    // Invert values: Depth-Anything-V2 outputs higher=farther, we need higher=closer
                    Array(outShape[1]) { y -> FloatArray(outShape[2]) { x -> 1.0f - raw[0][y][x] } }
                }
                2 -> {
                    // Shape: (518, 518) - unlikely but handle it
                    val raw = Array(outShape[0]) { FloatArray(outShape[1]) }
                    depthInterpreter.run(input, raw)
                    Array(outShape[0]) { y -> FloatArray(outShape[1]) { x -> 1.0f - raw[y][x] } }
                }
                else -> {
                    Log.e(TAG, "Unsupported depth output shape: ${outShape.joinToString()}")
                    Array(DEPTH_RESOLUTION) { FloatArray(DEPTH_RESOLUTION) { Float.MAX_VALUE } }
                }
            }
            return depthMap
        } catch (e: Exception) {
            Log.e(TAG, "Depth estimation error: ${e.message}", e)
            return Array(DEPTH_RESOLUTION) { FloatArray(DEPTH_RESOLUTION) { Float.MAX_VALUE } }
        }
    }

    fun avgDepthInBoxFixed(depthMap: Array<FloatArray>, box: RectF, imageWidth: Int, imageHeight: Int): Float {
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

    fun avgMaskDepthFixed(depthMap: Array<FloatArray>, mask: Array<FloatArray>): Float {
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

    /**
     * Check if an obstacle is blocking the path to the target.
     * Uses horizontal overlap to determine if the obstacle intersects with the corridor
     * from the user's position to the target.
     */
    fun isObstacleInPath(obstacleBox: RectF, targetBox: RectF, imageWidth: Float = 640f): Boolean {
        // Define a path corridor from user to target
        // The path is centered on the target and extends to cover a reasonable walking path
        val targetCenter = (targetBox.left + targetBox.right) / 2
        // Use a dynamic path width based on target width and image size
        val pathHalfWidth = max(targetBox.width() * 1.5f, imageWidth * 0.25f)
        val pathLeft = targetCenter - pathHalfWidth
        val pathRight = targetCenter + pathHalfWidth
        
        // Check horizontal overlap: obstacle overlaps with the path corridor
        val obstacleOverlapsHorizontally = obstacleBox.right > pathLeft && obstacleBox.left < pathRight
        
        // The obstacle must also be in front of us (we assume obstacles are detected in the path)
        return obstacleOverlapsHorizontally
    }

    private fun cropBitmap(bitmap: Bitmap, rect: RectF): Bitmap {
        val left = rect.left.toInt().coerceIn(0, bitmap.width)
        val top = rect.top.toInt().coerceIn(0, bitmap.height)
        val right = rect.right.toInt().coerceIn(0, bitmap.width)
        val bottom = rect.bottom.toInt().coerceIn(0, bitmap.height)
        return Bitmap.createBitmap(bitmap, left, top, right - left, bottom - top)
    }

    /**
     * Advanced classical door confirmation using multiple computer vision techniques:
     * 1. Line detection with parallel line pairing
     * 2. Rectangular frame detection
     * 3. Corner/handle detection via Harris corners
     * 4. Color uniformity analysis
     * 5. Symmetry analysis
     */
    private fun confirmDoorWithClassicalMethods(bitmap: Bitmap): Boolean {
        if (bitmap.width < 20 || bitmap.height < 20) return false
        
        val mat = Mat()
        Utils.bitmapToMat(bitmap, mat)
        val gray = Mat()
        Imgproc.cvtColor(mat, gray, Imgproc.COLOR_BGR2GRAY)
        
        // Apply Gaussian blur to reduce noise
        val blurred = Mat()
        Imgproc.GaussianBlur(gray, blurred, org.opencv.core.Size(5.0, 5.0), 0.0)
        
        // Adaptive edge detection based on image statistics
        val mean = org.opencv.core.Core.mean(blurred)
        val lowThresh = max(30.0, mean.`val`[0] * 0.4)
        val highThresh = min(200.0, mean.`val`[0] * 1.2)
        val edges = Mat()
        Imgproc.Canny(blurred, edges, lowThresh, highThresh)
        
        // Morphological operations to connect broken edges
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, org.opencv.core.Size(3.0, 3.0))
        Imgproc.dilate(edges, edges, kernel)
        Imgproc.erode(edges, edges, kernel)
        
        // Detect lines using probabilistic Hough transform
        val lines = Mat()
        Imgproc.HoughLinesP(edges, lines, 1.0, Math.PI / 180, 40, 30.0, 15.0)
        
        val verticalLines = mutableListOf<DoubleArray>()
        val horizontalLines = mutableListOf<DoubleArray>()
        var score = 0.0
        
        for (i in 0 until lines.rows()) {
            val line = lines.get(i, 0)
            val x1 = line[0]
            val y1 = line[1]
            val x2 = line[2]
            val y2 = line[3]
            val angle = atan2(y2 - y1, x2 - x1) * 180 / Math.PI
            
            // Classify lines with some tolerance
            if (abs(angle) < 15 || abs(angle - 180) < 15 || abs(angle + 180) < 15) {
                horizontalLines.add(line)
            } else if (abs(abs(angle) - 90) < 15) {
                verticalLines.add(line)
            }
        }
        
        // Feature 1: Parallel vertical lines (door frame sides)
        val parallelVerticalPairs = countParallelLinePairs(verticalLines, bitmap.width * 0.15, bitmap.width * 0.9)
        if (parallelVerticalPairs > 0) score += 25.0
        
        // Feature 2: Parallel horizontal lines (top/bottom of door)
        val parallelHorizontalPairs = countParallelLinePairs(horizontalLines, bitmap.height * 0.1, bitmap.height * 0.5)
        if (parallelHorizontalPairs > 0) score += 15.0
        
        // Feature 3: Sufficient vertical and horizontal lines for a rectangular frame
        if (verticalLines.size >= 2) score += 10.0
        if (horizontalLines.isNotEmpty()) score += 5.0
        
        // Feature 4: Contour analysis for rectangular shapes
        val contours = mutableListOf<MatOfPoint>()
        val hierarchy = Mat()
        Imgproc.findContours(edges.clone(), contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
        
        val minArea = bitmap.width * bitmap.height * 0.1  // At least 10% of image
        
        for (contour in contours) {
            val area = Imgproc.contourArea(contour)
            if (area < minArea) continue
            
            val approx = MatOfPoint2f()
            val contour2f = MatOfPoint2f(*contour.toArray())
            val epsilon = Imgproc.arcLength(contour2f, true) * 0.02
            Imgproc.approxPolyDP(contour2f, approx, epsilon, true)
            
            val vertices = approx.toArray().size
            if (vertices in 4..6) {
                val rect = Imgproc.boundingRect(MatOfPoint(*approx.toArray().map { org.opencv.core.Point(it.x, it.y) }.toTypedArray()))
                val aspectRatio = rect.height.toFloat() / max(1, rect.width)
                
                // Doors typically have aspect ratio between 1.5 and 3.5
                if (aspectRatio in 1.3..4.0) {
                    score += 20.0
                    
                    // Bonus for larger contours (more likely to be the door)
                    val areaRatio = area / (bitmap.width * bitmap.height)
                    if (areaRatio > 0.3) score += 10.0
                }
            }
        }
        
        // Feature 5: Harris corner detection for door handles/knobs
        val corners = Mat()
        Imgproc.cornerHarris(blurred, corners, 5, 3, 0.04)
        
        // Count significant corners (potential door hardware)
        val normalizedCorners = Mat()
        org.opencv.core.Core.normalize(corners, normalizedCorners, 0.0, 255.0, org.opencv.core.Core.NORM_MINMAX)
        var cornerCount = 0
        // Sample the right third of the image where handles usually are
        val handleRegionStart = (bitmap.width * 0.6).toInt()
        for (y in 0 until normalizedCorners.rows()) {
            for (x in handleRegionStart until normalizedCorners.cols()) {
                if (normalizedCorners.get(y, x)[0] > 150) {
                    cornerCount++
                }
            }
        }
        if (cornerCount in 5..200) score += 10.0  // Reasonable number of corners suggests handle/hardware
        
        // Feature 6: Color uniformity check (doors often have uniform color)
        val hsv = Mat()
        Imgproc.cvtColor(mat, hsv, Imgproc.COLOR_BGR2HSV)
        val channels = mutableListOf<Mat>()
        org.opencv.core.Core.split(hsv, channels)
        val satStdDev = org.opencv.core.MatOfDouble()
        val satMean = org.opencv.core.MatOfDouble()
        org.opencv.core.Core.meanStdDev(channels[1], satMean, satStdDev)
        
        // Low saturation variance suggests uniform color (common for doors)
        if (satStdDev.toArray().isNotEmpty() && satStdDev.toArray()[0] < 50) score += 10.0
        
        // Release Mats to prevent memory leaks
        mat.release()
        gray.release()
        blurred.release()
        edges.release()
        kernel.release()
        lines.release()
        hierarchy.release()
        corners.release()
        normalizedCorners.release()
        hsv.release()
        channels.forEach { it.release() }
        satStdDev.release()
        satMean.release()
        contours.forEach { it.release() }
        
        // Score threshold: 40+ indicates strong door likelihood
        Log.d(TAG, "Door confirmation score: $score")
        return score >= 40.0
    }
    
    /**
     * Count pairs of parallel lines that are spaced appropriately for a door frame
     */
    private fun countParallelLinePairs(lines: List<DoubleArray>, minSpacing: Double, maxSpacing: Double): Int {
        var pairs = 0
        for (i in lines.indices) {
            for (j in i + 1 until lines.size) {
                val line1 = lines[i]
                val line2 = lines[j]
                
                // Calculate average x position for vertical lines, y for horizontal
                val pos1 = (line1[0] + line1[2]) / 2
                val pos2 = (line2[0] + line2[2]) / 2
                val spacing = abs(pos1 - pos2)
                
                if (spacing in minSpacing..maxSpacing) {
                    pairs++
                }
            }
        }
        return pairs
    }

    private fun dilateArray(array: Array<FloatArray>): Array<FloatArray> {
        val result = Array(array.size) { FloatArray(array[0].size) }
        val radius = 1
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

    @Suppress("unused") // Available for future NMS needs
    private fun applyNMS(dets: List<Triple<RectF, FloatArray, String>>, iouThresh: Float): List<Triple<RectF, FloatArray, String>> {
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
}