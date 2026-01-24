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
        private const val YOLO26_MAX_DETECTIONS = 300
        private const val YOLO26_FEATURES = 38  // 4 bbox + 2 confidence + 32 mask coefs
        private const val YOLO26_PROTO_SIZE = 160
        private const val YOLO26_MASK_COEFS = 32
        private const val TEMPORAL_WINDOW_SIZE = 3
    }

    // Adaptive parameters that adjust based on environment
    data class AdaptiveParams(
        var detectionThreshold: Float = 0.45f,
        var proximityThresholdClose: Float = 0.25f,  // meters for "you have reached"
        var proximityThresholdWarn: Float = 0.5f,    // meters for obstacle warning
        var iouThreshold: Float = 0.5f,
        var depthScaleFactor: Float = 100.0f,
        var maskThreshold: Float = 0.5f
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

    private lateinit var tflite: Interpreter
    private var gpuDelegate: GpuDelegate? = null
    private lateinit var yolo26SegInterpreter: Interpreter  // YOLO26 NMS-free segmentation (NPU accelerated)
    private lateinit var depthInterpreter: Interpreter
    private var depthGpuDelegate: GpuDelegate? = null

    private lateinit var detectionProcessor: ImageProcessor
    private lateinit var imageProcessor: ImageProcessor
    private lateinit var depthProcessor: ImageProcessor
    private lateinit var yolo26Processor: ImageProcessor  // For YOLO26's 640x640 input

    private var numDetections: Int = 0


    fun loadModels(): Boolean {
        try {
            val compatList = CompatibilityList()
            var useGpu = compatList.isDelegateSupportedOnThisDevice

            val modelFile = if (context.useYolo12s) "yolo12s.tflite" else "yolo8n.tflite"
            val model = try {
                FileUtil.loadMappedFile(context, modelFile)
            } catch (e: Exception) {
                Log.e(TAG, "Failed to load YOLO model $modelFile: ${e.message}", e)
                context.runOnUiThread {
                    context.positionTextView.text = "Error loading YOLO model: ${e.message}"
                }
                return false
            }

            if (this::tflite.isInitialized) {
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

            // Try NNAPI first (leverages NPU on supported devices like Qualcomm, Samsung, MediaTek)
            // NNAPI provides hardware acceleration and can use NPU, DSP, or GPU depending on device
            var yolo26Loaded = false
            try {
                val nnapiOptions = Interpreter.Options().apply {
                    numThreads = min(Runtime.getRuntime().availableProcessors(), 4)
                    useNNAPI = true  // Enable NNAPI for NPU/DSP acceleration
                }
                yolo26SegInterpreter = Interpreter(yolo26Model, nnapiOptions)
                yolo26Loaded = true
                Log.d(TAG, "YOLO26 loaded with NNAPI (NPU/DSP acceleration enabled)")
            } catch (e: Exception) {
                Log.w(TAG, "NNAPI failed for YOLO26: ${e.message}. Trying GPU delegate...")
            }
            
            // Fallback to GPU delegate if NNAPI fails
            if (!yolo26Loaded && compatList.isDelegateSupportedOnThisDevice) {
                try {
                    val gpuOptions = Interpreter.Options()
                    val gpuDelegate = GpuDelegate(compatList.bestOptionsForThisDevice)
                    gpuOptions.addDelegate(gpuDelegate)
                    yolo26SegInterpreter = Interpreter(yolo26Model, gpuOptions)
                    yolo26Loaded = true
                    Log.d(TAG, "YOLO26 loaded with GPU delegate")
                } catch (e: Exception) {
                    Log.w(TAG, "GPU delegate failed for YOLO26: ${e.message}. Falling back to CPU.")
                }
            }
            
            // Final fallback to multi-threaded CPU
            if (!yolo26Loaded) {
                val cpuOptions = Interpreter.Options().apply {
                    numThreads = min(Runtime.getRuntime().availableProcessors(), 4)
                    useNNAPI = false
                }
                yolo26SegInterpreter = Interpreter(yolo26Model, cpuOptions)
                Log.d(TAG, "YOLO26 loaded with CPU (multi-threaded)")
            }

            val depthModel = try {
                FileUtil.loadMappedFile(context, "MiDas.tflite")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to load depth model: ${e.message}", e)
                context.runOnUiThread {
                    context.positionTextView.text = "Error loading depth model: ${e.message}"
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

        depthProcessor = ImageProcessor.Builder()
            .add(ResizeOp(PROCESSING_SIZE, PROCESSING_SIZE, ResizeOp.ResizeMethod.BILINEAR))
            .add(NormalizeOp(0f, 255f))
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
        val tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(bitmap)
        val processedImage = detectionProcessor.process(tensorImage)
        val inputBuffer = processedImage.buffer
        inputBuffer.rewind()
        val outputs = Array(1) { Array(5) { FloatArray(8400) } }
        tflite.run(inputBuffer, outputs)
        val threshold = 0.5f
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
                // Float32 output
                val detFloatOut = Array(1) { Array(numSlots) { FloatArray(numFeatures) } }
                val protoFloatOut = Array(1) { Array(YOLO26_MASK_COEFS) { Array(YOLO26_PROTO_SIZE) { FloatArray(YOLO26_PROTO_SIZE) } } }
                val outputs = mapOf(0 to detFloatOut, 1 to protoFloatOut)
                yolo26SegInterpreter.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputs)
                detOut = detFloatOut[0]
                protoOut = protoFloatOut[0]
            }
            
            val obstacles = mutableListOf<Obstacle>()
            val threshold = params.detectionThreshold
            
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
                
                if (confidence <= threshold || confidence.isNaN() || !confidence.isFinite()) continue
                
                // Get mask coefficients (32 values)
                val maskCoefs = FloatArray(YOLO26_MASK_COEFS) { c ->
                    if (6 + c < row.size) row[6 + c] else 0f
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
                
                // Generate mask from proto and coefficients
                try {
                    val mask = Array(PROCESSING_SIZE) { FloatArray(PROCESSING_SIZE) }
                    var activePixels = 0
                    
                    for (dy in 0 until PROCESSING_SIZE) {
                        for (dx in 0 until PROCESSING_SIZE) {
                            val py = (dy * YOLO26_PROTO_SIZE / PROCESSING_SIZE).coerceIn(0, YOLO26_PROTO_SIZE - 1)
                            val px = (dx * YOLO26_PROTO_SIZE / PROCESSING_SIZE).coerceIn(0, YOLO26_PROTO_SIZE - 1)
                            
                            var maskValue = 0f
                            for (c in 0 until YOLO26_MASK_COEFS) {
                                maskValue += maskCoefs[c] * protoOut[c][py][px]
                            }
                            // Apply sigmoid
                            maskValue = 1.0f / (1.0f + exp(-maskValue))
                            
                            if (maskValue > params.maskThreshold) {
                                mask[dy][dx] = 1f
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
                        
                        val dilatedMask = dilateArray(mask)
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

    fun runDepthEstimation(bmp: Bitmap): Array<FloatArray> {
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

    fun isObstacleInPath(obstacleBox: RectF, doorBox: RectF): Boolean {
        val obstacleCenter = (obstacleBox.left + obstacleBox.right) / 2
        val doorCenter = (doorBox.left + doorBox.right) / 2
        val pathWidth = doorBox.width() * 1.75f
        return abs(obstacleCenter - doorCenter) < pathWidth / 2
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