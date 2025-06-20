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

    private lateinit var tflite: Interpreter
    private var gpuDelegate: GpuDelegate? = null
    private lateinit var segInterpreter: Interpreter
    private var segGpuDelegate: GpuDelegate? = null
    private lateinit var depthInterpreter: Interpreter
    private var depthGpuDelegate: GpuDelegate? = null

    private lateinit var detectionProcessor: ImageProcessor
    private lateinit var imageProcessor: ImageProcessor
    private lateinit var depthProcessor: ImageProcessor

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

            val segModel = try {
                FileUtil.loadMappedFile(context, "yolo11s-seg.tflite")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to load segmentation model: ${e.message}", e)
                context.runOnUiThread {
                    context.positionTextView.text = "Error loading segmentation model: ${e.message}"
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

    fun runSegmentation(roi: Bitmap): List<Obstacle> {
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
                            if (maskValue > 0.01f) {
                                mask[dy][dx] = 1f
                                activePixels++
                            }
                        }
                    }
                    if (activePixels >= 50) {
                        if ((context.shouldDetectCars && className == "car") || (context.shouldDetectChairs && className == "chair")) {
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
                            if (maskValue > 0.01f) {
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
                            if (maskValue > 0.01f) {
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