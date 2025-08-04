package com.amb.aivision

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import android.view.View
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.genai.llminference.GraphOptions
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import com.google.mediapipe.tasks.genai.llminference.LlmInferenceSession
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.lang.IllegalStateException

@SuppressLint("SetTextI18n")
class DeepSceneDiscovery(private val context: Context) {

    companion object {
        private const val TAG = "DeepSceneDiscovery"
        private const val MODEL_PATH = "models/gemma-3n-E4B-it-int4.task"
    }

    private var llmInference: LlmInference? = null
    private var isInitialized = false
    var initializationComplete = false
    private lateinit var mainActivity: MainActivity

    @Volatile
    private var isProcessing = false
    @Volatile
    private var readyToProcess = false

    private val inferenceLock = Any()

    suspend fun initialize() {
        if (isInitialized) {
            initializationComplete = true
            return
        }

        withContext(Dispatchers.IO) {
            try {
                val modelFile = File(context.getExternalFilesDir(null), MODEL_PATH)
                if (!modelFile.exists()) {
                    throw IllegalStateException("Model file not found: ${modelFile.absolutePath}")
                }

                val inferenceOptions = LlmInference.LlmInferenceOptions.builder()
                    .setModelPath(modelFile.absolutePath)
                    .setPreferredBackend(LlmInference.Backend.GPU)
                    .setMaxNumImages(1)
                    .setMaxTokens(1024)
                    .build()
                llmInference = LlmInference.createFromOptions(context, inferenceOptions)

                isInitialized = true
                initializationComplete = true
                Log.d(TAG, "LlmInference engine loaded successfully")

            } catch (e: Exception) {
                Log.e(TAG, "Failed to initialize model: ${e.message}", e)
                initializationComplete = false
                throw e
            }
        }
    }

    fun start() {
        if (!initializationComplete) {
            mainActivity.speak("Model is not ready yet")
            return
        }
        isProcessing = false
        readyToProcess = false
        mainActivity.runOnUiThread {
            mainActivity.swipeInstructionTextView.text = "Swipe Down to Stop Detecting"
            mainActivity.swipeInstructionTextView.visibility = View.VISIBLE
        }
        mainActivity.speak("Starting Deep Scene Discovery")
    }

    fun stop() {
        isProcessing = false
        readyToProcess = false
        mainActivity.runOnUiThread {
            mainActivity.swipeInstructionTextView.visibility = View.GONE
        }
    }

    fun onSpeechFinished() {
        isProcessing = false
        if (!readyToProcess) {
            readyToProcess = true
        }
    }

    fun processFrame(bitmap: Bitmap) {
        if (!initializationComplete || isProcessing || !readyToProcess || llmInference == null) {
            return
        }

        isProcessing = true
        mainActivity.runOnUiThread {
            mainActivity.positionTextView.text = ""
        }

        CoroutineScope(Dispatchers.Default).launch {
            var newSession: LlmInferenceSession? = null
            synchronized(inferenceLock) {
                try {
                    val sessionOptions = LlmInferenceSession.LlmInferenceSessionOptions.builder()
                        .setTopK(40)
                        .setTemperature(1.0f)
                        .setGraphOptions(
                            GraphOptions.builder()
                                .setEnableVisionModality(true)
                                .build()
                        )
                        .build()
                    newSession = LlmInferenceSession.createFromOptions(llmInference!!, sessionOptions)

                    val prompt = "You are a concise AI assistant for the visually impaired. Your response must be direct and follow these rules precisely, in this exact order. Generate only the information requested.\n" +
                            "\n" +
                            "1.  **Scene Description:** Provide a single, brief sentence describing the overall scene.\n" +
                            "\n" +
                            "2.  **Text Recognition:** Identify and quote any visible text from physical objects like signs and papers, or from digital screens. If text is present but unreadable, state \"Unclear text is visible.\"\n" +
                            "\n" +
                            "3.  **Navigation Path (Strict Rule):**\n" +
                            "    * **ONLY** if a **door**, **car**, or **chair** is clearly visible, describe the path to it and mention any obstacles.\n" +
                            "    * Do **NOT** describe a path to any other object.\n" +
                            "    * If none of these three specific objects are visible, omit this section from your response entirely." +
                            "Provide a single cohesive paragraph, do not reply with multiple disjoint points or sentences and do not announce the title of each point like (Scene Description, Text Recognition and Navigation Path). Finally, if there is no path or text then just skip it's point."

                    newSession.addQueryChunk(prompt)
                    newSession.addImage(BitmapImageBuilder(bitmap).build())

                    val result = newSession.generateResponse()

                    mainActivity.runOnUiThread {
                        mainActivity.positionTextView.text = result
                        if (result != null) {
                            mainActivity.speak(result)
                        }
                    }
                    onSpeechFinished()

                } catch (e: Exception) {
                    Log.e(TAG, "Error during inference", e)
                    mainActivity.runOnUiThread {
                        mainActivity.speak("There was an error during analysis.")
                    }
                    onSpeechFinished()
                } finally {
                    newSession?.close()
                }
            }
        }
    }

    fun setMainActivity(activity: MainActivity) {
        this.mainActivity = activity
    }
}
