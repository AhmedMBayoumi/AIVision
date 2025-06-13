package com.amb.aivision

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.google.ai.client.generativeai.GenerativeModel
import com.google.ai.client.generativeai.type.generationConfig
import com.google.ai.client.generativeai.type.content
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

@SuppressLint("SetTextI18n")
class DeepSceneDiscovery(private val context: Context) {

    companion object {
        private const val TAG = "DeepSceneDiscovery"
    }

    @Volatile
    private var isProcessing = false

    @Volatile
    private var readyToProcess = false

    private lateinit var generativeModel: GenerativeModel

    init {
        try {
            val modelName = "gemini-2.5-flash-preview-05-20"
            val apiKey = BuildConfig.GEMINI_API_KEY
            val generationConfig = generationConfig {
                temperature = 0.4f
                topK = 32
                topP = 1.0f
                maxOutputTokens = 8192
            }

            generativeModel = GenerativeModel(
                modelName = modelName,
                apiKey = apiKey,
                generationConfig = generationConfig
            )
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize GenerativeModel", e)
        }
    }

    fun start() {
        isProcessing = false
        readyToProcess = false
        Log.d(TAG, "Deep Scene Discovery started")

        (context as MainActivity).runOnUiThread {
            context.swipeInstructionTextView.text = "Swipe Down to Stop Detecting"
            context.swipeInstructionTextView.visibility = android.view.View.VISIBLE
            Log.d(TAG, "Set swipeInstructionTextView to VISIBLE")
        }
        context.speak("Starting Deep Scene Discovery")
    }

    fun stop() {
        isProcessing = false
        readyToProcess = false
        Log.d(TAG, "Deep Scene Discovery stopped")

        (context as MainActivity).runOnUiThread {
            context.swipeInstructionTextView.visibility = android.view.View.GONE
            Log.d(TAG, "Set swipeInstructionTextView to GONE")
        }
    }

    fun onSpeechFinished() {
        isProcessing = false
        if (!readyToProcess) {
            readyToProcess = true
            Log.d(TAG, "Startup announcement finished. Ready to process frames.")
        }
        Log.d(TAG, "Speech finished. Ready for next frame.")
    }

    fun processFrame(bitmap: Bitmap) {
        if (isProcessing || !readyToProcess) {
            Log.d(TAG, "Skipping frame processing: isProcessing=$isProcessing, readyToProcess=$readyToProcess")
            return
        }

        isProcessing = true
        Log.d(TAG, "Processing new frame...")

        (context as MainActivity).runOnUiThread {
            context.positionTextView.text = "Analyzing scene..."
        }

        CoroutineScope(Dispatchers.IO).launch {
            try {
                val prompt = "briefly describe the scene in front of the camera. if you see a paper say that there is a paper and try to read what is in it and if you can not read just say there is a paper with contents that is not clear. Additionally, if you see a door, car or a chair explain the path i should take to reach the it and how to avoid anything i can bump into. if there is no object from what i talked about then do not say anything about them and only describe the scene. do not use any introduction or ending just generate what i asked for."

                val inputContent = content {
                    image(bitmap)
                    text(prompt)
                }

                val response = generativeModel.generateContent(inputContent)

                response.text?.let {
                    if (!isProcessing) return@let
                    Log.d(TAG, "Gemini Response: $it")
                    context.runOnUiThread {
                        context.speak(it)
                        context.positionTextView.text = it
                    }
                } ?: run {
                    Log.e(TAG, "Gemini response was null.")
                    context.runOnUiThread {
                        context.speak("I could not analyze the scene.")
                        context.positionTextView.text = "Error: Null response from API."
                    }
                    onSpeechFinished()
                }

            } catch (e: Exception) {
                Log.e(TAG, "Error calling Gemini API: ${e.message}", e)
                context.runOnUiThread {
                    context.positionTextView.text = "Error: ${e.message}"
                    context.speak("There was an error.")
                }
                onSpeechFinished()
            }
        }
    }
}