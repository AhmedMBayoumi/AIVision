package com.amb.aivision

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.google.ai.client.generativeai.GenerativeModel
import com.google.ai.client.generativeai.type.generationConfig // Make sure this import is present
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

    private lateinit var generativeModel: GenerativeModel

    init {
        try {

            val modelName = "gemini-2.0-flash"
//            val apiKey = BuildConfig.GEMINI_API_KEY // This should now work
            val apiKey = "AIzaSyCypNFKB74uZXjZCa73Yd62CwHRXnl2vXM"
            // CORRECTED: Use the Kotlin DSL for generationConfig
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
        Log.d(TAG, "Deep Scene Discovery started")
    }

    fun stop() {
        isProcessing = true
        Log.d(TAG, "Deep Scene Discovery stopped")
    }

    fun isProcessing(): Boolean {
        return isProcessing
    }

    fun onSpeechFinished() {
        isProcessing = false
        Log.d(TAG, "Speech finished. Ready for next frame.")
    }


    fun processFrame(bitmap: Bitmap) {
        if (isProcessing) {
            return
        }

        isProcessing = true
        Log.d(TAG, "Processing new frame...")

        (context as MainActivity).runOnUiThread {
            context.positionTextView.text = "Analyzing scene..."
        }

        CoroutineScope(Dispatchers.IO).launch {
            try {
                val prompt = "Explain the scene and explain the path to take to reach the door."

                val inputContent = content {
                    image(bitmap)
                    text(prompt)
                }

                val response = generativeModel.generateContent(inputContent)

                response.text?.let {
                    Log.d(TAG, "Gemini Response: $it")
                    (context as MainActivity).runOnUiThread {
                        (context as MainActivity).speak(it)
                        context.positionTextView.text = it
                    }
                } ?: run {
                    Log.e(TAG, "Gemini response was null.")
                    (context as MainActivity).runOnUiThread {
                        (context as MainActivity).speak("I could not analyze the scene.")
                        context.positionTextView.text = "Error: Null response from API."
                    }
                    onSpeechFinished()
                }

            } catch (e: Exception) {
                Log.e(TAG, "Error calling Gemini API: ${e.message}", e)
                (context as MainActivity).runOnUiThread {
                    context.positionTextView.text = "Error: ${e.message}"
                    (context as MainActivity).speak("There was an error.")
                }
                onSpeechFinished()
            }
        }
    }
}