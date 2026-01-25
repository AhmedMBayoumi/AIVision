package com.amb.aivision

import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import java.util.Locale

class VoiceManager(
    private val context: Context,
    private val onCommandRecognized: (String) -> Unit,
    private val onTtsStart: () -> Unit = {},
    private val onTtsDone: () -> Unit = {},
    private val onTtsError: (String) -> Unit = {}
) : TextToSpeech.OnInitListener {

    private var speechRecognizer: SpeechRecognizer? = null
    private var tts: TextToSpeech? = null
    private var isRecognizerListening = false
    var isSpeaking = false
        private set
    var shouldListen = true

    companion object {
        private const val TAG = "VoiceManager"
    }

    init {
        tts = TextToSpeech(context, this)
        initSpeechRecognizer()
    }

    private fun initSpeechRecognizer() {
        if (SpeechRecognizer.isRecognitionAvailable(context)) {
            speechRecognizer = SpeechRecognizer.createSpeechRecognizer(context).apply {
                setRecognitionListener(object : RecognitionListener {
                    override fun onReadyForSpeech(params: Bundle?) {}
                    override fun onBeginningOfSpeech() {}
                    override fun onRmsChanged(rmsdB: Float) {}
                    override fun onBufferReceived(buffer: ByteArray?) {}
                    override fun onEndOfSpeech() {
                        isRecognizerListening = false
                    }

                    override fun onError(error: Int) {
                        isRecognizerListening = false
                        // Restart listening if we should be listening and not speaking
                        if (shouldListen && !isSpeaking) {
                            startListening()
                        }
                    }

                    override fun onResults(results: Bundle?) {
                        isRecognizerListening = false
                        val matches = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                        if (!matches.isNullOrEmpty()) {
                            val command = matches[0].lowercase()
                            onCommandRecognized(command)
                        }
                        // Restart listening
                        if (shouldListen && !isSpeaking) {
                            startListening()
                        }
                    }

                    override fun onPartialResults(partialResults: Bundle?) {}
                    override fun onEvent(eventType: Int, params: Bundle?) {}
                })
            }
        } else {
            Log.e(TAG, "Speech recognition not available")
        }
    }

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            tts?.language = Locale.US
            tts?.setSpeechRate(1.25f)
            tts?.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
                override fun onStart(utteranceId: String?) {
                    isSpeaking = true
                    onTtsStart()
                    stopListening() // Don't listen while speaking
                }

                override fun onDone(utteranceId: String?) {
                    isSpeaking = false
                    onTtsDone()
                    if (shouldListen) {
                        startListening()
                    }
                }

                @Deprecated("Deprecated in Java")
                override fun onError(utteranceId: String?) {
                    isSpeaking = false
                    onTtsError("TTS Error")
                }
                
                override fun onError(utteranceId: String?, errorCode: Int) {
                    isSpeaking = false
                    onTtsError("TTS Error Code: $errorCode")
                }
            })
        } else {
            Log.e(TAG, "TTS Initialization failed")
        }
    }

    fun speak(message: String, id: String = "messageId") {
        if (message.isBlank()) return
        tts?.let {
            if (isSpeaking) it.stop()
            val params = Bundle()
            params.putString(TextToSpeech.Engine.KEY_PARAM_UTTERANCE_ID, id)
            it.speak(message, TextToSpeech.QUEUE_FLUSH, params, id)
        }
    }

    fun startListening() {
        if (isRecognizerListening || isSpeaking || speechRecognizer == null) return
        try {
            val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
                putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
                putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.getDefault())
                putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 1)
            }
            // Must run on main thread
            val mainHandler = android.os.Handler(android.os.Looper.getMainLooper())
            mainHandler.post {
                try {
                    speechRecognizer?.startListening(intent)
                    isRecognizerListening = true
                } catch (e: Exception) {
                    Log.e(TAG, "Start listening failed: ${e.message}")
                    isRecognizerListening = false
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error starting speech recognizer: ${e.message}")
        }
    }

    fun stopListening() {
        if (!isRecognizerListening) return
        val mainHandler = android.os.Handler(android.os.Looper.getMainLooper())
        mainHandler.post {
            try {
                speechRecognizer?.stopListening()
                isRecognizerListening = false
            } catch (e: Exception) {
                 Log.e(TAG, "Stop listening failed: ${e.message}")
            }
        }
    }

    fun shutdown() {
        try {
            speechRecognizer?.destroy()
            tts?.shutdown()
        } catch (e: Exception) {
            Log.e(TAG, "Error shutting down VoiceManager: ${e.message}")
        }
    }
    
    fun stopTts() {
        if (tts?.isSpeaking == true) {
            tts?.stop()
        }
    }
}
