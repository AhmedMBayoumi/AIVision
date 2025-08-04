import org.jetbrains.kotlin.gradle.dsl.JvmTarget
import java.util.Properties
import java.io.FileInputStream

plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    id("com.google.android.libraries.mapsplatform.secrets-gradle-plugin")
}

android {
    namespace = "com.amb.aivision"
    compileSdk = 36

    buildFeatures {
        buildConfig = true
    }

    defaultConfig {
        applicationId = "com.amb.aivision"
        minSdk = 33
        targetSdk = 36
        versionCode = 2
        versionName = "2.0"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        // Make the API token available in BuildConfig
        buildConfigField("String", "HF_TOKEN", "\"${project.findProperty("HF_TOKEN") ?: ""}\"")
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }

    kotlin {
        compilerOptions {
            jvmTarget.set(JvmTarget.JVM_11)
        }
    }
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.secrets.gradle.plugin)
    implementation(libs.androidx.appcompat)
    implementation(libs.material)
    implementation(libs.firebase.crashlytics.buildtools)
    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
    implementation(libs.androidx.camera.core)
    implementation(libs.androidx.camera.camera2)
    implementation(libs.androidx.camera.lifecycle)
    implementation(libs.androidx.camera.view)
    // TensorFlow Lite and MediaPipe dependencies
    implementation(libs.tensorflow.lite.v2161) // org.tensorflow:tensorflow-lite:2.16.1
    implementation(libs.tensorflow.lite.support.v044) // org.tensorflow:tensorflow-lite-support:0.4.4
    implementation(libs.tensorflow.lite.metadata.v044) // org.tensorflow:tensorflow-lite-metadata:0.4.4
    implementation(libs.tensorflow.lite.gpu.v2161) // org.tensorflow:tensorflow-lite-gpu:2.16.1
    implementation(libs.tensorflow.lite.gpu.api)
    implementation(libs.tensorflow.lite.gpu.delegate.plugin)
    implementation(libs.core)
    implementation(libs.opencv)
    implementation(libs.kotlinx.coroutines.android)
    implementation(libs.tasks.genai)
    implementation(libs.tasks.vision)
    implementation(libs.mediapipe.tasks.genai)
    implementation(libs.androidx.work.runtime.ktx)
}