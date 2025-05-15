plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
}

android {
    namespace = "com.amb.aivision"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.amb.aivision"
        minSdk = 24
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
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
    kotlinOptions {
        jvmTarget = "11"
    }
}

dependencies {
    implementation(libs.androidx.core.ktx)
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
    // TensorFlow Lite dependencies
    implementation(libs.tensorflow.lite.v2161) // org.tensorflow:tensorflow-lite:2.16.1
    implementation(libs.tensorflow.lite.support.v044) // org.tensorflow:tensorflow-lite-support:0.4.4
    implementation(libs.tensorflow.lite.metadata.v044) // org.tensorflow:tensorflow-lite-metadata:0.4.4
    implementation(libs.tensorflow.lite.gpu.v2161) // org.tensorflow:tensorflow-lite-gpu:2.16.1
    implementation(libs.tensorflow.lite.gpu.api)
    implementation(libs.tensorflow.lite.gpu.delegate.plugin)
    implementation(libs.core)
//    implementation (libs.opencv.android)
}