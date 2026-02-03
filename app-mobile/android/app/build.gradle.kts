plugins {
    id("com.android.application")
    id("kotlin-android")
    // The Flutter Gradle Plugin must be applied after the Android and Kotlin Gradle plugins.
    id("dev.flutter.flutter-gradle-plugin")
}

android {
    namespace = "com.example.sudoku_solver"
    compileSdk = flutter.compileSdkVersion
    ndkVersion = flutter.ndkVersion

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = JavaVersion.VERSION_17.toString()
    }

    defaultConfig {
        // TODO: Specify your own unique Application ID (https://developer.android.com/studio/build/application-id.html).
        applicationId = "com.example.sudoku_solver"
        // You can update the following values to match your application needs.
        // For more information, see: https://flutter.dev/to/review-gradle-config.
        minSdk = flutter.minSdkVersion
        targetSdk = flutter.targetSdkVersion
        versionCode = flutter.versionCode
        versionName = flutter.versionName
        externalNativeBuild {
            cmake {
                // Correct way to pass arguments in Kotlin DSL
                cppFlags("-std=c++23")
                arguments("-DANDROID_STL=c++_shared")
            }
        }
    }

    packaging {
        jniLibs {
            useLegacyPackaging = true 
        }
    }

    buildTypes {
        getByName("release") {
            ndk {
                abiFilters.clear()
                abiFilters.add("arm64-v8a")
            }
            signingConfig = signingConfigs.getByName("debug")
        }
    }

    // This is the specific block causing the 'path' error
    externalNativeBuild {
        cmake {
            path = file("../../../core-vision/CMakeLists.txt")
        }
    }

    // Fix the jvmTarget warning correctly
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    sourceSets {
        getByName("main") {
            jniLibs.srcDirs("../../../deps/opencv/android/OpenCV-android-sdk/sdk/native/libs", 
                           "../../../deps/onnxruntime/android/libs")
        }
    }
}

flutter {
    source = "../.."
}
