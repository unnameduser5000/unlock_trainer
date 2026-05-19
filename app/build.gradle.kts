//plugins {
//    alias(libs.plugins.android.application)
//    // 根据上一个问题，如果已经解决了 Kotlin 冲突，保留或注释下面这行
//    alias(libs.plugins.kotlin.android) apply false
//
//    // 🌟 1. 移除原来的 com.google.protobuf，使用 Wire 官方插件
//    id("com.squareup.wire") version "5.5.1"
//}
//
//android {
//    namespace = "com.example.sid_trainer"
//    compileSdk {
//        version = release(36) {
//            minorApiLevel = 1
//        }
//    }
//
//    defaultConfig {
//        applicationId = "com.example.sid_trainer"
//        minSdk = 24
//        targetSdk = 36
//        versionCode = 1
//        versionName = "1.0"
//
//        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
//    }
//
//    buildTypes {
//        release {
//            isMinifyEnabled = false
//            proguardFiles(
//                getDefaultProguardFile("proguard-android-optimize.txt"),
//                "proguard-rules.pro"
//            )
//        }
//    }
//    compileOptions {
//        sourceCompatibility = JavaVersion.VERSION_11
//        targetCompatibility = JavaVersion.VERSION_11
//    }
//    externalNativeBuild {
//        cmake {
//            path = file("src/main/cpp/CMakeLists.txt")
//            version = "3.22.1"
//        }
//    }
//    buildFeatures {
//        viewBinding = true
//    }
//}
//
//// 🌟 2. 删除原有的 protobuf { ... } 整个代码块，替换为 Wire 配置
//wire {
//    kotlin {
//        // 使用 Kotlin 协程 (suspend function) 方式生成 gRPC 代码，非常适合移动端
//        rpcCallStyle = "suspending"
//        // 你的 App 作为 gRPC 的请求端
//        rpcRole = "client"
//    }
//    // 💡 默认情况下，Wire 会自动去 app/src/main/proto/ 寻找你的 .proto 文件
//}
//
//dependencies {
//    implementation(libs.androidx.core.ktx)
//    implementation(libs.androidx.appcompat)
//    implementation(libs.material)
//    implementation(libs.androidx.constraintlayout)
//    testImplementation(libs.junit)
//    androidTestImplementation(libs.androidx.junit)
//    androidTestImplementation(libs.androidx.espresso.core)
//
//    // 🌟 3. 彻底删除原先的 io.grpc:* 和 org.apache.tomcat 依赖
//
//    // 🌟 4. 引入 Wire 核心运行时和专用的 Wire gRPC 客户端
//    implementation("com.squareup.wire:wire-runtime:5.5.1")
//    implementation("com.squareup.wire:wire-grpc-client:5.5.1")
//
//    // Wire gRPC 客户端底层基于 OkHttp，如果你的项目中没有显式引入，建议加上它
//    implementation("com.squareup.okhttp3:okhttp:4.12.0")
//}

plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android) // 降级后，这个插件就能正常用了！
    id("com.google.protobuf") version "0.9.4" // 官方最稳版本
}

android {
    namespace = "com.example.sid_trainer"
    // 🌟 降级到稳定的 API 34，抛弃 36 的预览版特性
    compileSdk = 34

    defaultConfig {
        applicationId = "com.example.sid_trainer"
        minSdk = 24
        targetSdk = 34 // 🌟 同步降级
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
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }
    buildFeatures {
        viewBinding = true
        compose=true
    }
    composeOptions {
        kotlinCompilerExtensionVersion = "1.5.10"
    }
}

protobuf {
    protoc {
        artifact = "com.google.protobuf:protoc:3.25.1"
    }
    plugins {
        create("grpc") {
            artifact = "io.grpc:protoc-gen-grpc-java:1.62.2"
        }
        // 🚨 加上这个 Kotlin 协程代码生成器！
        create("grpckt") {
            artifact = "io.grpc:protoc-gen-grpc-kotlin:1.4.1:jdk8@jar"
        }
    }
    generateProtoTasks {
        all().forEach { task ->
            task.builtins {
                create("java") {
                    option("lite")
                }
            }
            task.plugins {
                create("grpc") {
                    option("lite")
                }
                // 🚨 触发 Kotlin 代码生成！
                create("grpckt") {
                    option("lite")
                }
            }
        }
    }
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.material)
    implementation(libs.androidx.constraintlayout)
    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)

    // 🌟 恢复官方标准的 gRPC 和 Protobuf 核心依赖
    implementation("io.grpc:grpc-okhttp:1.62.2")
    implementation("io.grpc:grpc-protobuf-lite:1.62.2")
    implementation("io.grpc:grpc-stub:1.62.2")
    compileOnly("org.apache.tomcat:annotations-api:6.0.53")
    val composeBom = platform("androidx.compose:compose-bom:2024.02.00")
    implementation(composeBom)
    androidTestImplementation(composeBom)

    implementation("androidx.activity:activity-compose:1.8.2")
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-graphics")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.foundation:foundation")

    debugImplementation("androidx.compose.ui:ui-tooling")
    debugImplementation("androidx.compose.ui:ui-test-manifest")

    implementation("io.grpc:grpc-kotlin-stub:1.4.1")
    implementation("org.pytorch:executorch-android:1.2.0")
}
