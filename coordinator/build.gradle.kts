plugins {
    kotlin("jvm")
    application
    id("com.google.protobuf") version "0.9.4"
}

java {
    toolchain {
        languageVersion.set(JavaLanguageVersion.of(17))
    }
}

kotlin {
    jvmToolchain(17)
}

application {
    mainClass.set("com.example.sid_coordinator.CoordinatorMainKt")
}

tasks.register<JavaExec>("runSubmitDemo") {
    group = "application"
    description = "Submit a minimal ForwardChunkRequest to the coordinator."
    classpath = sourceSets["main"].runtimeClasspath
    mainClass.set("com.example.sid_coordinator.SubmitRequestMainKt")
}

tasks.register<JavaExec>("runSubmitPreparedRequest") {
    group = "application"
    description = "Submit a prepared tensor ForwardChunkRequest JSONL record to the coordinator."
    classpath = sourceSets["main"].runtimeClasspath
    mainClass.set("com.example.sid_coordinator.SubmitPreparedRequestMainKt")
    workingDir = rootProject.projectDir
}

tasks.register<JavaExec>("runPreparedExperiment") {
    group = "application"
    description = "Submit a range of prepared tensor ForwardChunkRequest JSONL records and write a CSV summary."
    classpath = sourceSets["main"].runtimeClasspath
    mainClass.set("com.example.sid_coordinator.RunPreparedExperimentMainKt")
    workingDir = rootProject.projectDir
}

tasks.register<JavaExec>("runPreparedPipelineExperiment") {
    group = "application"
    description = "Submit prepared tensor requests with a bounded in-flight window for pipeline-overlap experiments."
    classpath = sourceSets["main"].runtimeClasspath
    mainClass.set("com.example.sid_coordinator.RunPreparedPipelineExperimentMainKt")
    workingDir = rootProject.projectDir
}

tasks.register<JavaExec>("runPreparedStagePipelineExperiment") {
    group = "application"
    description = "Run coordinator-managed per-stage prepared tensor pipeline experiments."
    classpath = sourceSets["main"].runtimeClasspath
    mainClass.set("com.example.sid_coordinator.RunPreparedStagePipelineExperimentMainKt")
    workingDir = rootProject.projectDir
}

sourceSets {
    main {
        proto {
            srcDir("../app/src/main/proto")
        }
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
        create("grpckt") {
            artifact = "io.grpc:protoc-gen-grpc-kotlin:1.4.1:jdk8@jar"
        }
    }
    generateProtoTasks {
        all().forEach { task ->
            task.plugins {
                create("grpc")
                create("grpckt")
            }
        }
    }
}

dependencies {
    implementation(kotlin("stdlib"))
    implementation("com.google.protobuf:protobuf-kotlin:3.25.1")
    implementation("io.grpc:grpc-netty-shaded:1.62.2")
    implementation("io.grpc:grpc-protobuf:1.62.2")
    implementation("io.grpc:grpc-stub:1.62.2")
    implementation("io.grpc:grpc-services:1.62.2")
    implementation("io.grpc:grpc-kotlin-stub:1.4.1")
    implementation("org.apache.tomcat:annotations-api:6.0.53")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.8.1")
    implementation("com.google.code.gson:gson:2.11.0")
    implementation("org.xerial:sqlite-jdbc:3.46.0.0")
    implementation("org.slf4j:slf4j-api:2.0.13")
    runtimeOnly("ch.qos.logback:logback-classic:1.5.6")

    testImplementation(kotlin("test"))
}
