package com.example.sid_coordinator

import com.google.gson.Gson
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
import java.security.MessageDigest

object CoordinatorConfigLoader {
    private val gson = Gson()

    fun load(configPath: String?): CoordinatorConfig {
        return load(resolveConfigPath(configPath))
    }

    fun load(path: Path): CoordinatorConfig {
        val json = Files.readString(requireReadable(path))
        val parsed = gson.fromJson(json, CoordinatorConfig::class.java)
        val configDir = requireReadable(path).toAbsolutePath().normalize().parent
        val normalizedStages = parsed.stages.map { stage ->
            if (stage.modelArtifactPath.isBlank()) {
                stage
            } else {
                val artifactPath = configDir.resolve(stage.modelArtifactPath).normalize()
                require(Files.exists(artifactPath)) {
                    "Model artifact does not exist for stage ${stage.stageId}: $artifactPath"
                }
                require(Files.isReadable(artifactPath)) {
                    "Model artifact is not readable for stage ${stage.stageId}: $artifactPath"
                }
                val sha256 = sha256Hex(artifactPath)
                val bytes = Files.size(artifactPath)
                val downloadUrl = parsed.artifactBaseUrl
                    .trimEnd('/')
                    .takeIf { it.isNotBlank() }
                    ?.plus("/artifacts/stages/${stage.stageId}/model")
                    .orEmpty()
                stage.copy(
                    modelArtifactPath = artifactPath.toString(),
                    modelDownloadUrl = downloadUrl,
                    modelSha256 = sha256,
                    modelBytes = bytes
                )
            }
        }
        return parsed.copy(stages = normalizedStages).validated()
    }

    fun resolveConfigPath(configPath: String?): Path {
        if (!configPath.isNullOrBlank()) {
            return requireReadable(Paths.get(configPath))
        }

        val candidates = listOf(
            Paths.get("coordinator", "config", "pipeline.json"),
            Paths.get("config", "pipeline.json")
        )
        candidates.firstOrNull(Files::exists)?.let { candidate ->
            return requireReadable(candidate)
        }

        error(
            "Coordinator config not found. Tried: ${candidates.joinToString()} . " +
                "Pass --config <path> or create coordinator/config/pipeline.json"
        )
    }

    private fun requireReadable(path: Path): Path {
        require(Files.exists(path)) { "Config file does not exist: $path" }
        require(Files.isReadable(path)) { "Config file is not readable: $path" }
        return path
    }

    private fun sha256Hex(path: Path): String {
        val digest = MessageDigest.getInstance("SHA-256")
        Files.newInputStream(path).use { input ->
            val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
            while (true) {
                val read = input.read(buffer)
                if (read < 0) {
                    break
                }
                if (read > 0) {
                    digest.update(buffer, 0, read)
                }
            }
        }
        return digest.digest().joinToString("") { byte ->
            "%02x".format(byte)
        }
    }
}
