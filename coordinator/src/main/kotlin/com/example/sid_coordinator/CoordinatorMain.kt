package com.example.sid_coordinator

import org.slf4j.LoggerFactory

fun main(args: Array<String>) {
    val logger = LoggerFactory.getLogger("CoordinatorMain")
    val configPath = parseConfigPath(args)
    val resolvedConfigPath = CoordinatorConfigLoader.resolveConfigPath(configPath)
    val config = CoordinatorConfigLoader.load(resolvedConfigPath)
    val runtime = CoordinatorRuntime(resolvedConfigPath, config)

    Runtime.getRuntime().addShutdownHook(
        Thread {
            runtime.stop()
        }
    )

    logger.info(
        "Loaded pipeline={} stages={}",
        config.pipelineName,
        config.stages.joinToString { "${it.stageId}:${it.deviceId}" }
    )

    runtime.start()
    runtime.awaitTermination()
}

private fun parseConfigPath(args: Array<String>): String? {
    val index = args.indexOf("--config")
    if (index == -1) {
        return null
    }
    require(index + 1 < args.size) { "--config requires a file path" }
    return args[index + 1]
}
