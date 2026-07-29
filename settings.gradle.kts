pluginManagement {
    repositories {
        maven { url = uri("https://maven.aliyun.com/repository/google") }
        maven {
            url = uri("https://maven.aliyun.com/repository/public")
            content {
                // 🌟 明确告诉 Gradle：阿里云不负责解析 Wire，防止半同步劫持
                excludeGroup("com.squareup.wire")
            }
        }

        google {
            content {
                includeGroupByRegex("com\\.android.*")
                includeGroupByRegex("com\\.google.*")
                includeGroupByRegex("androidx.*")
            }
        }
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        maven { url = uri("https://maven.aliyun.com/repository/google") }
        maven {
            url = uri("https://maven.aliyun.com/repository/public")
            content {
                // 🌟 同样在这里排除，让依赖也能顺利去官方源下载
                excludeGroup("com.squareup.wire")
            }
        }

        google()
        mavenCentral()
    }
}

rootProject.name = "SIDWorker"
include(":app")
include(":coordinator")
