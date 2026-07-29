#include <android/log.h>
#include <jni.h>

#include <fstream>
#include <string>
#include <vector>

namespace {

constexpr const char* kLogTag = "sid_trainer_native";

std::vector<jbyte> ReadByteArray(JNIEnv* env, jbyteArray array) {
    if (array == nullptr) {
        return {};
    }

    const jsize length = env->GetArrayLength(array);
    std::vector<jbyte> bytes(static_cast<size_t>(length));
    if (length > 0) {
        env->GetByteArrayRegion(array, 0, length, bytes.data());
    }
    return bytes;
}

bool FileExists(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    return file.good();
}

jbyteArray ToJByteArray(JNIEnv* env, const std::vector<jbyte>& bytes) {
    jbyteArray result = env->NewByteArray(static_cast<jsize>(bytes.size()));
    if (result == nullptr) {
        return nullptr;
    }

    if (!bytes.empty()) {
        env->SetByteArrayRegion(result, 0, static_cast<jsize>(bytes.size()), bytes.data());
    }
    return result;
}

}  // namespace

extern "C" JNIEXPORT jbyteArray JNICALL
Java_com_example_sid_1trainer_NativeShardRunner_runShard(
        JNIEnv* env,
        jobject /* this */,
        jstring model_path,
        jbyteArray hidden_states,
        jbyteArray attention_mask,
        jbyteArray position_ids,
        jbyteArray labels,
        jbyteArray shift_log_p_prev) {
    const char* raw_model_path = env->GetStringUTFChars(model_path, nullptr);
    std::string model_path_string = raw_model_path == nullptr ? "" : raw_model_path;
    if (raw_model_path != nullptr) {
        env->ReleaseStringUTFChars(model_path, raw_model_path);
    }

    if (!FileExists(model_path_string)) {
        __android_log_print(ANDROID_LOG_ERROR, kLogTag,
                            "Model shard not found: %s", model_path_string.c_str());
        return env->NewByteArray(0);
    }

    auto hidden_state_bytes = ReadByteArray(env, hidden_states);
    (void) ReadByteArray(env, attention_mask);
    (void) ReadByteArray(env, position_ids);
    (void) ReadByteArray(env, labels);
    (void) ReadByteArray(env, shift_log_p_prev);

    __android_log_print(ANDROID_LOG_INFO, kLogTag,
                        "Shard invoked with %zu hidden-state bytes using model %s",
                        hidden_state_bytes.size(), model_path_string.c_str());

    return ToJByteArray(env, hidden_state_bytes);
}
