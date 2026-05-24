import com.google.protobuf.ByteString;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import sid.CoordinatingServiceGrpc;
import sid.Sid;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.time.Instant;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public final class SubmitRealRequest {
    private record ModelShape(int hiddenDim, int vocabSize) {}

    private static final Map<String, ModelShape> MODEL_SHAPES = new HashMap<>();

    static {
        MODEL_SHAPES.put("tinyllama", new ModelShape(2048, 32000));
        MODEL_SHAPES.put("phi2", new ModelShape(2560, 51200));
        MODEL_SHAPES.put("smollm2_360m", new ModelShape(960, 49152));
    }

    public static void main(String[] args) {
        String host = args.length > 0 ? args[0] : "127.0.0.1";
        int port = args.length > 1 ? Integer.parseInt(args[1]) : 50051;
        String requestId = args.length > 2 ? args[2] : "demo-" + Instant.now().toEpochMilli();
        String modelPreset = args.length > 3 ? args[3].toLowerCase() : "tinyllama";
        int seqLen = args.length > 4 ? Integer.parseInt(args[4]) : 64;
        int batchSize = args.length > 5 ? Integer.parseInt(args[5]) : 1;
        int chunkIdx = args.length > 6 ? Integer.parseInt(args[6]) : 0;

        ModelShape shape = MODEL_SHAPES.get(modelPreset);
        if (shape == null) {
            throw new IllegalArgumentException("Unknown model preset '" + modelPreset + "'. Supported: " + MODEL_SHAPES.keySet());
        }

        ManagedChannel channel = ManagedChannelBuilder.forAddress(host, port)
            .usePlaintext()
            .build();

        try {
            Sid.ForwardChunkRequest request = Sid.ForwardChunkRequest.newBuilder()
                .setRequestId(requestId)
                .setBatchId(1)
                .setChunkIdx(chunkIdx)
                .setHiddenStates(buildFloatTensor(new int[]{batchSize, seqLen, shape.hiddenDim}, i -> ((i % 97) * 0.01f) - 0.5f))
                .setAttentionMask(buildFloatTensor(new int[]{batchSize, 1, seqLen, seqLen}, i -> 0f))
                .setPositionIds(buildLongTensor(new int[]{batchSize, seqLen}, i -> (long) (i % seqLen)))
                .setLabels(buildLongTensor(new int[]{batchSize, seqLen}, i -> (long) ((i % 16) + 1)))
                .setShiftLogPPrev(chunkIdx == 0
                    ? emptyTensor("float32")
                    : buildFloatTensor(new int[]{batchSize, seqLen, shape.vocabSize}, i -> -((i % 23) * 0.02f)))
                .build();

            Sid.ForwardChunkResponse response = CoordinatingServiceGrpc.newBlockingStub(channel)
                .submitRequest(request);

            System.out.println("requestId=" + requestId);
            System.out.println("modelPreset=" + modelPreset);
            System.out.println("seqLen=" + seqLen);
            System.out.println("batchSize=" + batchSize);
            System.out.println("chunkIdx=" + chunkIdx);
            System.out.println("success=" + response.getSuccess());
            System.out.println("message=" + response.getMessage());
            System.out.println("processedStageId=" + response.getProcessedStageId());
            System.out.println("processedChunkIdx=" + response.getProcessedChunkIdx());
            System.out.println("terminal=" + response.getTerminal());
            System.out.println("outputHiddenBytes=" + response.getOutputHiddenStates().getData().size());
        } finally {
            channel.shutdownNow();
        }
    }

    private interface FloatGenerator {
        float generate(int index);
    }

    private interface LongGenerator {
        long generate(int index);
    }

    private static Sid.TensorData buildFloatTensor(int[] shape, FloatGenerator generator) {
        int elementCount = Arrays.stream(shape).reduce(1, Math::multiplyExact);
        ByteBuffer bytes = ByteBuffer.allocate(elementCount * Float.BYTES).order(ByteOrder.nativeOrder());
        for (int index = 0; index < elementCount; index++) {
            bytes.putFloat(generator.generate(index));
        }
        Sid.TensorData.Builder builder = Sid.TensorData.newBuilder()
            .setData(ByteString.copyFrom(bytes.array()))
            .setDataType("float32");
        for (int dim : shape) {
            builder.addShape(dim);
        }
        return builder.build();
    }

    private static Sid.TensorData buildLongTensor(int[] shape, LongGenerator generator) {
        int elementCount = Arrays.stream(shape).reduce(1, Math::multiplyExact);
        ByteBuffer bytes = ByteBuffer.allocate(elementCount * Long.BYTES).order(ByteOrder.nativeOrder());
        for (int index = 0; index < elementCount; index++) {
            bytes.putLong(generator.generate(index));
        }
        Sid.TensorData.Builder builder = Sid.TensorData.newBuilder()
            .setData(ByteString.copyFrom(bytes.array()))
            .setDataType("int64");
        for (int dim : shape) {
            builder.addShape(dim);
        }
        return builder.build();
    }

    private static Sid.TensorData emptyTensor(String dataType) {
        return Sid.TensorData.newBuilder()
            .setData(ByteString.EMPTY)
            .setDataType(dataType)
            .build();
    }
}
