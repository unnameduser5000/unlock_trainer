package org.pytorch.executorch;

import java.nio.Buffer;

/** Package-local bridge for optimizers that need in-place tensor updates. */
public final class TensorBufferAccess {
  private TensorBufferAccess() {}

  public static Buffer rawDataBuffer(Tensor tensor) {
    return tensor.getRawDataBuffer();
  }
}
