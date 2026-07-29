from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist

@dataclass
class P2PResult:
    works:list[dist.Work]
    post_ms:float
    desc:str

def batch_p2p(
        ops:list[dist.P2POp],
        *,
        desc:str,
        device: torch.device,
        comm_stream: Optional[torch.cuda.Stream]=None
)-> P2PResult:
    """
     Real wrapper over dist.batch_isend_irecv.

    This mirrors torch.distributed.pipelining.schedules._batch_p2p,
    but keeps timing and CUDA stream handling explicit for BP-free traces.
    """
    if not ops:
        return P2PResult([], 0.0, desc)
    started=time.perf_counter()

    if device.type == "cuda" and comm_stream is not None:
        with torch.cuda.stream(comm_stream):
            works = dist.batch_isend_irecv(ops)
    else:
        works = dist.batch_isend_irecv(ops)
    post_ms = (time.perf_counter() - started) * 1000
    return P2PResult(works, post_ms, desc)

def wait_batch_p2p(
      works:list[dist.Work],
        *,
        desc:str,
)-> float:
    """
     Wait for a batch of P2P works.

    This is intentionally as boring as PyTorch's _wait_batch_p2p:
    the schedule decides when waiting is required.
    """
    if not works:
        return 0.0
    started=time.perf_counter()
    for work in works:
        work.wait()
    return (time.perf_counter() - started) * 1000

def make_isend(tensor: torch.Tensor, dst:int)-> dist.P2POp:
    return dist.P2POp(
       dist.isend,tensor,dst
    )
def make_irecv(tensor: torch.Tensor, src:int)-> dist.P2POp:
    return dist.P2POp(
       dist.irecv,tensor,src
    )