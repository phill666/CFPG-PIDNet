import torch
from thop.profile import register_hooks

def profile_shared_modules(model, sample):
    """Use THOP's counters once per physical module, counting its actual executions.

    STDC exposes the same Conv2d through features and stage aliases. THOP's tree
    walk registers duplicate hooks and later leaves earlier hooks attached. A
    unique-module traversal avoids duplicated MACs and cleans up every hook.
    """
    handles = []
    modules = list(model.modules())
    original_keys = set(model.state_dict())
    try:
        for module in modules:
            module.register_buffer('total_ops', torch.zeros(1, dtype=torch.float64))
            counter = register_hooks.get(type(module))
            if counter is not None:
                handles.append(module.register_forward_hook(counter))
        model(sample)
        return sum((module.total_ops.item() for module in modules))
    finally:
        for handle in handles:
            handle.remove()
        for module in modules:
            module._buffers.pop('total_ops', None)
        assert set(model.state_dict()) == original_keys, 'Profiler altered inference state'

import torch

@torch.inference_mode()
def capture_graph(model, sample):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(30):
            model(sample)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = model(sample)
    graph.replay()
    expected = model(sample)
    torch.testing.assert_close(output, expected, rtol=1e-05, atol=1e-06)
    return (graph, output)

@torch.inference_mode()
def graph_latency(graph):
    for _ in range(100):
        graph.replay()
    torch.cuda.synchronize()
    (start, end) = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
    start.record()
    for _ in range(1000):
        graph.replay()
    end.record()
    end.synchronize()
    ms = start.elapsed_time(end) / 1000
    return dict(Latency_ms=ms, FPS=1000 / ms)
