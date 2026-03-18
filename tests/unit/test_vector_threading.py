from src.indices.vector import VectorIndex


class _FakeTorch:
    def __init__(self) -> None:
        self.num_threads = None
        self.interop_threads = None

    def set_num_threads(self, value: int) -> None:
        self.num_threads = value

    def set_num_interop_threads(self, value: int) -> None:
        self.interop_threads = value


def test_vector_index_configures_torch_threads(monkeypatch) -> None:
    fake_torch = _FakeTorch()
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)

    vector = VectorIndex(torch_num_threads=8)
    vector._configure_torch_threads()

    assert fake_torch.num_threads == 8
    assert fake_torch.interop_threads == 2
    assert vector._torch_threads_configured is True
