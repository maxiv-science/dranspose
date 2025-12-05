import numpy as np

from dranspose.helpers.incremental import IncrementalBuffer


def test_1d() -> None:
    rng = np.random.default_rng(2)
    n = 10
    data = rng.integers(0, 1000, size=(n,), dtype=np.int32)
    idx = np.arange(n)
    rng.shuffle(idx)
    print("data", data)
    buf = IncrementalBuffer(initial_rows=2, filler_value=-1)
    assert buf.is_empty()
    for e, i in enumerate(idx):
        buf.add_entry(i, float(data[i]))
        print(i, buf._arr)
        print(i, buf[0])
        if e == 0:
            assert buf[0] == -1  # filler_value
    assert not buf.is_empty()
    print(buf.preview())
    out = np.asarray(buf)
    assert out.shape == data.shape
    assert np.allclose(out, data, atol=0)


def test_2d() -> None:
    rng = np.random.default_rng(3)
    rows, cols = 5, 6
    data = rng.integers(0, 256, size=(rows, cols)).astype(np.float32)
    print("data", data)
    idx = np.arange(rows)
    rng.shuffle(idx)
    buf = IncrementalBuffer(initial_rows=2, grow_factor=1.3, filler_value=np.nan)
    for e, i in enumerate(idx):
        buf.add_entry(i, data[i])
        if e == 0:
            assert np.isnan(buf[0, 0])  # filler_value
        print(i, buf.preview())

    out = np.asarray(buf)
    assert buf.shape == data.shape
    assert out.shape == data.shape
    assert buf.dtype == data.dtype
    assert out.dtype == data.dtype
    assert np.allclose(out, data, atol=0)


def test_3d() -> None:
    rng = np.random.default_rng(3)
    rows, x_, y_ = 5, 2, 3
    data = np.arange(rows * x_ * y_, dtype=np.uint8).reshape(rows, x_, y_)
    print("data", data)
    idx = np.arange(rows)
    rng.shuffle(idx)
    buf = IncrementalBuffer(initial_rows=2)
    for e, i in enumerate(idx):
        buf.add_entry(i, data[i])
        print(i, buf.preview())
        if e == 0:
            assert np.all(buf[0] == 0)  # filler_value
    out = np.asarray(buf)
    assert out.shape == data.shape
    assert out.dtype == data.dtype
    assert np.allclose(out, data, atol=0)
