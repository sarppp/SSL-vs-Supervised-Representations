import os
import pickle
from typing import Iterable, List, Union, Any

try:
    import numpy as np  # optional, for legacy formats
except Exception:
    np = None

FILE_A = "/root/dino_cnn/clean_dataset.pkl"
FILE_B = "/root/clean_dataset.pkl"


def _as_str(x: Any) -> Union[str, None]:
    if isinstance(x, str):
        return x
    if isinstance(x, bytes):
        try:
            return x.decode("utf-8", errors="ignore")
        except Exception:
            return None
    return None


def _flatten(iterable: Iterable[Any]) -> Iterable[Any]:
    for x in iterable:
        if isinstance(x, (list, tuple)):
            yield from _flatten(x)
        elif np is not None and isinstance(x, np.ndarray):
            yield from _flatten(x.tolist())
        else:
            yield x


def _extract_paths(obj: Union[dict, list, tuple, Any]) -> List[str]:
    """Extract list of image paths from various pickle formats.
    Handles:
      - dict with 'paths'
      - list/tuple/ndarray of strings
      - list/tuple of (path, label)
      - dicts with common keys: 'image_paths', 'images', 'files', 'X'
      - deep/nested structures containing strings/bytes
    """
    paths: List[str] = []

    # numpy container to list
    if np is not None and isinstance(obj, np.ndarray):
        obj = obj.tolist()

    if isinstance(obj, dict):
        # Primary new format
        if "paths" in obj and isinstance(obj["paths"], (list, tuple)):
            raw = obj["paths"]
            data_dir = obj.get("data_dir")
            use_rel = obj.get("use_relative_paths", False)
            for p in raw:
                s = _as_str(p)
                if s is None:
                    continue
                if not use_rel and data_dir:
                    try:
                        s = os.path.relpath(s, data_dir)
                    except Exception:
                        pass
                paths.append(s)
        else:
            # Common legacy keys
            for key in ("image_paths", "images", "files", "X"):
                v = obj.get(key)
                if v is None:
                    continue
                if np is not None and isinstance(v, np.ndarray):
                    v = v.tolist()
                if isinstance(v, (list, tuple)) and v:
                    if isinstance(v[0], (list, tuple)) and v[0]:
                        s = _as_str(v[0][0])
                        if s is not None:
                            for item in v:
                                if isinstance(item, (list, tuple)) and item:
                                    s = _as_str(item[0])
                                    if s is not None:
                                        paths.append(s)
                    else:
                        for item in v:
                            s = _as_str(item)
                            if s is not None:
                                paths.append(s)
            # Deep scan fallback
            if not paths:
                for x in _flatten(obj.values()):
                    s = _as_str(x)
                    if s is not None:
                        paths.append(s)

    elif isinstance(obj, (list, tuple)):
        if obj:
            # list of pairs
            if isinstance(obj[0], (list, tuple)) and obj[0]:
                for item in obj:
                    if isinstance(item, (list, tuple)) and item:
                        s = _as_str(item[0])
                        if s is not None:
                            paths.append(s)
            else:
                for item in obj:
                    s = _as_str(item)
                    if s is not None:
                        paths.append(s)
    else:
        # Deep scan for any strings/bytes
        for x in _flatten([obj]):
            s = _as_str(x)
            if s is not None:
                paths.append(s)

    return paths


def _to_class_filename(p: str) -> str:
    parts = [q for q in p.replace("\\", "/").split("/") if q]
    if len(parts) >= 2:
        return f"{parts[-2]}/{parts[-1]}"
    return parts[-1] if parts else p


def _describe(obj: Any, name: str) -> None:
    def head(seq, n=3):
        try:
            return list(seq)[:n]
        except Exception:
            return []
    print(f"\n{name} structure summary:")
    print(f"  type: {type(obj)}")
    if isinstance(obj, dict):
        print(f"  keys: {list(obj.keys())[:8]}")
        for k in list(obj.keys())[:4]:
            v = obj[k]
            print(f"    - {k}: type {type(v)}")
    elif isinstance(obj, (list, tuple)):
        print(f"  length: {len(obj)}")
        print(f"  first items types: {[type(x) for x in head(obj)]}")
    elif np is not None and isinstance(obj, np.ndarray):
        print(f"  ndarray shape: {obj.shape}, dtype: {obj.dtype}")


def compare_internal_names(a_path: str, b_path: str) -> None:
    if not os.path.exists(a_path) or not os.path.exists(b_path):
        print("Error: one or both files do not exist")
        return

    with open(a_path, "rb") as fa:
        a_obj = pickle.load(fa)
    with open(b_path, "rb") as fb:
        b_obj = pickle.load(fb)

    a_paths = _extract_paths(a_obj)
    b_paths = _extract_paths(b_obj)

    if not a_paths:
        _describe(a_obj, "A")
    if not b_paths:
        _describe(b_obj, "B")

    a_names = {_to_class_filename(p) for p in a_paths}
    b_names = {_to_class_filename(p) for p in b_paths}

    inter = a_names & b_names
    only_a = a_names - b_names
    only_b = b_names - a_names

    print(f"A source: {a_path}")
    print(f"B source: {b_path}")
    print(f"A count: {len(a_names)} unique names")
    print(f"B count: {len(b_names)} unique names")
    print(f"Overlap: {len(inter)}")
    print(f"Only in A: {len(only_a)}")
    print(f"Only in B: {len(only_b)}")

    if len(only_a) or len(only_b):
        sample_a = sorted(list(only_a))[:10]
        sample_b = sorted(list(only_b))[:10]
        if sample_a:
            print("\nExamples only in A:")
            for s in sample_a:
                print(f"  - {s}")
        if sample_b:
            print("\nExamples only in B:")
            for s in sample_b:
                print(f"  - {s}")
    else:
        print("\n Internal name sets are IDENTICAL")


if __name__ == "__main__":
    compare_internal_names(FILE_A, FILE_B)
