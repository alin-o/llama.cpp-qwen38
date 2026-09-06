#!/usr/bin/env python3
import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path


SCALAR_FORMATS = {
    0: "B",
    1: "b",
    2: "H",
    3: "h",
    4: "I",
    5: "i",
    6: "f",
    7: "?",
    10: "Q",
    11: "q",
    12: "d",
}


class GGUFError(RuntimeError):
    pass


def read_exact(handle, size):
    data = handle.read(size)
    if len(data) != size:
        raise GGUFError("unexpected end of file")
    return data


def read_string(handle):
    size = struct.unpack("<Q", read_exact(handle, 8))[0]
    return read_exact(handle, size).decode("utf-8", errors="replace")


def read_value(handle, value_type, keep=True):
    if value_type in SCALAR_FORMATS:
        value = struct.unpack("<" + SCALAR_FORMATS[value_type], read_exact(handle, struct.calcsize(SCALAR_FORMATS[value_type])))[0]
        return value if keep else None
    if value_type == 8:
        value = read_string(handle)
        return value if keep else None
    if value_type == 9:
        element_type = struct.unpack("<I", read_exact(handle, 4))[0]
        count = struct.unpack("<Q", read_exact(handle, 8))[0]
        if keep and count <= 32:
            return [read_value(handle, element_type, True) for _ in range(count)]
        for _ in range(count):
            read_value(handle, element_type, False)
        return {"element_type": element_type, "count": count} if keep else None
    raise GGUFError(f"unsupported metadata type {value_type}")


def inspect(path):
    metadata = {}
    spans = {}
    tensor_names = []
    nextn_tensors = []
    with path.open("rb") as handle:
        if read_exact(handle, 4) != b"GGUF":
            raise GGUFError("invalid GGUF magic")
        version = struct.unpack("<I", read_exact(handle, 4))[0]
        tensor_count = struct.unpack("<Q", read_exact(handle, 8))[0]
        kv_count = struct.unpack("<Q", read_exact(handle, 8))[0]
        for _ in range(kv_count):
            key = read_string(handle)
            value_type = struct.unpack("<I", read_exact(handle, 4))[0]
            start = handle.tell()
            keep = key in {
                "general.architecture",
                "general.name",
                "qwen35.context_length",
                "qwen35.nextn_predict_layers",
            }
            metadata[key] = read_value(handle, value_type, keep)
            spans[key] = (start, handle.tell())
        for _ in range(tensor_count):
            name = read_string(handle)
            n_dims = struct.unpack("<I", read_exact(handle, 4))[0]
            read_exact(handle, 8 * n_dims)
            tensor_type = struct.unpack("<I", read_exact(handle, 4))[0]
            read_exact(handle, 8)
            tensor_names.append(name)
            if "nextn" in name or "mtp" in name:
                nextn_tensors.append({"name": name, "type": tensor_type})

        tokenizer_hasher = hashlib.sha256()
        tokenizer_keys = sorted(key for key in spans if key.startswith("tokenizer.") and "chat_template" not in key)
        for key in tokenizer_keys:
            start, end = spans[key]
            handle.seek(start)
            tokenizer_hasher.update(key.encode())
            tokenizer_hasher.update(read_exact(handle, end - start))

    return {
        "path": str(path),
        "size": path.stat().st_size,
        "version": version,
        "tensor_count": tensor_count,
        "architecture": metadata.get("general.architecture"),
        "name": metadata.get("general.name"),
        "context_length": metadata.get("qwen35.context_length"),
        "nextn_predict_layers": metadata.get("qwen35.nextn_predict_layers"),
        "nextn_tensor_count": sum("nextn" in name or "mtp" in name for name in tensor_names),
        "nextn_tensors": nextn_tensors,
        "tokenizer_key_count": len(tokenizer_keys),
        "tokenizer_fingerprint": tokenizer_hasher.hexdigest() if tokenizer_keys else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--mtp", type=Path, required=True)
    parser.add_argument("--mmproj", type=Path, required=True)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    errors = []
    try:
        target = inspect(args.target)
        mtp = inspect(args.mtp)
        mmproj = inspect(args.mmproj)
    except (OSError, GGUFError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if target["architecture"] != "qwen35":
        errors.append(f"target architecture is {target['architecture']!r}, expected 'qwen35'")
    if target["context_length"] != 262144:
        errors.append(f"target context is {target['context_length']!r}, expected 262144")
    if not str(mtp["architecture"]).startswith("qwen35"):
        errors.append(f"MTP architecture is {mtp['architecture']!r}, expected qwen35 family")
    if mtp["tensor_count"] == 0:
        errors.append("MTP file contains no tensors")
    if mmproj["tensor_count"] == 0:
        errors.append("mmproj file contains no tensors")
    if target["tokenizer_fingerprint"] and mtp["tokenizer_fingerprint"] and target["tokenizer_fingerprint"] != mtp["tokenizer_fingerprint"]:
        errors.append("target and MTP tokenizer metadata do not match")

    report = {"target": target, "mtp": mtp, "mmproj": mmproj, "errors": errors}
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(rendered + "\n")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
