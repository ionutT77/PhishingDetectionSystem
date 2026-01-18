"""Patch a .keras (Keras v3) archive to remove `Lambda` layers.

Why:
- `keras.layers.Lambda` is serialized using Python bytecode in Keras v3.
- Loading that model on a different Python version (e.g., Python 3.13 on Streamlit)
  can crash the interpreter (segfault) during deserialization.

This script replaces the known Lambda layer in this project with a portable custom
layer (`UI/custom_layers.py::ReduceSum`).

Usage:
  python tools/patch_keras_remove_lambda.py \
    --in results_2mil238k_dataset_arhitectureV2/best_model.keras \
    --out results_2mil238k_dataset_arhitectureV2/best_model_portable.keras
"""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path


def patch_config(config: dict) -> tuple[dict, bool]:
    layers = config.get("config", {}).get("layers", [])
    changed = False

    for layer in layers:
        if layer.get("class_name") != "Lambda":
            continue
        if layer.get("config", {}).get("name") != "lambda":
            continue

        # Replace Lambda with portable custom layer. Keep the same name so the graph wiring
        # (keras_history references) remains valid.
        old = layer
        old_cfg = old.get("config", {})
        dtype = old_cfg.get("dtype")
        trainable = old_cfg.get("trainable", True)

        layer.clear()
        layer.update(
            {
                "module": "custom_layers",
                "class_name": "ReduceSum",
                "config": {
                    "name": old_cfg.get("name", "lambda"),
                    "trainable": trainable,
                    "dtype": dtype,
                    "axis": 1,
                    "keepdims": False,
                },
                "registered_name": "PhishingDetection>ReduceSum",
                "build_config": old.get("build_config", {"input_shape": [None, 50, 256]}),
                "name": old.get("name", old_cfg.get("name", "lambda")),
                "inbound_nodes": old.get(
                    "inbound_nodes",
                    [
                        {
                            "args": [
                                {
                                    "class_name": "__keras_tensor__",
                                    "config": {
                                        "shape": [None, 50, 256],
                                        "dtype": "float32",
                                        "keras_history": ["multiply", 0, 0],
                                    },
                                }
                            ],
                            "kwargs": {"mask": None},
                        }
                    ],
                ),
            }
        )

        changed = True

    return config, changed


def patch_keras_archive(input_path: Path, output_path: Path) -> bool:
    with zipfile.ZipFile(input_path, "r") as zin:
        files = zin.namelist()
        if "config.json" not in files:
            raise RuntimeError("This .keras archive does not contain config.json")

        config = json.loads(zin.read("config.json").decode("utf-8"))
        config, changed = patch_config(config)
        if not changed:
            return False

        # Write a new archive with the updated config.json, copying other members as-is.
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zout:
            for name in files:
                if name == "config.json":
                    data = json.dumps(config, ensure_ascii=False).encode("utf-8")
                    zout.writestr(name, data)
                else:
                    zout.writestr(name, zin.read(name))

    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input", required=True, help="Input .keras file")
    parser.add_argument("--out", dest="output", required=True, help="Output .keras file")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    changed = patch_keras_archive(input_path, output_path)
    if not changed:
        print("No matching Lambda layer found; nothing changed.")
    else:
        print(f"Patched model written to: {output_path}")


if __name__ == "__main__":
    main()
