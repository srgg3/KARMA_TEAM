from pathlib import Path
from datetime import datetime
import argparse
import time
import warnings

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
from torchvision import models, transforms

warnings.filterwarnings("ignore", category=UserWarning, module="pydicom")

MODEL_DIR = Path("models")
OUTPUT_COLUMNS = [
    "path_to_study",
    "study_uid",
    "image_uid",
    "anatomical_region",
    "quality_class",
    "violation_type",
    "processing_status",
    "time_of_processing",
]

MODEL_FILES = {
    "spine_quality": MODEL_DIR / "spine_resnet18_best.pt",
    "hip_quality": MODEL_DIR / "hip_resnet18_best.pt",
    "spine_artifacts": MODEL_DIR / "spine_artifacts_resnet18_best.pt",
    "hip_position_rotation": MODEL_DIR / "hip_position_rotation_resnet18_best.pt",
}


def detect_region(dcm):
    """
    Dataset/device-specific rule established during EDA.
    Do NOT treat this as a universal DXA anatomy detector.
    """
    columns = int(getattr(dcm, "Columns", 0) or 0)
    if columns == 300:
        return "spine"
    if columns in (280, 248):
        return "hip"
    return "unknown"


def preprocess(dcm, image_size):
    image = dcm.pixel_array.astype(np.float32)

    if getattr(dcm, "PhotometricInterpretation", "") == "MONOCHROME1":
        image = image.max() - image

    lo, hi = np.percentile(image, [1, 99])
    if hi <= lo:
        lo, hi = float(image.min()), float(image.max())

    image = np.clip(image, lo, hi)
    if hi > lo:
        image = (image - lo) / (hi - lo)
    else:
        image = np.zeros_like(image)

    image = (image * 255).astype(np.uint8)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return transform(image).unsqueeze(0)


def load_model(path, device):
    if not path.exists():
        raise FileNotFoundError(f"Не найдена модель: {path}")

    checkpoint = torch.load(path, map_location=device)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)

    # Поддерживаем все форматы checkpoint, которые использовались
    # в наших train-скриптах.
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        # train_hip.py / train_violations.py
        state_dict = checkpoint["state_dict"]
        image_size = int(checkpoint.get("image_size", 224))
        threshold = float(checkpoint.get("threshold", 0.5))

    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        # первая версия train_spine.py
        state_dict = checkpoint["model_state_dict"]
        image_size = int(checkpoint.get("image_size", 224))
        threshold = float(checkpoint.get("threshold", 0.5))

    else:
        # На случай checkpoint, состоящего только из state_dict.
        state_dict = checkpoint
        image_size = 224
        threshold = 0.5

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return {
        "model": model,
        "image_size": image_size,
        "threshold": threshold,
    }


@torch.no_grad()
def probability(bundle, dcm, device):
    x = preprocess(dcm, bundle["image_size"]).to(device)
    logit = bundle["model"](x).squeeze()
    return float(torch.sigmoid(logit).cpu().item())


def safe_uid(dcm, name, fallback):
    value = getattr(dcm, name, None)
    if value is None or str(value).strip() == "":
        return fallback
    return str(value)


def process_one(path, bundles, device):
    started = time.perf_counter()
    now = datetime.now().astimezone().isoformat(timespec="seconds")

    base = {
        "path_to_study": str(path),
        "study_uid": "",
        "image_uid": "",
        "anatomical_region": "unknown",
        "quality_class": "",
        "violation_type": "",
        "processing_status": "",
        "time_of_processing": now,
    }

    try:
        dcm = pydicom.dcmread(path)

        base["study_uid"] = safe_uid(
            dcm, "StudyInstanceUID", "Unknown_Study"
        )
        base["image_uid"] = safe_uid(
            dcm, "SOPInstanceUID", "Unknown_Image"
        )

        region = detect_region(dcm)
        base["anatomical_region"] = region

        if region == "unknown":
            base["processing_status"] = (
                "Failure: unsupported anatomy/dimensions"
            )
            base["violation_type"] = "not_assessed"
            return base

        if region == "spine":
            q = probability(bundles["spine_quality"], dcm, device)
            v = probability(bundles["spine_artifacts"], dcm, device)

            q_thr = bundles["spine_quality"]["threshold"]
            v_thr = bundles["spine_artifacts"]["threshold"]

            quality_class = int(q >= q_thr)

            if quality_class == 0:
                violation = "no_violation"
            elif v >= v_thr:
                violation = "artifacts"
            else:
                # Quality model sees a violation, but the available
                # violation model cannot assign a supported subtype.
                violation = "quality_violation_unspecified"

        else:
            q = probability(bundles["hip_quality"], dcm, device)
            v = probability(
                bundles["hip_position_rotation"], dcm, device
            )

            q_thr = bundles["hip_quality"]["threshold"]
            v_thr = bundles["hip_position_rotation"]["threshold"]

            quality_class = int(q >= q_thr)

            if quality_class == 0:
                violation = "no_violation"
            elif v >= v_thr:
                violation = "position_rotation"
            else:
                violation = "quality_violation_unspecified"

        base["quality_class"] = quality_class
        base["violation_type"] = violation
        base["processing_status"] = "Success"

    except Exception as e:
        base["processing_status"] = (
            f"Failure: {type(e).__name__}: {e}"
        )
        if not base["violation_type"]:
            base["violation_type"] = "not_assessed"

    # Время конкретного файла оставляем в console, а в обязательной
    # колонке — timestamp обработки.
    elapsed = time.perf_counter() - started
    return base, elapsed


def find_dicoms(input_path):
    input_path = Path(input_path)

    if input_path.is_file():
        return [input_path]

    if not input_path.exists():
        raise FileNotFoundError(
            f"Входной путь не существует: {input_path}"
        )

    # Не полагаемся только на .dcm: в реальных DICOM расширение
    # может отсутствовать. Сначала берём .dcm; если их нет —
    # проверяем все файлы.
    dcm = sorted(
        p for p in input_path.rglob("*")
        if p.is_file() and p.suffix.lower() == ".dcm"
    )

    if dcm:
        return dcm

    return sorted(p for p in input_path.rglob("*") if p.is_file())


def main():
    parser = argparse.ArgumentParser(
        description="DXA quality-control inference"
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="Датасет",
        help="DICOM-файл или папка с DICOM. По умолчанию: Датасет",
    )
    parser.add_argument(
        "--output",
        default="result.csv",
        help="CSV результата. По умолчанию: result.csv",
    )
    args = parser.parse_args()

    print("=" * 72)
    print("DXA QUALITY CONTROL — INFERENCE")
    print("=" * 72)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Устройство:", device)

    print("\nЗагрузка моделей...")
    bundles = {}
    for name, path in MODEL_FILES.items():
        print(f"  {name}: {path}")
        bundles[name] = load_model(path, device)

    files = find_dicoms(args.input)
    print("\nНайдено файлов:", len(files))

    rows = []
    total_started = time.perf_counter()

    for i, path in enumerate(files, 1):
        result = process_one(path, bundles, device)

        if isinstance(result, tuple):
            row, elapsed = result
        else:
            row = result
            elapsed = 0.0

        rows.append(row)

        status = row["processing_status"]
        print(
            f"[{i}/{len(files)}] "
            f"{path.name} | "
            f"{row['anatomical_region']} | "
            f"quality={row['quality_class']} | "
            f"{row['violation_type']} | "
            f"{status} | {elapsed:.2f}s"
        )

    result_df = pd.DataFrame(rows)

    for col in OUTPUT_COLUMNS:
        if col not in result_df.columns:
            result_df[col] = ""

    result_df = result_df[OUTPUT_COLUMNS]
    result_df.to_csv(
        args.output,
        index=False,
        encoding="utf-8-sig",
    )

    total_elapsed = time.perf_counter() - total_started

    print("\n" + "=" * 72)
    print("ГОТОВО")
    print("=" * 72)
    print("Результат:", Path(args.output).resolve())
    print("Строк:", len(result_df))
    print(
        "Success:",
        int(result_df["processing_status"].eq("Success").sum())
    )
    print(
        "Failure:",
        int((~result_df["processing_status"].eq("Success")).sum())
    )
    print(f"Общее время: {total_elapsed:.2f} сек.")

    if len(result_df):
        print("\nAnatomical region:")
        print(
            result_df["anatomical_region"]
            .value_counts(dropna=False)
            .to_string()
        )

        print("\nQuality class:")
        print(
            result_df["quality_class"]
            .value_counts(dropna=False)
            .to_string()
        )

        print("\nViolation type:")
        print(
            result_df["violation_type"]
            .value_counts(dropna=False)
            .to_string()
        )

    print(
        "\nLIMITATION: anatomy detection by Columns and supported "
        "violation subtypes are baseline rules for the supplied dataset."
    )


if __name__ == "__main__":
    main()
