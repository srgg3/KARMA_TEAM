import os
import pydicom
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="pydicom")

class DicomDataLoader:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)

    def get_dcm_value(self, dcm, tag_name, default=np.nan):
        """Безопасное извлечение именно 'значения' из тега."""
        if tag_name in dcm:
            val = dcm[tag_name].value
            # Если тег есть, но он пустой, возвращаем дефолтное значение
            if val is None or str(val).strip() == '':
                return default
            return val
        return default

    def metadata_load(self, dicom_path: Path) -> dict:
        try:
            dcm = pydicom.dcmread(dicom_path, stop_before_pixels=True)

            study_uid = self.get_dcm_value(dcm, "StudyInstanceUID", "Unknown_Study")
            image_uid = self.get_dcm_value(dcm, "SOPInstanceUID", "Unknown_Image")

            # DXA аппараты часто прячут размер пикселя в ImagerPixelSpacing
            pixel_spacing = self.get_dcm_value(dcm, "PixelSpacing", None)
            if pixel_spacing is None:
                pixel_spacing = self.get_dcm_value(dcm, "ImagerPixelSpacing", [np.nan, np.nan])

            # Безопасное разделение на X и Y
            try:
                ps_y, ps_x = float(pixel_spacing[0]), float(pixel_spacing[1])
            except (TypeError, IndexError, ValueError):
                ps_y, ps_x = np.nan, np.nan

            body_part = str(self.get_dcm_value(dcm, "BodyPartExamined", "Unknown"))
            view_position = str(self.get_dcm_value(dcm, "ViewPosition", "Unknown"))
            num_frames = int(self.get_dcm_value(dcm, "NumberOfFrames", 1))

            return {
                "path_to_study": str(dicom_path),
                "study_uid": str(study_uid),
                "image_uid": str(image_uid),
                "pixel_spacing_y": ps_y,
                "pixel_spacing_x": ps_x,
                "body_part": body_part,
                "view_position": view_position,
                "num_frames": num_frames,
                "processing_status": "Success",
                "error": None
            }
        except Exception as e:
            return {
                "path_to_study": str(dicom_path),
                "processing_status": "Failure",
                "error": str(e)
            }

    def pixel_model(self, dicom_path: Path):
        try:
            dcm = pydicom.dcmread(dicom_path)
            img_array = dcm.pixel_array

            img_array = img_array.astype(np.float32)
            img_array = (img_array - np.min(img_array)) / (np.max(img_array) - np.min(img_array) + 1e-8)

            num_frames = int(self.get_dcm_value(dcm, "NumberOfFrames", 1))
            return img_array, num_frames
        except Exception as e:
            print(f"Ошибка чтения пикселей {dicom_path}: {e}")
            return None, 0

    def build_eda(self) -> pd.DataFrame:
        records = []
        for root, _, files in os.walk(self.base_dir):
            for file in files:
                if file.lower().endswith('.dcm'):
                    file_path = Path(root) / file
                    records.append(self.metadata_load(file_path))
        return pd.DataFrame(records)

    def visualize_sample(self, dicom_path: Path):
        img_array, num_frames = self.pixel_model(dicom_path)
        if img_array is None:
            return

        if num_frames > 1 and len(img_array.shape) == 3:
            fig, axes = plt.subplots(1, num_frames, figsize=(15, 5))
            for i in range(num_frames):
                axes[i].imshow(img_array[i], cmap='gray')
                axes[i].set_title(f"Frame {i + 1}")
                axes[i].axis('off')
            plt.show()
        else:
            plt.figure(figsize=(8, 8))
            plt.imshow(img_array, cmap='gray')
            plt.title("DICOM Image")
            plt.axis('off')
            plt.show()

# Запуск
loader = DicomDataLoader(r"/Users/whynot/PycharmProjects/lct/Датасет")
df_metadata = loader.build_eda()
df_metadata.to_csv("dicom_metadata.csv", index=False, encoding="utf-8")
print("Сбор завершен. Датафрейм сохранен в dicom_metadata.csv")

if not df_metadata[df_metadata['processing_status'] == 'Success'].empty:
    sample_path = Path(df_metadata[df_metadata['processing_status'] == 'Success'].iloc[0]['path_to_study'])
    loader.visualize_sample(sample_path)