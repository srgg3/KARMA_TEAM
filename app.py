#запустить проект streamlit run app.py
import time
import tempfile
from pathlib import Path

import streamlit as st
import pandas as pd
import torch

# Импортируем классы и константы из вашего основного скрипта
from double_bubble import DXAPipeline, load_pytorch_model, MODEL_FILES, OUTPUT_COLUMNS

# Настройка страницы (должна быть первой командой Streamlit)
st.set_page_config(page_title="DXA Анализ", layout="wide")
st.title("Анализ снимков DXA (DEXA)")


# Кэшируем загрузку моделей, чтобы не ждать при каждом запуске анализа
@st.cache_resource
def get_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundles = {}
    for name, path in MODEL_FILES.items():
        model_bundle = load_pytorch_model(path, device)
        if model_bundle:
            bundles[name] = model_bundle
    return device, bundles


device, bundles = get_models()

# Загрузчики файлов (подсказываем пользователю про папки)
st.markdown("💡 **Подсказка:** Вы можете выделить файлы или *перетащить целую папку* прямо в область ниже.")
uploaded_dicoms = st.file_uploader(
    "Загрузите файлы DICOM или перетащите папку",
    accept_multiple_files=True
)
uploaded_labels = st.file_uploader(
    "Загрузите файл разметки (Excel)",
    type=["xlsx", "xls"]
)

if st.button("Запустить анализ"):
    if not uploaded_dicoms:
        st.warning("Пожалуйста, загрузите хотя бы один файл или папку со снимками.")
    else:
        # 1. Чтение разметки
        labels_df = None
        if uploaded_labels is not None:
            labels_df = pd.read_excel(uploaded_labels)
            st.info(f"Загружена разметка: {len(labels_df)} записей.")

        # 2. Инициализация пайплайна
        pipeline = DXAPipeline(device, bundles)

        # 3. Создаем временную папку
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmp_dir = Path(tmpdirname)
            dicom_paths = []
            original_names = []

            for i, uploaded_file in enumerate(uploaded_dicoms):
                # Фильтруем скрытые файлы (Mac/Windows) и табличные форматы,
                # которые могли случайно попасть при загрузке папки
                filename = uploaded_file.name
                if filename.startswith('.') or filename.lower().endswith(('.xlsx', '.csv', '.py', '.txt')):
                    continue

                # Добавляем индекс к имени, чтобы файлы из разных подпапок
                # с одинаковым названием (например, 1.dcm) не перезаписали друг друга
                safe_name = f"{i}_{filename}"
                file_path = tmp_dir / safe_name

                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                dicom_paths.append(file_path)
                # Сохраняем оригинальное имя для красивого вывода в таблице
                original_names.append(filename)

            if not dicom_paths:
                st.error("В загруженной папке не найдено подходящих DICOM файлов.")
                st.stop()

            # Элементы интерфейса для отображения прогресса
            progress_bar = st.progress(0)
            status_text = st.empty()

            results = []
            total_start = time.perf_counter()

            # 4. Обработка файлов
            for i, path in enumerate(dicom_paths):
                original_name = original_names[i]
                status_text.text(f"Обработка файла {i + 1} из {len(dicom_paths)}: {original_name}")

                # Запуск оригинальной функции
                row = pipeline.process_file(path, labels_df)

                # Меняем путь во временной папке на оригинальное имя файла для чистоты отчета
                row["path_to_study"] = original_name
                results.append(row)

                # Обновляем прогресс-бар
                progress_bar.progress((i + 1) / len(dicom_paths))

            # 5. Формирование итогового DataFrame
            result_df = pd.DataFrame(results)
            for col in OUTPUT_COLUMNS:
                if col not in result_df.columns:
                    result_df[col] = ""
            result_df = result_df[OUTPUT_COLUMNS]

            total_time = time.perf_counter() - total_start

            # Очищаем текст статуса и выводим успех
            status_text.empty()
            st.success(f"Анализ успешно завершен за {total_time:.2f} сек.!")

            # Показываем таблицу на экране
            st.dataframe(result_df)

            # 6. Подготовка файла для скачивания
            csv_data = result_df.to_csv(index=False, encoding="utf-8-sig", sep=";").encode("utf-8-sig")

            st.download_button(
                label="Скачать результаты CSV",
                data=csv_data,
                file_name="final_submission.csv",
                mime="text/csv"
            )