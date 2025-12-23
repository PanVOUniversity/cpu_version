"""
Скрипт для запуска инференса обученной модели на изображениях.

Этот модуль предоставляет функциональность для запуска инференса обученной модели
Detectron2 на изображениях. Модуль поддерживает обнаружение объектов, сохранение масок,
визуализацию результатов и обнаружение перекрывающихся детекций.

Основные возможности:
    - Загрузка и настройка модели Detectron2
    - Обработка изображений и детекция объектов
    - Сохранение масок сегментации
    - Визуализация результатов детекции
    - Обнаружение и анализ перекрывающихся детекций
    - Экспорт результатов в JSON формат

Пример использования:
    python inference.py \
        --img-dir images \
        --weights model_final.pth \
        --config-file config/config.yaml \
        --output-dir output \
        --num-classes 1 \
        --thing-classes frame
"""

from __future__ import annotations

# Стандартные библиотеки Python
import argparse
import json
import os
import re
import warnings
from typing import Optional
from pathlib import Path

# Библиотеки для работы с изображениями и массивами
import cv2
import numpy as np

# Попытка импорта setuptools для совместимости с некоторыми установками
try:
    import setuptools  # noqa: F401
except ImportError:
    pass

# Подавление предупреждений от NumPy и PyTorch для более чистого вывода
warnings.filterwarnings("ignore", category=UserWarning, message=".*NumPy array is not writeable.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*__floordiv__ is deprecated.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*torch.meshgrid.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Skip loading parameter.*")
# Подавление FutureWarning о torch.load weights_only (безопасно для наших обученных моделей)
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.load.*weights_only.*")

# Импорт компонентов Detectron2
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.detection_utils import read_image
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode

# Путь к конфигурационному файлу по умолчанию
DEFAULT_CONFIG = "config/config.yaml"


def setup_cfg(args: argparse.Namespace):
    """
    Настраивает конфигурацию Detectron2 для инференса.
    
    Эта функция создает и настраивает объект конфигурации Detectron2 на основе
    аргументов командной строки. Она устанавливает веса модели, количество классов,
    порог уверенности, устройство выполнения (CPU/GPU) и метаданные для визуализации.
    
    Args:
        args: Объект с аргументами командной строки, содержащий:
            - config_file: путь к YAML конфигурационному файлу
            - weights: путь к файлу с весами модели (.pth)
            - num_classes: количество классов для детекции
            - confidence_threshold: минимальный порог уверенности для детекций
            - device: устройство для выполнения ("cpu" или "cuda")
            - thing_classes: список имен классов
    
    Returns:
        CfgNode: Замороженная конфигурация Detectron2, готовая к использованию
    """
    # Создание базовой конфигурации Detectron2
    cfg = get_cfg()
    # Загрузка конфигурации из YAML файла
    cfg.merge_from_file(args.config_file)
    # Установка формата масок в bitmask для совместимости
    cfg.INPUT.MASK_FORMAT = "bitmask"
    
    # Настройка параметров модели
    cfg.MODEL.WEIGHTS = args.weights  # Путь к обученной модели
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.num_classes  # Количество классов
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold  # Порог уверенности
    
    # Установка устройства (CPU или GPU)
    if args.device:
        cfg.MODEL.DEVICE = args.device
    else:
        # Автоматическое определение: используем CPU если CUDA недоступна
        import torch
        if torch.cuda.is_available():
            cfg.MODEL.DEVICE = "cuda"
        else:
            cfg.MODEL.DEVICE = "cpu"
            print("CUDA недоступна, используется CPU")
    
    # Установка метаданных для визуализации
    # Используем уникальное имя датасета для инференса, чтобы избежать конфликтов
    # с встроенными датасетами COCO
    inference_dataset_name = "__inference__"
    cfg.DATASETS.TEST = (inference_dataset_name,)
    
    # Устанавливаем метаданные для визуализации
    # Это безопасно, так как мы используем уникальное имя датасета
    metadata = MetadataCatalog.get(inference_dataset_name)
    # Проверяем, можно ли установить метаданные (они могут быть уже установлены)
    try:
        if not hasattr(metadata, 'thing_classes') or not metadata.thing_classes:
            metadata.thing_classes = args.thing_classes
    except (AttributeError, AssertionError):
        # Если метаданные уже установлены и конфликтуют, просто пропускаем
        # Визуализация будет использовать существующие метаданные
        pass
    
    cfg.freeze()
    return cfg


def iou(box1, box2):
    """
    Вычисляет Intersection over Union (IoU) для двух bounding boxes.
    Использует ту же логику, что и в html_generator.py.
    
    Args:
        box1: Кортеж или список (x1, y1, x2, y2)
        box2: Кортеж или список (x1, y1, x2, y2)
    
    Returns:
        float: IoU значение от 0.0 до 1.0
    """
    # Преобразуем в кортежи для единообразия
    if isinstance(box1, np.ndarray):
        box1 = (float(box1[0]), float(box1[1]), float(box1[2]), float(box1[3]))
    if isinstance(box2, np.ndarray):
        box2 = (float(box2[0]), float(box2[1]), float(box2[2]), float(box2[3]))
    
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    iou_value = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou_value


def mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Вычисляет Intersection over Union (IoU) для двух масок сегментации.

    Args:
        mask1: Boolean/uint8 маска первого объекта
        mask2: Boolean/uint8 маска второго объекта

    Returns:
        float: IoU значение от 0.0 до 1.0
    """
    # Приводим к булевым массивам
    m1 = mask1.astype(bool)
    m2 = mask2.astype(bool)

    intersection = np.logical_and(m1, m2).sum()
    if intersection == 0:
        return 0.0

    union = np.logical_or(m1, m2).sum()
    if union == 0:
        return 0.0

    return float(intersection) / float(union)


def detect_overlaps(instances, iou_threshold: float = 0.0, method: str = "bbox") -> dict:
    """
    Обнаруживает перекрывающиеся детекции используя ту же логику, что и в html_generator.py.
    
    Args:
        instances: Instances объект из Detectron2
        iou_threshold: Порог IoU для определения перекрытия
        method: Метод вычисления IoU:
            - "bbox": по bounding boxes (как в html_generator.py)
            - "mask": по пиксельным маскам сегментации
    
    Returns:
        dict: Словарь с информацией о перекрытиях
    """
    num_instances = len(instances)
    overlaps = {
        'overlaps': [],
        'total_overlaps': 0,
      'instances_info': [],
      'method': method
    }

    method = (method or "bbox").lower()
    if method not in ("bbox", "mask"):
        raise ValueError(f"Неизвестный метод вычисления IoU: {method}. Ожидалось 'bbox' или 'mask'.")
    
    if num_instances < 2:
        return overlaps
    
    boxes = instances.pred_boxes.tensor.numpy()
    scores = instances.scores.numpy()
    masks = instances.pred_masks.numpy()
    
    # Сохраняем информацию о каждом объекте
    for i in range(num_instances):
        bbox = boxes[i]
        mask_area = masks[i].sum()
        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]
        
        overlaps['instances_info'].append({
            'id': i,
            'score': float(scores[i]),
            'bbox': [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
            'bbox_center': [float((bbox[0] + bbox[2]) / 2), float((bbox[1] + bbox[3]) / 2)],
            'bbox_size': [float(bbox_width), float(bbox_height)],
            'mask_area': int(mask_area)
        })
    
    # Проверяем перекрытия используя ту же логику, что и в html_generator.py
    # В html_generator.py проверяется: sum(iou(new_box, existing) for existing in placed_boxes) < 0.1
    # Здесь мы проверяем все пары объектов
    for i in range(num_instances):
        for j in range(i + 1, num_instances):
            box1 = (float(boxes[i][0]), float(boxes[i][1]), float(boxes[i][2]), float(boxes[i][3]))
            box2 = (float(boxes[j][0]), float(boxes[j][1]), float(boxes[j][2]), float(boxes[j][3]))

            if method == "bbox":
                iou_value = iou(box1, box2)
            else:
                iou_value = mask_iou(masks[i], masks[j])

            if iou_value >= iou_threshold:
                overlaps['overlaps'].append({
                    'instance1': i,
                    'instance2': j,
                    'iou': iou_value,
                    'score1': float(scores[i]),
                    'score2': float(scores[j]),
                    'bbox1': list(box1),
                    'bbox2': list(box2)
                })
                overlaps['total_overlaps'] += 1
    
    return overlaps


def print_overlap_info(overlaps: dict, iou_threshold: float = 0.0, image_name: str = None):
    """
    Выводит информацию о перекрытиях с детальным описанием объектов.
    
    Функция форматирует и выводит в консоль подробную информацию о всех
    обнаруженных объектах и их перекрытиях. Информация включает координаты
    bounding boxes, размеры, площади масок и значения IoU для перекрывающихся пар.
    
    Args:
        overlaps: Словарь с информацией о перекрытиях, содержащий:
        
            - total_overlaps: общее количество перекрытий
            - instances_info: список информации о каждом объекте
            - overlaps: список пар перекрывающихся объектов
        
        iou_threshold: Порог IoU, использованный для определения перекрытий
        image_name: Имя обрабатываемого изображения (для ссылок на маски)
    """
    if overlaps['total_overlaps'] == 0:
        print("\n✓ Перекрывающихся детекций не обнаружено")
        return

    method = overlaps.get("method", "bbox")
    print(f"\n⚠ Обнаружены перекрывающиеся детекции (метод: {method}, IoU >= {iou_threshold}): {overlaps['total_overlaps']}")
    
    # Выводим информацию о каждом объекте
    if overlaps['instances_info']:
        print("\n  Информация об объектах:")
        print("    (Каждый объект можно увидеть в сохраненной маске и визуализации)")
        for info in overlaps['instances_info']:
            mask_ref = f" (маска: {Path(image_name).stem if image_name else 'image'}_mask_{info['id']}.png)" if image_name else ""
            print(f"    Объект {info['id']}{mask_ref}:")
            print(f"      - Уверенность (Score): {info['score']:.3f}")
            print(f"      - Bounding Box: [{info['bbox'][0]:.0f}, {info['bbox'][1]:.0f}, {info['bbox'][2]:.0f}, {info['bbox'][3]:.0f}]")
            print(f"        (левый_верхний_x, левый_верхний_y, правый_нижний_x, правый_нижний_y)")
            print(f"      - Размер: {info['bbox_size'][0]:.0f} x {info['bbox_size'][1]:.0f} пикселей")
            print(f"      - Центр: ({info['bbox_center'][0]:.0f}, {info['bbox_center'][1]:.0f})")
            print(f"      - Площадь маски: {info['mask_area']} пикселей")
    
    # Выводим информацию о перекрытиях
    if overlaps['total_overlaps'] > 0:
        print(f"\n  Перекрывающиеся объекты:")
        for overlap in overlaps['overlaps']:
            obj1_info = overlaps['instances_info'][overlap['instance1']]
            obj2_info = overlaps['instances_info'][overlap['instance2']]
            print(f"\n    Перекрытие между объектом {overlap['instance1']} и {overlap['instance2']}:")
            print(f"      IoU: {overlap['iou']:.3f} ({overlap['iou']*100:.1f}%)")
            print(f"      Объект {overlap['instance1']}: Score={overlap['score1']:.3f}, "
                  f"BBox=[{obj1_info['bbox'][0]:.0f}, {obj1_info['bbox'][1]:.0f}, "
                  f"{obj1_info['bbox'][2]:.0f}, {obj1_info['bbox'][3]:.0f}], "
                  f"Размер={obj1_info['bbox_size'][0]:.0f}x{obj1_info['bbox_size'][1]:.0f}, "
                  f"Площадь={obj1_info['mask_area']} px²")
            print(f"      Объект {overlap['instance2']}: Score={overlap['score2']:.3f}, "
                  f"BBox=[{obj2_info['bbox'][0]:.0f}, {obj2_info['bbox'][1]:.0f}, "
                  f"{obj2_info['bbox'][2]:.0f}, {obj2_info['bbox'][3]:.0f}], "
                  f"Размер={obj2_info['bbox_size'][0]:.0f}x{obj2_info['bbox_size'][1]:.0f}, "
                  f"Площадь={obj2_info['mask_area']} px²")


def convert_to_json_serializable(obj):
    """
    Преобразует numpy типы в стандартные Python типы для JSON сериализации.
    
    Рекурсивно обходит структуры данных (словари, списки) и преобразует
    numpy типы (int32, int64, float32, float64, ndarray) в стандартные
    Python типы (int, float, list), которые могут быть сериализованы в JSON.
    
    Args:
        obj: Объект для преобразования (может быть dict, list, numpy тип или другой тип)
    
    Returns:
        Преобразованный объект с Python типами вместо numpy типов
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    else:
        return obj


def save_overlaps_json(overlaps: dict, output_dir: Path, image_name: str, iou_threshold: float = 0.0):
    """
    Сохраняет информацию о перекрытиях в JSON файл.
    
    Создает JSON файл с полной информацией о всех обнаруженных объектах
    и их перекрытиях. Файл сохраняется с именем вида "{image_name}_overlaps.json".
    
    Args:
        overlaps: Словарь с информацией о перекрытиях
        output_dir: Директория для сохранения JSON файла
        image_name: Имя исходного изображения
        iou_threshold: Порог IoU, использованный для определения перекрытий
    
    Returns:
        Path: Путь к сохраненному JSON файлу
    """
    json_data = {
        "image": image_name,
        "iou_threshold": float(iou_threshold),
        "method": overlaps.get("method", "bbox"),
        "total_overlaps": int(overlaps.get('total_overlaps', 0)),
        "instances": convert_to_json_serializable(overlaps.get('instances_info', [])),
        "overlaps": convert_to_json_serializable(overlaps.get('overlaps', []))
    }
    
    json_filename = output_dir / f"{Path(image_name).stem}_overlaps.json"
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    return json_filename


def save_masks(predictions, output_dir: Path, image_name: str, detect_overlapping: bool = True, args=None):
    """
    Сохраняет маски сегментации в отдельные файлы.
    
    Функция извлекает маски из предсказаний модели и сохраняет их в нескольких форматах:
    - Отдельные PNG файлы для каждой маски (черно-белые)
    - Объединенная цветная маска со всеми объектами
    - JSON файл с информацией о перекрытиях (если включено)
    
    Args:
        predictions: Словарь с предсказаниями модели, содержащий ключ "instances"
        output_dir: Директория для сохранения масок
        image_name: Имя исходного изображения
        detect_overlapping: Флаг для включения обнаружения перекрытий
        args: Объект с аргументами командной строки (для получения iou_threshold)
    
    Returns:
        dict или None: Словарь с информацией о перекрытиях или None, если объектов не найдено
    """
    instances = predictions["instances"].to("cpu")
    num_instances = len(instances)
    
    if num_instances == 0:
        print("Не обнаружено объектов на изображении")
        return None
    
    overlaps = None
    # Обнаружение перекрытий
    if detect_overlapping:
        iou_threshold = getattr(args, 'iou_threshold', 0.0) if args else 0.0
        method = getattr(args, 'method', 'bbox') if args else 'bbox'
        if num_instances > 1:
            overlaps = detect_overlaps(instances, iou_threshold=iou_threshold, method=method)
            print_overlap_info(overlaps, iou_threshold=iou_threshold, image_name=image_name)
        else:
            # Создаем пустую структуру перекрытий, если объектов меньше 2
            overlaps = {
                'overlaps': [],
                'total_overlaps': 0,
                'instances_info': []
            }
            # Добавляем информацию об объектах, если есть хотя бы один
            if num_instances == 1:
                boxes = instances.pred_boxes.tensor.numpy()
                scores = instances.scores.numpy()
                masks = instances.pred_masks.numpy()
                bbox = boxes[0]
                mask_area = masks[0].sum()
                overlaps['instances_info'].append({
                    'id': 0,
                    'score': float(scores[0]),
                    'bbox': [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                    'bbox_center': [float((bbox[0] + bbox[2]) / 2), float((bbox[1] + bbox[3]) / 2)],
                    'bbox_size': [float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1])],
                    'mask_area': int(mask_area)
                })
        
        # Сохраняем перекрытия в JSON (даже если их нет)
        save_overlaps_json(overlaps, output_dir, image_name, iou_threshold=iou_threshold)
    
    masks_dir = output_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    # Сохраняем каждую маску отдельно
    for i in range(num_instances):
        mask = instances.pred_masks[i].numpy().astype(np.uint8) * 255
        mask_filename = masks_dir / f"{Path(image_name).stem}_mask_{i}.png"
        cv2.imwrite(str(mask_filename), mask)
    
    # Создаем объединенную маску (цветную)
    height, width = instances.image_size
    combined_mask = np.zeros((height, width, 3), dtype=np.uint8)
    
    for i in range(num_instances):
        mask = instances.pred_masks[i].numpy()
        # Используем простую цветовую схему
        color = np.array([(i * 50) % 255, (i * 100) % 255, (i * 150) % 255], dtype=np.uint8)
        combined_mask[mask > 0] = color
    
    combined_mask_filename = masks_dir / f"{Path(image_name).stem}_combined_mask.png"
    cv2.imwrite(str(combined_mask_filename), combined_mask)
    
    return overlaps


def main():
    """
    Главная функция для запуска инференса на изображениях.
    
    Парсит аргументы командной строки, настраивает модель, обрабатывает
    все изображения в указанной директории и сохраняет результаты.
    """
    # Создание парсера аргументов командной строки
    parser = argparse.ArgumentParser(description="Запуск инференса на изображениях")
    parser.add_argument("--img-dir", required=True, type=str, help="Путь к папке с входными изображениями")
    parser.add_argument("--weights", required=True, type=str, help="Путь к обученной модели (model_final.pth)")
    parser.add_argument("--config-file", default=DEFAULT_CONFIG, type=str, help="Путь к конфигурационному файлу")
    parser.add_argument("--output-dir", default="output/inference", type=str, help="Директория для сохранения результатов")
    parser.add_argument("--num-classes", default=1, type=int, help="Количество классов")
    parser.add_argument("--thing-classes", nargs="+", default=["frame"], type=str, help="Имена классов")
    parser.add_argument("--confidence-threshold", default=0.5, type=float, help="Порог уверенности для детекций")
    parser.add_argument("--device", default=None, type=str, choices=["cpu", "cuda"], help="Устройство для инференса (cpu/cuda). По умолчанию определяется автоматически")
    parser.add_argument("--detect-overlaps", action="store_true", default=True, help="Обнаруживать перекрывающиеся детекции")
    parser.add_argument("--iou-threshold", default=0.0, type=float, help="Порог IoU для определения перекрытия (по умолчанию 0.0, как в html_generator.py)")
    parser.add_argument(
        "--method",
        default="bbox",
        choices=["bbox", "mask"],
        help="Метод вычисления перекрытий: 'bbox' (по умолчанию, по прямоугольникам) или 'mask' (по пиксельным маскам сегментации)",
    )
    parser.add_argument("--html-metadata", type=str, default=None, 
                       help="Путь к JSON файлу с метаданными HTML элементов для сопоставления. Можно использовать шаблон {page_id} для автоматической подстановки номера страницы")
    
    args = parser.parse_args()
    
    # Настройка конфигурации
    cfg = setup_cfg(args)
    
    # Создание предиктора
    predictor = DefaultPredictor(cfg)
    
    # Проверка директории с изображениями
    img_dir = Path(args.img_dir)
    if not img_dir.exists():
        raise FileNotFoundError(f"Директория не найдена: {img_dir}")
    if not img_dir.is_dir():
        raise ValueError(f"Указанный путь не является директорией: {img_dir}")
    
    # Поиск всех изображений в директории
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}
    image_files = [f for f in img_dir.iterdir() 
                   if f.is_file() and f.suffix.lower() in image_extensions]
    
    if not image_files:
        raise ValueError(f"В директории {img_dir} не найдено изображений")
    
    print(f"Найдено изображений: {len(image_files)}")
    
    # Создание директории для результатов
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Обработка каждого изображения
    total_instances = 0
    for image_path in image_files:
        print(f"\nОбработка изображения: {image_path.name}")
        
        # Определяем путь к HTML метаданным для этого изображения
        html_metadata_path = None
        if args.html_metadata:
            html_metadata_template = args.html_metadata
            # Пытаемся извлечь page_id из имени файла (например, page_1.png -> 1)
            page_id_match = re.search(r'page[_\s]*(\d+)', image_path.stem, re.IGNORECASE)
            if page_id_match:
                page_id = page_id_match.group(1)
                html_metadata_path = html_metadata_template.format(page_id=page_id)
            elif '{page_id}' not in html_metadata_template:
                # Если шаблон не используется, используем путь как есть
                html_metadata_path = html_metadata_template
            else:
                print(f"⚠️ Не удалось извлечь page_id из имени файла {image_path.name}")
            
            # Создаем временный args объект с путем к метаданным для этого изображения
            class TempArgs:
                def __init__(self, original_args, html_metadata_path):
                    for attr in dir(original_args):
                        if not attr.startswith('_'):
                            setattr(self, attr, getattr(original_args, attr))
                    self.html_metadata = html_metadata_path
                    self.image = str(image_path)
            
            temp_args = TempArgs(args, html_metadata_path)
        else:
            temp_args = args
            temp_args.image = str(image_path)
        
        # Загрузка изображения
        image = read_image(str(image_path), format="BGR")
        
        # Запуск инференса
        predictions = predictor(image)
        
        # Сохранение масок
        save_masks(predictions, output_dir, image_path.name, detect_overlapping=args.detect_overlaps, args=temp_args)
        
        # Сохранение визуализации
        metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        visualizer = Visualizer(image[:, :, ::-1], metadata=metadata, scale=1.0, instance_mode=ColorMode.IMAGE)
        vis_output = visualizer.draw_instance_predictions(predictions=predictions["instances"].to("cpu"))
        
        vis_filename = output_dir / f"{image_path.stem}_visualization.png"
        vis_output.save(str(vis_filename))
        print(f"Сохранена визуализация: {vis_filename}")
        
        # Вывод статистики для текущего изображения
        num_instances = len(predictions["instances"])
        total_instances += num_instances
        print(f"Обнаружено объектов на изображении: {num_instances}")
        if num_instances > 0:
            scores = predictions["instances"].scores.cpu().numpy()
            print(f"Средняя уверенность: {scores.mean():.3f}")
            print(f"Минимальная уверенность: {scores.min():.3f}")
            print(f"Максимальная уверенность: {scores.max():.3f}")
    
    # Итоговая статистика
    print(f"\n{'='*50}")
    print(f"Обработано изображений: {len(image_files)}")
    print(f"Всего обнаружено объектов: {total_instances}")
    print(f"Результаты сохранены в: {output_dir}")


if __name__ == "__main__":
    main()
