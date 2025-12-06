"""
Скрипт для запуска инференса обученной модели на изображении.

Пример использования:
    python train/inference.py \
        --image path/to/image.png \
        --weights output/my_target/model_final.pth \
        --config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
        --output-dir output/inference \
        --num-classes 1 \
        --thing-classes frame
"""

from __future__ import annotations

import argparse
import os
import warnings
import cv2
import numpy as np
from pathlib import Path

try:
    import setuptools  # noqa: F401
except ImportError:
    pass

warnings.filterwarnings("ignore", category=UserWarning, message=".*NumPy array is not writeable.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*__floordiv__ is deprecated.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*torch.meshgrid.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Skip loading parameter.*")

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.detection_utils import read_image
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode

DEFAULT_CONFIG = "configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"


def setup_cfg(args: argparse.Namespace):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.INPUT.MASK_FORMAT = "bitmask"
    
    cfg.MODEL.WEIGHTS = args.weights
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.num_classes
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    
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


def calculate_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Вычисляет Intersection over Union (IoU) для двух масок."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def calculate_bbox_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """Вычисляет IoU для двух bounding boxes в формате [x1, y1, x2, y2]."""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    return intersection / union


def detect_overlaps(instances, iou_threshold: float = 0.3) -> dict:
    """
    Обнаруживает перекрывающиеся детекции.
    
    Args:
        instances: Instances объект из Detectron2
        iou_threshold: Порог IoU для определения перекрытия (по умолчанию 0.3)
    
    Returns:
        dict: Словарь с информацией о перекрытиях
    """
    num_instances = len(instances)
    overlaps = {
        'mask_overlaps': [],
        'bbox_overlaps': [],
        'total_mask_overlaps': 0,
        'total_bbox_overlaps': 0,
        'instances_info': []  # Информация о каждом объекте
    }
    
    if num_instances < 2:
        return overlaps
    
    masks = instances.pred_masks.numpy()
    boxes = instances.pred_boxes.tensor.numpy()
    scores = instances.scores.numpy()
    
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
    
    # Проверяем перекрытия масок
    for i in range(num_instances):
        for j in range(i + 1, num_instances):
            iou_mask = calculate_iou(masks[i], masks[j])
            iou_bbox = calculate_bbox_iou(boxes[i], boxes[j])
            
            if iou_mask >= iou_threshold:
                overlaps['mask_overlaps'].append({
                    'instance1': i,
                    'instance2': j,
                    'iou': iou_mask,
                    'score1': float(scores[i]),
                    'score2': float(scores[j])
                })
                overlaps['total_mask_overlaps'] += 1
            
            if iou_bbox >= iou_threshold:
                overlaps['bbox_overlaps'].append({
                    'instance1': i,
                    'instance2': j,
                    'iou': iou_bbox,
                    'score1': float(scores[i]),
                    'score2': float(scores[j])
                })
                overlaps['total_bbox_overlaps'] += 1
    
    return overlaps


def print_overlap_info(overlaps: dict, iou_threshold: float = 0.3, image_name: str = None):
    """Выводит информацию о перекрытиях с детальным описанием объектов."""
    if overlaps['total_mask_overlaps'] == 0 and overlaps['total_bbox_overlaps'] == 0:
        print("\n✓ Перекрывающихся детекций не обнаружено")
        return
    
    print("\n⚠ Обнаружены перекрывающиеся детекции:")
    
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
    
    if overlaps['total_mask_overlaps'] > 0:
        print(f"\n  Перекрывающиеся маски (IoU >= {iou_threshold}): {overlaps['total_mask_overlaps']}")
        for overlap in overlaps['mask_overlaps']:
            obj1_info = overlaps['instances_info'][overlap['instance1']]
            obj2_info = overlaps['instances_info'][overlap['instance2']]
            print(f"\n    Перекрытие между объектом {overlap['instance1']} и {overlap['instance2']}:")
            print(f"      IoU масок: {overlap['iou']:.3f}")
            print(f"      Объект {overlap['instance1']}: Score={overlap['score1']:.3f}, "
                  f"BBox=[{obj1_info['bbox'][0]:.0f}, {obj1_info['bbox'][1]:.0f}, "
                  f"{obj1_info['bbox'][2]:.0f}, {obj1_info['bbox'][3]:.0f}], "
                  f"Площадь={obj1_info['mask_area']} px²")
            print(f"      Объект {overlap['instance2']}: Score={overlap['score2']:.3f}, "
                  f"BBox=[{obj2_info['bbox'][0]:.0f}, {obj2_info['bbox'][1]:.0f}, "
                  f"{obj2_info['bbox'][2]:.0f}, {obj2_info['bbox'][3]:.0f}], "
                  f"Площадь={obj2_info['mask_area']} px²")
    
    if overlaps['total_bbox_overlaps'] > 0:
        print(f"\n  Перекрывающиеся bounding boxes (IoU >= {iou_threshold}): {overlaps['total_bbox_overlaps']}")
        for overlap in overlaps['bbox_overlaps']:
            obj1_info = overlaps['instances_info'][overlap['instance1']]
            obj2_info = overlaps['instances_info'][overlap['instance2']]
            print(f"\n    Перекрытие между объектом {overlap['instance1']} и {overlap['instance2']}:")
            print(f"      IoU bounding boxes: {overlap['iou']:.3f}")
            print(f"      Объект {overlap['instance1']}: Score={overlap['score1']:.3f}, "
                  f"BBox=[{obj1_info['bbox'][0]:.0f}, {obj1_info['bbox'][1]:.0f}, "
                  f"{obj1_info['bbox'][2]:.0f}, {obj1_info['bbox'][3]:.0f}], "
                  f"Размер={obj1_info['bbox_size'][0]:.0f}x{obj1_info['bbox_size'][1]:.0f}")
            print(f"      Объект {overlap['instance2']}: Score={overlap['score2']:.3f}, "
                  f"BBox=[{obj2_info['bbox'][0]:.0f}, {obj2_info['bbox'][1]:.0f}, "
                  f"{obj2_info['bbox'][2]:.0f}, {obj2_info['bbox'][3]:.0f}], "
                  f"Размер={obj2_info['bbox_size'][0]:.0f}x{obj2_info['bbox_size'][1]:.0f}")


def save_masks(predictions, output_dir: Path, image_name: str, detect_overlapping: bool = True, args=None):
    """Сохраняет маски в отдельные файлы."""
    instances = predictions["instances"].to("cpu")
    num_instances = len(instances)
    
    if num_instances == 0:
        print("Не обнаружено объектов на изображении")
        return
    
    # Обнаружение перекрытий
    if detect_overlapping and num_instances > 1:
        iou_threshold = getattr(args, 'iou_threshold', 0.3) if args else 0.3
        overlaps = detect_overlaps(instances, iou_threshold=iou_threshold)
        print_overlap_info(overlaps, iou_threshold=iou_threshold, image_name=image_name)
    
    masks_dir = output_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    # Сохраняем каждую маску отдельно
    for i in range(num_instances):
        mask = instances.pred_masks[i].numpy().astype(np.uint8) * 255
        mask_filename = masks_dir / f"{Path(image_name).stem}_mask_{i}.png"
        cv2.imwrite(str(mask_filename), mask)
        print(f"Сохранена маска: {mask_filename}")
    
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
    print(f"Сохранена объединенная маска: {combined_mask_filename}")


def main():
    parser = argparse.ArgumentParser(description="Запуск инференса на изображении")
    parser.add_argument("--image", required=True, type=str, help="Путь к входному изображению")
    parser.add_argument("--weights", required=True, type=str, help="Путь к обученной модели (model_final.pth)")
    parser.add_argument("--config-file", default=DEFAULT_CONFIG, type=str, help="Путь к конфигурационному файлу")
    parser.add_argument("--output-dir", default="output/inference", type=str, help="Директория для сохранения результатов")
    parser.add_argument("--num-classes", default=1, type=int, help="Количество классов")
    parser.add_argument("--thing-classes", nargs="+", default=["frame"], type=str, help="Имена классов")
    parser.add_argument("--confidence-threshold", default=0.5, type=float, help="Порог уверенности для детекций")
    parser.add_argument("--device", default=None, type=str, choices=["cpu", "cuda"], help="Устройство для инференса (cpu/cuda). По умолчанию определяется автоматически")
    parser.add_argument("--detect-overlaps", action="store_true", default=True, help="Обнаруживать перекрывающиеся детекции")
    parser.add_argument("--iou-threshold", default=0.3, type=float, help="Порог IoU для определения перекрытия (по умолчанию 0.3)")
    
    args = parser.parse_args()
    
    # Настройка конфигурации
    cfg = setup_cfg(args)
    
    # Создание предиктора
    predictor = DefaultPredictor(cfg)
    
    # Загрузка изображения
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Изображение не найдено: {image_path}")
    
    image = read_image(str(image_path), format="BGR")
    
    # Запуск инференса
    print(f"Запуск инференса на изображении: {image_path}")
    predictions = predictor(image)
    
    # Сохранение масок
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_masks(predictions, output_dir, image_path.name, detect_overlapping=args.detect_overlaps, args=args)
    
    # Сохранение визуализации
    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    visualizer = Visualizer(image[:, :, ::-1], metadata=metadata, scale=1.0, instance_mode=ColorMode.IMAGE)
    vis_output = visualizer.draw_instance_predictions(predictions=predictions["instances"].to("cpu"))
    
    vis_filename = output_dir / f"{image_path.stem}_visualization.png"
    vis_output.save(str(vis_filename))
    print(f"Сохранена визуализация: {vis_filename}")
    
    # Вывод статистики
    num_instances = len(predictions["instances"])
    print(f"\n Обнаружено объектов: {num_instances}")
    if num_instances > 0:
        scores = predictions["instances"].scores.cpu().numpy()
        print(f"Средняя уверенность: {scores.mean():.3f}")
        print(f"Минимальная уверенность: {scores.min():.3f}")
        print(f"Максимальная уверенность: {scores.max():.3f}")
    
    print(f"\nРезультаты сохранены в: {output_dir}")


if __name__ == "__main__":
    main()
