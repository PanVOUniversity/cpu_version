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


def save_masks(predictions, output_dir: Path, image_name: str):
    """Сохраняет маски в отдельные файлы."""
    instances = predictions["instances"].to("cpu")
    num_instances = len(instances)
    
    if num_instances == 0:
        print("Не обнаружено объектов на изображении")
        return
    
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
    save_masks(predictions, output_dir, image_path.name)
    
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
