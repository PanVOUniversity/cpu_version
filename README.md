# CPU Version Docker Image

Docker образ для запуска модели Detectron2 на CPU без GPU.

## Сборка образа

```bash
docker build -t detectron2-cpu:latest .
```

## Запуск контейнера

```bash
docker run -it --rm \
  -v $(pwd)/images:/home/appuser/detectron2_repo/validation/images \
  -v $(pwd)/output:/home/appuser/detectron2_repo/validation/output \
  detectron2-cpu:latest
```

## Запуск инференса

После запуска контейнера вы уже будете в директории validation:

```bash
# Вы находитесь в /home/appuser/detectron2_repo/validation
```

Затем запустите инференс:

```bash
python inference.py \
  --img-dir images \
  --weights model_final.pth \
  --config-file config/config.yaml \
  --output-dir output \
  --num-classes 1 \
  --thing-classes frame \
  --confidence-threshold 0.5 \
  --device cpu
```

## Структура

- `/home/appuser/detectron2_repo/validation/` - рабочая директория с inference.py и model_final.pth
- `/home/appuser/detectron2_repo/validation/config/` - конфигурационный файл config.yaml
- `/home/appuser/detectron2_repo/configs/` - конфигурационные файлы Detectron2
- `/home/appuser/.local/` - локальные пакеты Python пользователя (установлены через --user)

## Пример использования с монтированием томов

```bash
# Создайте директории для изображений и результатов
mkdir -p images output

# Поместите ваши изображения в папку images
cp your_image*.png images/

# Запустите контейнер с монтированием томов
docker run -it --rm \
  -v $(pwd)/images:/home/appuser/detectron2_repo/validation/images \
  -v $(pwd)/output:/home/appuser/detectron2_repo/validation/output \
  detectron2-cpu:latest \
  bash -c "cd /home/appuser/detectron2_repo/validation && python inference.py --img-dir images --weights model_final.pth --config-file config/config.yaml --output-dir output --num-classes 1 --thing-classes frame --device cpu"
```

## Требования

- Docker установлен на системе
- Минимум 4-8 GB RAM (рекомендуется 8+ GB)
- ~2 GB свободного места на диске

## Примечания

- Образ использует Ubuntu 20.04
- PyTorch установлен в CPU версии
- Все пакеты Python установлены через `pip install --user` в `/home/appuser/.local/`
- Контейнер работает от имени пользователя `appuser` (не root)
- Detectron2 компилируется из исходников при сборке образа
- Структура аналогична оригинальному Dockerfile из `train/docker/Dockerfile`

## Пример JSON с перекрытиями

После запуска инференса для страницы `page_1.png` в директории `output` создаётся файл `page_1_overlaps.json` со структурой примерно такого вида:

```json
{
  "image": "page_1.png",
  "iou_threshold": 0.0,
  "total_overlaps": 210,
  "instances": [
    {
      "id": 0,
      "score": 0.9999973773956299,
      "bbox": [194.08, 1778.99, 388.22, 1888.69],
      "bbox_center": [291.15, 1833.84],
      "bbox_size": [194.13, 109.69],
      "mask_area": 20858
    }
    // ... другие объекты ...
  ],
  "overlaps": [
    {
      "instance1": 0,
      "instance2": 12,
      "iou": 0.0429,
      "score1": 0.9999973773956299,
      "score2": 0.9999692440032959,
      "bbox1": [194.08, 1778.99, 388.22, 1888.69],
      "bbox2": [78.11, 1847.26, 261.59, 2101.76]
    }
    // ... остальные пары перекрытий ...
  ]
}
```

