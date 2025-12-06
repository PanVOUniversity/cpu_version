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
  --image images/your_image.png \
  --weights model_final.pth \
  --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
  --output-dir output \
  --num-classes 1 \
  --thing-classes frame \
  --confidence-threshold 0.5 \
  --device cpu
```

## Структура

- `/home/appuser/detectron2_repo/validation/` - рабочая директория с inference.py и model_final.pth
- `/home/appuser/detectron2_repo/configs/` - конфигурационные файлы Detectron2
- `/home/appuser/.local/` - локальные пакеты Python пользователя (установлены через --user)

## Пример использования с монтированием томов

```bash
# Создайте директории для изображений и результатов
mkdir -p images output

# Поместите ваше изображение в папку images
cp your_image.png images/

# Запустите контейнер с монтированием томов
docker run -it --rm \
  -v $(pwd)/images:/home/appuser/detectron2_repo/validation/images \
  -v $(pwd)/output:/home/appuser/detectron2_repo/validation/output \
  detectron2-cpu:latest \
  bash -c "cd /home/appuser/detectron2_repo/validation && python inference.py --image images/your_image.png --weights model_final.pth --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml --output-dir output --num-classes 1 --thing-classes frame --device cpu"
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

