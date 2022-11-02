# post_stamps

### Проект по распознаванию фейоквых и оригинальных почтовых марок

### Зачем это нужно?
* 
*

Установка заивисмостей
```
pip install -r requirements.txt
```

Запуск на данных
```
# положить изображения в папку data/marks (временно, исправить на подавать паппку в аргументах)
python main.py
```

| Model          | Описание                    | Metric Value |
| ------------- |------------------| -----|
| Classic CV | ORB keypoints + crop + ssim                              |
| Neural CV |  ReID   | |