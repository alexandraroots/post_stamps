# post_stamps

### Проект по распознаванию фейковых и оригинальных почтовых марок

### Зачем это нужно?
* Коллекционерам важно, чтобы марки были подлинными
* Хорошо развита область подделок, т.к. стоимость марки может быть в диапазоне 1$ - 1.000.000$
* Коллекционеры платят биохимическим лабораториям, чтобы понять оригинал ли перед ними
    * биохимический анализ клея
    * анализ печатей на марках (поверх/внизу какого-то текста)
    * проверяют ребристый контур марки, совпадает ли с оригиналом
    
*но* обычно есть данные о марках из энциклопедий и справочников и есть понимание, как выглядит оригинал

Установка заивисмостей
```
pip install -r requirements.txt
```

## Classic CV

| Anchor/Positive        | Anchor/Negative 
| ------------- |------------------|
|<img src="https://github.com/alexandraroots/post_stamps/raw/master/data/diff/diff_orig_8.png" width="1200"> | <img src="https://github.com/alexandraroots/post_stamps/raw/master/data/diff/diff_8.png" width="1200"> 

```
# положить изображения в папку data/marks (временно, исправить на подавать паппку в аргументах)
python main.py
```

## Neural CV
Напришивается метрическое обучение (metric learning) и triplet loss


| Anchor         | Positive                  | Negative |
| ------------- |------------------| -----|
|<img src="https://github.com/alexandraroots/post_stamps/raw/master/data/images/anchor.png" width="1200"> | <img src="https://github.com/alexandraroots/post_stamps/raw/master/data/images/positive.png" width="1200"> |<img src="https://github.com/alexandraroots/post_stamps/raw/master/data/images/negative.png" width="1200">


### Результаты

| Model          | Описание        | Metric Value |Производительность на 1 изображении|
| ------------- |------------------| -----|-----|
| Classic CV | ORB keypoints + crop + ssim                              ||
| Neural CV |  MetricLearning   | ||