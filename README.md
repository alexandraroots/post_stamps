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

Заупск
```
git clone https://github.com/alexandraroots/post_stamps.git
docker build . -t main
docker run main
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
| Classic CV | ORB keypoints + crop + ssim                              |7.22| 0.88
| Neural CV |  MetricLearning   |Не получилось реализивать из-за маленького размера выборки ||

Описание экспериментов в grid_search.ipynb

Важно отметить (!) исследования проводились на малом колчичестве данных и только для фальшивок, отличаюзихся типом печати (точки/палочки)
