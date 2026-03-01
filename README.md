# Audio Anti-Spoofing with Deep Learning

<p align="center">
<a href="https://github.com/Blinorot/pytorch_project_template/generate">
  <img src="https://img.shields.io/badge/use%20this-template-green?logo=github">
</a>
<a href="https://github.com/ILYA-TDW-CHEP/Intro2DL/src/LICENSE">
   <img src=https://img.shields.io/badge/license-MIT-blue.svg>
</a>
</p>

## О проекте

Данный репозиторий содержит реализацию модели для решения задачи audio anti-spoofing - обнаружения поддельной (синтезированной или воспроизведённой) речи.

Проект основан на современных научных публикациях и воспроизводит архитектуру победителя международного соревнования ASVspoof 2019.

В качестве основы использован публичный шаблон для Deep Learning-проектов, разработанный в рамках курса по глубокому обучению в
National Research University Higher School of Economics.

Шаблон обеспечивает:

- воспроизводимость экспериментов

- удобную конфигурацию

- логирование

## Основные возможности
- Обучение модели для задачи audio anti-spoofing
  
- Поддержка гибкой конфигурации через Hydra

- Логирование экспериментов (WandB / Comet ML)

- Воспроизводимые эксперименты

- Чистая модульная архитектура

- Поддержка inference и evaluation
  
## Обучение модели

Для запуска обучения используйте:

```
python3 train.py -cn=CONFIG_NAME
```

## Инференс и оценка

Для запуска inference:

```
python3 inference.py model.checkpoint=checkpoints/best.ckpt
```

## Логирование экспериментов
Проект поддерживает:

- Weights & Biases (WandB)

- Comet ML

## Полезные ссылки

Данный проект выполнен на основе следующего template: [pytorch-template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
