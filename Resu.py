#print('Код запустился')
#import os
#os.environ["WANDB_MODE"] = "disabled"
## Импорт библиотек
#import numpy as np
#import torch
#import nltk
#from transformers import T5TokenizerFast, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
#from transformers.optimization import Adafactor, AdafactorSchedule
#from datasets import load_dataset
#import evaluate
#
## Путь к модели и параметры для обучения
#MODEL_NAME = "t5-small" # Введите наазвание выбранной модели из хаба
#MAX_INPUT = 256  # Введите максимальную длинну входных данных  в токенах (длинна входных фраз в словах (можно считать полслова токен))
#MAX_OUTPUT  = 256 # Введите максимальную длинну прогнозов в токенах (можно уменьшить для задач суммризации или других задач где выход короче)
#BATCH_SIZE = 16
#DATASET = 'UrukHan/t5-russian-spell_I'   # Введите наазвание название датасет
#
## Проверка доступности GPU
#device = "cuda" if torch.cuda.is_available() else "cpu"
#print("Используемое устройство:", device)
#torch.cuda.empty_cache()
#print('Кэш очищен')
#
#data = load_dataset(DATASET)
## Загрузка токенизатора и модели
#tokenizer = T5TokenizerFast.from_pretrained(MODEL_NAME)
#model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
#model.config.max_length = MAX_OUTPUT
#
##train = data['train']
##test = data['test'].train_test_split(0.02)['test']
#data_split = data['train'].train_test_split(test_size=0.70)
#train = data_split['train']
#
#train_split = train.train_test_split(test_size=0.05)
#train = train_split['train']
#test = train_split['test']
#
#def compute_metrics(eval_pred):
#  predictions, labels = eval_pred
#  decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
#  # Replace -100 in the labels as we can't decode them.
#  labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
#  decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
#
#  # Rouge expects a newline after each sentence
#  decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
#  decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
#
#  result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
#  # Extract a few results
#  result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
#
#  # Add mean generated length
#  prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
#  result["gen_len"] = np.mean(prediction_lens)
#
#  return {k: round(v, 4) for k, v in result.items()}
#
#training_args = Seq2SeqTrainingArguments(
#  output_dir="C://Users//user//Downloads//fsdfs//my_model_output7",  # Папка для сохранения модели
#  run_name="C://Users//user//Downloads//fsdfs//my_unique_run_name",   # Уникальное имя для записи в W&B
#  eval_strategy='steps',
#  eval_steps=2000,
#  save_steps=2000,
#  num_train_epochs=1,
#  predict_with_generate=True,
#  per_device_train_batch_size=BATCH_SIZE,
#  per_device_eval_batch_size=BATCH_SIZE,
#  fp16=True,
#  save_total_limit=2,
#  weight_decay=0.05,
#  push_to_hub=False,
#)
#
#optimizer = Adafactor(
#    model.parameters(),
#    eps=(1e-30, 1e-3),
#    clip_threshold=1.0,
#    decay_rate=-0.8,
#    beta1=None,
#    weight_decay=0.0,
#    relative_step=True,
#    scale_parameter=True,
#    warmup_init=True,
#)
#
#lr_scheduler = AdafactorSchedule(optimizer)
#
#trainer = Seq2SeqTrainer(
#  model=model,
#  args=training_args,
#  train_dataset = train,
#  eval_dataset = test,
#  optimizers = (optimizer, lr_scheduler),
#  tokenizer = tokenizer,
#)
#
## Обучение модели
#print("Запуск обучения модели")
#trainer.train()
#print('Повезло сохранилась все таки')
## Сохранение обученной модели и токенизатора
#trainer.save_model("C://Users//user//Downloads//fsdfs//Results7")
#tokenizer.save_pretrained("C://Users//user//Downloads//fsdfs//Results7")
#
#print("Модель успешно обучена и сохранена")
#print('Повезло сохранилась все таки')






#print('Запуск кода')
#import torch
#from transformers import AutoModelForSeq2SeqLM, T5TokenizerFast, Trainer, TrainingArguments
#from datasets import load_dataset
#import re
#
## Параметры
#MODEL_NAME = 'C://Users//user//Downloads//fsdfs//ezpz'  # Исходная модель
#DATASET_NAME = 'ai-forever/spellcheck_punctuation_benchmark'
#MAX_INPUT_LENGTH = 128
#MAX_TARGET_LENGTH = 128
#BATCH_SIZE = 16
#EPOCHS = 3
#LEARNING_RATE = 3e-5
#
#device = "cuda" if torch.cuda.is_available() else "cpu"
#print("Используемое устройство:", device)
#torch.cuda.empty_cache()
#print('Кэш очищен')
#
## Загрузка токенизатора и модели
#tokenizer = T5TokenizerFast.from_pretrained(MODEL_NAME)
#model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
#
## Загрузка и подготовка датасета
#dataset = load_dataset(DATASET_NAME, 'MultidomainGold', trust_remote_code=True)
#
#def remove_punctuation(text):
#    return re.sub(r'[.,]', '', text)
#print(dataset)
#dataset = dataset.map(lambda x: {'source': remove_punctuation(x['source'])})
#print(dataset['train'][0])  # Печатаем первый пример для проверки
#def preprocess_data(batch):
#    # Добавление префикса к исходному тексту для указания задачи исправления
#    inputs = [f"Spell correct: {text}" for text in batch['source']]
#    targets = batch['correction']
#    
#    # Токенизация входных данных и меток
#    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
#    labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")
#
#    # Добавление меток в формат модели
#    model_inputs['labels'] = labels['input_ids']
#    return model_inputs
#
## Преобразование данных
#
#tokenized_dataset = dataset.map(preprocess_data, batched=True)
#train_test_split = tokenized_dataset['test'].train_test_split(test_size=0.1)
#train_dataset = train_test_split['train']
#eval_dataset = train_test_split['test']
## Печать для проверки
#print(f"Размер тренировочного набора: {len(train_dataset)}")
#print(f"Размер валидационного набора: {len(eval_dataset)}")
## Печать структуры датасета, чтобы увидеть доступные части
#print(tokenized_dataset)
#print('ПРОШЕЛ ЭТОТ ЭТАП!!!')
## Задание параметров обучения
#training_args = TrainingArguments(
#    output_dir="C://Users//user//Downloads//fsdfs//saved",
#    evaluation_strategy="epoch",
#    learning_rate=LEARNING_RATE,
#    per_device_train_batch_size=BATCH_SIZE,
#    per_device_eval_batch_size=BATCH_SIZE,
#    num_train_epochs=EPOCHS,
#    weight_decay=0.01,
#    save_total_limit=1,
#    save_strategy="epoch",
#    logging_dir='C://Users//user//Downloads//fsdfs//logs',
#    logging_steps=50,
#    load_best_model_at_end=True,
#    metric_for_best_model="loss"
#)
#
## Определение метрик
#trainer = Trainer(
#    model=model,
#    args=training_args,
#    train_dataset=train_dataset,
#    eval_dataset=eval_dataset,
#    tokenizer=tokenizer
#)
#
## Запуск обучения
#trainer.train()
#
#
## Сохранение модели
#trainer.save_model("C://Users//user//Downloads//fsdfs//ezpzzzzz")
#tokenizer.save_pretrained("C://Users//user//Downloads//fsdfs//ezpzzzzz")





print('Запуск кода')
import torch
from transformers import AutoModelForSeq2SeqLM, T5TokenizerFast, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
import pandas as pd
import random
import re

# Параметры
MODEL_NAME = 'C://Users//user//Downloads//fsdfs//ezpz'  # Исходная модель
DATASET_NAME = 'ai-forever/spellcheck_punctuation_benchmark'
MAX_INPUT_LENGTH = 128
MAX_TARGET_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 3e-5

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Используемое устройство:", device)
torch.cuda.empty_cache()
print('Кэш очищен')

# Загрузка токенизатора и модели
tokenizer = T5TokenizerFast.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# Загрузка исходного датасета
dataset = load_dataset(DATASET_NAME, 'MultidomainGold', trust_remote_code=True)

def remove_punctuation(text):
    return re.sub(r'[.,]', '', text)

# Увеличение датасета
def generate_mistakes(word, n=5):
    """Создаёт n вариантов с ошибками для одного слова."""
    def drop_random_char(w):  # Удаляет случайный символ
        if len(w) > 1:
            idx = random.randint(0, len(w) - 1)
            return w[:idx] + w[idx+1:]
        return w

    def add_random_char(w):  # Добавляет случайный символ
        idx = random.randint(0, len(w))
        char = random.choice("абвгдеёжзийклмнопрстуфхцчшщъыьэюя")
        return w[:idx] + char + w[idx:]

    def swap_adjacent_chars(w):  # Меняет соседние символы
        if len(w) > 1:
            idx = random.randint(0, len(w) - 2)
            return w[:idx] + w[idx+1] + w[idx] + w[idx+2:]
        return w

    def duplicate_random_char(w):  # Дублирует случайный символ
        if len(w) > 0:
            idx = random.randint(0, len(w) - 1)
            return w[:idx] + w[idx] + w[idx] + w[idx+1:]
        return w

    mistakes = []
    for _ in range(n):
        error_type = random.choice([drop_random_char, add_random_char, swap_adjacent_chars, duplicate_random_char])
        mistakes.append(error_type(word))
    return mistakes

# Преобразование исходного датасета в формат DataFrame
original_data = [{"source": item['source'], "correction": item['correction']} for item in dataset['train']]
df = pd.DataFrame(original_data)

# Генерация новых данных с ошибками
augmented_data = []
for _, row in df.iterrows():
    correct_text = row['correction']
    mistakes = generate_mistakes(correct_text)
    for mistake in mistakes:
        augmented_data.append({"source": mistake, "correction": correct_text})

# Объединение исходных и увеличенных данных
df_augmented = pd.DataFrame(augmented_data)
df_combined = pd.concat([df, df_augmented])

# Преобразование в Dataset
augmented_dataset = Dataset.from_pandas(df_combined)

# Токенизация данных
def preprocess_data(batch):
    inputs = [f"Spell correct: {text}" for text in batch['source']]
    targets = batch['correction']
    model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LENGTH, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=MAX_TARGET_LENGTH, truncation=True, padding="max_length")
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

tokenized_dataset = augmented_dataset.map(preprocess_data, batched=True)
train_test_split = tokenized_dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

# Параметры обучения
training_args = TrainingArguments(
    output_dir="C://Users//user//Downloads//fsdfs//test",
    evaluation_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    save_total_limit=1,
    save_strategy="epoch",
    logging_dir='C://Users//user//Downloads//fsdfs//logs',
    logging_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="loss"
)

# Создание и запуск тренера
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer
)

trainer.train()

# Сохранение модели
trainer.save_model("C://Users//user//Downloads//fsdfs//test")
tokenizer.save_pretrained("C://Users//user//Downloads//fsdfs//test")

print("Обучение завершено и модель сохранена.")
