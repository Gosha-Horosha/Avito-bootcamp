"""Нормализация и лемматизация текста, извлечение «Вид/Тип услуги» из параметров."""
from __future__ import annotations

import re
from functools import lru_cache

import pymorphy2

_TOKEN_RE = re.compile(r"[a-zа-я0-9]+")
_MORPH = pymorphy2.MorphAnalyzer()

# Ключи из item_infm_params_text: значение параметра тянется до следующего ключа
_PARAM_KEYS = [
    "Вид услуги", "Тип услуги", "Место оказания услуг", "Тип стоимости", "Начальная цена",
    "Работа по договору", "Гарантия", "Бригада", "Время для связи", "График работы",
    "Выполняю заказы", "Опыт работы", "Где вы оказываете услуги", "Ваши клиенты",
    "Кто оказывает услуги", "Услуга", "Стоимость", "Название услуги", "Куда выезжаете",
    "Продолжительность", "Готов закупить материалы", "Время работы", "Год окончания",
    "Специальность", "Учебное учреждение", "Вид работ", "Тип работ", "Тип техники",
    "Вид техники", "Марка", "Предмет", "Формат занятий",
]
_KEY_ALT = "|".join(re.escape(k) for k in sorted(_PARAM_KEYS, key=len, reverse=True))
_SERVICE_RE = re.compile(rf"(Вид услуги|Тип услуги)\s+(.*?)(?=\s+(?:{_KEY_ALT})\b|$)")


def normalize(text: str) -> str:
    text = text.lower().replace("ё", "е").replace("\xa0", " ")
    return " ".join(_TOKEN_RE.findall(text))


@lru_cache(maxsize=2_000_000)
def lemma(token: str) -> str:
    if len(token) < 3 or not token.isalpha():
        return token
    return _MORPH.parse(token)[0].normal_form.replace("ё", "е")


def lemmatize(text: str) -> str:
    return " ".join(lemma(t) for t in normalize(text).split())


def service_params(text: str) -> str:
    """Значения «Вид услуги» и «Тип услуги» — общий словарь фильтров запроса и параметров объявления."""
    if not text:
        return ""
    return " ".join(m.group(2) for m in _SERVICE_RE.finditer(text))
