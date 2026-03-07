# 🛒 CartPole Genesis Environment

> **Статус:** ⚠️ Экспериментальный
> **Текущий рекорд:** 🏆 71 баллов (@shuanat)

Требует дополнительной настройки типов нейронов. Спайки успешно проходят, и примерно за 30 эпизодов сеть обучается, достигая в пике 19 баллов, но затем начинается деградация. 
**Рекомендация:** стоит увеличить количество нейронов и подобрать оптимальные параметры.

---

## 🚀 Запуск среды

Введите в своем терминале чтобы автоматически найти и перейти в папку с примером:
```bash
cd $(find ~/Genesis -type d -name "cartpole" -print -quit)
```

При первом запуске необходимо создать виртуальное окружение и установить зависимости:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib gymnasium pygame
```
Собрать проект:
```bash
cargo build --release
```

## 📊 Как увидеть результаты?

- Посмотреть дебаг исходного файла `shard.state`:
```bash
python3 scripts/brain_debugger.py baked/MotorCortex/shard.state
```

- (После экспериментов) Посмотреть дебаг последнего чекпоинта:
```bash
python3 scripts/brain_debugger.py baked/MotorCortex/checkpoint.state
```

Для тестирования необходимо сначала запечь сеть, поднять две ноды (коры) и запустить Python-клиент с самим окружением.

### 0. Запекание сети (Baking)
> **Важно:** Перед каждым новым запеканием обязательно удаляйте старую папку `baked` в корне проекта!
```bash
rm -rf baked/ && \
pkill -f genesis-node; \
pkill -f genesis-baker-daemon; \
rm -f /dev/shm/genesis_shard_* && \
cargo run --release -p genesis-baker --bin baker -- --brain examples/cartpole/config/brain.toml
```

### 1. Запуск SensoryCortex (Нода №1)
```bash
cargo run --release -p genesis-node -- --manifest baked/SensoryCortex/manifest.toml --batch-size 100
```

### 2. Запуск MotorCortex (Нода №2)
```bash
cargo run --release -p genesis-node -- --manifest baked/MotorCortex/manifest.toml --batch-size 100
```

### 3. Запуск Клиента (CartPole)
```bash
python3 examples/cartpole/cartpole_client.py
```

### 4. Остановка всех процессов и очистка
```bash
pkill -f genesis-node
pkill -f genesis-baker-daemon
rm -f /dev/shm/genesis_shard_*
rm -rf baked/
```

---

## 🔬 Зона экспериментов

Успешный запуск — это уже победа! Но главная цель: **преодолеть затухание сети и побить рекорд в 71 балл.**

Что можно менять:
- **Настройки нейронов:** редактируйте параметры в файлах `blueprints.toml`
- **Параметры клиента:** модифицируйте логику в `cartpole_client.py`
- **Архитектуру:** экспериментируйте с радиусами, весами и синапсами.

### 🏆 Hall of Fame
Если вам удалось стабилизировать этот тест и перебить текущий рекорд — это победа вдвойне! 
Обязательно делитесь результатами: **присылайте PR**, ваша заслуга будет принята с радостью и зафиксирована в этом списке контрибьюторов.

## 🏆 Список участников (Leaderboard)

| Участник | Рекорд | Дата | Комментарий / Изменения | Коммит |
| :--- | :---: | :---: | :--- | :---: |
| **@shuanat** | **71** | 07.03.2026 | Добавил `random.randint(0, 1)` при равенстве `L=0`, `R=0` (смягчило затухание) | `3ed37ac` |
| **@H4V1K-dev** | **19** | 06.03.2026 | Просто запустил этот тест | `462110a` |

## Удачи! 🚀
