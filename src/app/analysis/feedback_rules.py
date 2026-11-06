"""Правила оценки техники: THRESHOLDS, детекция ошибок и рекомендации.

Шаг 6а рефакторинга: вынесено из src/app/main.py. Тела функций не менялись.
"""

# Эвристические пороги для оценки техники упражнений.
# Значения подобраны на основе типовых ошибок начинающих гимнасток
# и будут уточняться по мере накопления данных.
THRESHOLDS = {
    "risk-buhavy": {
        "min_tuck_tilt_deg": 60.0,   # минимальный макс. угол наклона при кувырке
        "min_reach_ratio": 0.30,     # минимальная высота подброса булав (0..1)
        "min_hip_low_y": 0.55,       # минимальный нижний уровень таза при кувырке (0..1, чем больше — тем ниже)
        "max_club_height_diff": 0.08,  # допустимая разница высот броска двух булав (0..1)
        "max_peak_frame_diff": 4,      # допустимая разница кадров достижения пика (одновременность)
        "max_catch_x_deviation": 0.20, # допустимое отклонение точки ловли от центра (по x)
        "max_jerk_count": 15,        # допустимое число рывков при ловле
    },
    "jump": {
        "min_jump_y_range": 0.15,    # минимальный вертикальный размах центра масс
        "min_backbend_tilt_deg": 35.0,  # минимальный макс. наклон корпуса при прогибе
        "min_leg_opening_deg": 90.0, # минимальный угол раскрытия ног (шпагат)
        "max_hip_lift_y": 0.45,      # максимальная верхняя точка таза при прыжке (0..1, чем меньше — тем выше)
        "max_ankle_lift_y": 0.80,    # максимальная высота стоп при отрыве (0..1, чем меньше — тем выше)
        "max_jerk_count": 15,        # допустимое число рывков при приземлении
    },
    "balance": {
        "max_com_x_range": 0.10,     # допустимый горизонтальный дрейф центра масс
        "max_avg_tilt_deg": 10.0,    # допустимый средний наклон корпуса
        "min_knee_extension_deg": 150.0,  # минимальный угол в колене опорной ноги (чем ближе к 180°, тем прямее)
        "min_ankle_height_diff": 0.08,  # минимальная разница высот стоп для определения опорной ноги
    },
    "rotation": {
        "max_avg_tilt_deg": 12.0,    # допустимый средний наклон оси вращения
        "max_com_x_range": 0.12,     # допустимое смещение оси вращения
        "max_signed_tilt_deg": 10.0, # допустимый систематический завал оси в одну сторону
        "max_arm_symmetry": 0.10,    # допустимая разница высот запястий (0..1)
        "min_turns": 1.0,            # ожидаемое минимальное число витков
        "max_jerk_count": 20,        # допустимое число рывков при вращении
    },
    "walkover-front": {
        "min_pass_tilt_deg": 150.0,    # полный проход через стойку на руках (макс. наклон корпуса)
        "min_split_deg": 150.0,        # раскрытие ног (шпагат) при проходе над головой
        "min_leg_straight_deg": 150.0, # прямые ноги при проходе (мин. угол в колене)
        "min_hands_down_y": 0.70,      # руки должны опуститься до пола (0..1, чем больше — тем ниже)
        "max_jerk_count": 15,          # допустимое число рывков при выходе из переворота
    },
}

# Штрафы за ошибки (используются для расчёта общей оценки)
SEVERITY_PENALTY = {
    "critical": 25,
    "warning": 10,
}

# Корректирующие упражнения: подсказка по ключевой фразе ошибки -> предлагаемое упражнение.
# Тренер может дополнить/уточнить формулировки — они выводятся в PDF-карточке.
SUGGESTED_DRILLS = [
    ("кувырок", "Перекаты в группировке вперёд-назад, 3×8 повторений"),
    ("подброс", "Подброс булав от плеча у стены с контролем высоты, 3×10"),
    ("прыжок слишком низкий", "Прыжки из полуприседа с выпрыгиванием вверх, 3×10"),
    ("прогиб", "Мостик из положения лёжа и «корзиночка», 2×10"),
    ("раскрыт", "Шпагаты и складки у опоры, ежедневная растяжка 10 мин"),
    ("вертикальный отрыв", "Прыжки на скакалке с высокой голенью, 3×30 сек"),
    ("стопы", "Подъёмы на носки и прыжки «стрелочка» с вытягиванием носков, 3×12"),
    ("раскачивание", "Стойка на одной ноге с закрытыми глазами у опоры, 3×30 сек"),
    ("опорн", "Подъёмы на полупальцы одной ноги у опоры, 3×12 на каждую"),
    ("ось вращения", "Повороты на 180°/360° с фиксацией позы, 3×8"),
    ("витков", "Пируэты с партнёром-страховкой, постепенно наращивая число витков"),
    ("асимметричн", "Отработка позиций рук перед зеркалом, 2×10"),
    ("одновременно", "Синхронный подброс двух булав под счёт, 3×10"),
    ("разной высоте", "Подброс одной булавы каждой рукой поочерёдно на одинаковую высоту, 3×10"),
    ("смещена от центра", "Ловля булав с фиксацией рук на оси тела перед зеркалом, 2×10"),
    ("жёсткая", "Жонглирование/ловля с мягким «доигрыванием» кистью, 3×5 мин"),
    ("стойку на руках", "Стойка на руках у стены с выходом в вертикаль, 4×15 сек"),
    ("не опускаются до пола", "Наклоны вперёд с постановкой рук на пол и переносом веса на руки, 3×8"),
    ("согнуты в коленях", "Махи ногами с выпрямленными коленями и растяжка под шпагат, 3×10"),
    ("резкий", "Плавные приземления из стойки на руках с амортизацией в колене, 3×8"),
]


def _suggested_drill(message: str):
    """Подбирает корректирующее упражнение по тексту ошибки."""
    msg = (message or "").lower()
    for key, drill in SUGGESTED_DRILLS:
        if key in msg:
            return drill
    return None


def _fmt(value, suffix=""):
    """Форматирует число для текста рекомендации."""
    if value is None:
        return "—"
    return f"{value}{suffix}"


def detect_errors_and_recommendations(exercise: str, movement: dict, clubs: dict = None) -> dict:
    """
    Детектирует ошибки техники и формирует рекомендации для конкретного
    упражнения на основе агрегированных метрик движения.

    clubs: данные трекинга булав (результат analyze_club_flight) или None.

    Возвращает:
    {
        "overall_score": 0..100,
        "errors": [{"severity": "critical"|"warning", "message": "..."}],
        "recommendations": ["..."],
        "summary": "..."
    }
    """
    errors = []
    recommendations = []

    if exercise not in THRESHOLDS:
        return {
            "overall_score": None,
            "errors": [],
            "recommendations": [
                "Упражнение не распознано. Выберите упражнение из списка и повторите анализ."
            ],
            "summary": "Не удалось подобрать правила проверки для выбранного упражнения.",
        }

    thresholds = THRESHOLDS[exercise]

    torso = movement.get("torso_tilt") or {}
    wrist = movement.get("wrist_height") or {}
    trajectory = movement.get("trajectory") or {}
    motion = movement.get("motion") or {}
    knee = movement.get("knee_angle") or {}
    leg_opening = movement.get("leg_opening") or {}
    hip_height = movement.get("hip_height") or {}
    torso_signed = movement.get("torso_tilt_signed") or {}
    ankle_height = movement.get("ankle_height") or {}
    ankle_y = movement.get("ankle_y") or {}
    arm_symmetry = movement.get("arm_symmetry") or {}
    rotation_turns = movement.get("rotation_turns") or {}

    # Если поза почти не была обнаружена — данных для проверки недостаточно
    if not torso and not wrist and not trajectory:
        return {
            "overall_score": None,
            "errors": [
                {"severity": "critical", "message": "Недостаточно данных о позе: человек не был стабильно обнаружен в кадре."}
            ],
            "recommendations": [
                "Снимите видео при хорошем освещении, в полный рост, без перекрытий.",
                "Убедитесь, что тело целиком попадает в кадр на протяжении всего упражнения.",
            ],
            "summary": "Анализ невозможен из-за отсутствия данных о позе.",
        }

    # --- Риск: ловля двух булав после кувырка ---
    if exercise == "risk-buhavy":
        max_tilt = torso.get("max_deg")
        if max_tilt is not None and max_tilt < thresholds["min_tuck_tilt_deg"]:
            errors.append({
                "severity": "critical",
                "message": (
                    f"Кувырок выполнен недостаточно глубоко: максимальный наклон корпуса "
                    f"{_fmt(max_tilt, '°')} (нужно не менее {thresholds['min_tuck_tilt_deg']:.0f}°). "
                    "Булавы не успеют перелететь над кувырком."
                ),
            })
            recommendations.append(
                "Сильнее группируйтесь и активнее перекатывайтесь через плечо, чтобы корпус "
                "проходил под траекторией полёта булав."
            )

        reach_ratio = wrist.get("reach_ratio")
        if reach_ratio is not None and reach_ratio < thresholds["min_reach_ratio"]:
            errors.append({
                "severity": "critical",
                "message": (
                    f"Подброс булав недостаточно высокий: руки поднялись лишь на "
                    f"{_fmt(round(reach_ratio * 100), '%')} высоты кадра "
                    f"(нужно не менее {thresholds['min_reach_ratio'] * 100:.0f}%)."
                ),
            })
            recommendations.append(
                "Подбрасывайте булавы энергичным движением от плеча, направляя их вверх, "
                "и добавляйте небольшой подъём на носки для увеличения высоты полёта."
            )

        # Асимметричный подброс: усреднение запястий может маскировать низкий
        # подброс одной руки, поэтому проверяем досягаем��сть каждой руки отдельно
        side_reaches = [
            (label, wrist.get(f"{side}_reach_ratio"))
            for side, label in (("left", "левой"), ("right", "правой"))
        ]
        for side_label, side_reach in side_reaches:
            if side_reach is not None and side_reach < thresholds["min_reach_ratio"]:
                errors.append({
                    "severity": "critical",
                    "message": (
                        f"Подброс недостаточно высокий для {side_label} руки: досягаемость "
                        f"{_fmt(round(side_reach * 100), '%')} высоты кадра "
                        f"(нужно не менее {thresholds['min_reach_ratio'] * 100:.0f}%). "
                        "Булава со стороны этой руки летит ниже."
                    ),
                })
                recommendations.append(
                    f"Поднимайте {side_label} руку так же высоко, как противоположную, "
                    "чтобы бросок был симметричным и булавы летели на одной высоте."
                )

        # Кувырок: низкая группировка — таз должен опускаться достаточно низко
        hip_max_y = hip_height.get("max_y")
        if hip_max_y is not None and hip_max_y < thresholds["min_hip_low_y"]:
            errors.append({
                "severity": "warning",
                "message": (
                    f"Недостаточная группировка при кувырке: таз опускается только до "
                    f"{_fmt(hip_max_y)} высоты кадра (нужно ниже {thresholds['min_hip_low_y']}). "
                    "Корпус проходит слишком высоко над полом."
                ),
            })
            recommendations.append(
                "Сильнее группируйтесь: подтягивайте колени к груди и ниже опускайте таз, "
                "чтобы корпус проходил под булавами."
            )

        # --- Трекинг булав: параметры полёта предметов ---
        clubs_info = clubs or {}
        throw_height = clubs_info.get("throw_height") or {}
        symmetry = clubs_info.get("symmetry") or {}
        simultaneity = clubs_info.get("simultaneity") or {}
        catch_position = clubs_info.get("catch_position") or {}

        if clubs_info.get("clubs_detected", 0) >= 2:
            # Высота броска (по пикам траекторий)
            max_height_y = throw_height.get("max")
            if max_height_y is not None and max_height_y > thresholds["min_reach_ratio"]:
                errors.append({
                    "severity": "warning",
                    "message": (
                        f"Бросок булав недостаточно высокий: максимальная точка полёта "
                        f"{_fmt(max_height_y)} высоты кадра (нужно выше "
                        f"{thresholds['min_reach_ratio']})."
                    ),
                })
                recommendations.append(
                    "Увеличивайте высоту подброса: энергичнее выталкивайте булавы вверх "
                    "от плеча и добавляйте импульс кистью."
                )

            # Симметрия высоты броска
            height_diff = symmetry.get("height_diff")
            if height_diff is not None and height_diff > thresholds["max_club_height_diff"]:
                errors.append({
                    "severity": "warning",
                    "message": (
                        f"Булавы летят на разной высоте: разница пиков "
                        f"{_fmt(height_diff)} (допустимо не более "
                        f"{thresholds['max_club_height_diff']})."
                    ),
                })
                recommendations.append(
                    "Подбрасывайте обе булавы с одинаковым усилием и на одну высоту, "
                    "контролируя симметрию движения рук."
                )

            # Одновременность
            peak_frame_diff = simultaneity.get("peak_frame_diff")
            if peak_frame_diff is not None and peak_frame_diff > thresholds["max_peak_frame_diff"]:
                errors.append({
                    "severity": "warning",
                    "message": (
                        f"Булавы подброшены не одновременно: разница достижения верхней точки "
                        f"{_fmt(peak_frame_diff, ' кадр.')} (допустимо не более "
                        f"{thresholds['max_peak_frame_diff']})."
                    ),
                })
                recommendations.append(
                    "Работайте руками синхронно: подбрасывайте булавы одним движением, "
                    "не растягивая фазу броска."
                )

            # Позиция ловли: проверяем отклонение от центра по x
            for side_key, label in [("left", "левой"), ("right", "правой")]:
                pos = catch_position.get(side_key) or {}
                catch_x = pos.get("x")
                if catch_x is not None and abs(catch_x - 0.5) > thresholds["max_catch_x_deviation"]:
                    errors.append({
                        "severity": "warning",
                        "message": (
                            f"Ловля {label} булавы смещена от центра: позиция по x "
                            f"{_fmt(round(catch_x, 2))} (центр 0.50, допустимое отклонение "
                            f"{thresholds['max_catch_x_deviation']})."
                        ),
                    })
                    recommendations.append(
                        f"Ловите {label} булаву ближе к оси тела, не уводя руки в сторону."
                    )
        elif clubs_info.get("clubs_detected") == 1:
            errors.append({
                "severity": "warning",
                "message": "Обнаружена только одна булава — оценить симметрию и одновременность броска невозможно.",
            })
            recommendations.append(
                "Убедитесь, что обе булавы хорошо видны в кадре и имеют яркую, контрастную окраску."
            )
        elif clubs is not None and clubs_info.get("clubs_detected") == 0:
            errors.append({
                "severity": "warning",
                "message": "Булавы не обнаружены в кадре — параметры полёта предметов не оценены.",
            })
            recommendations.append(
                "Снимайте при хорошем освещении; булавы должны быть яркими и не сливаться с фоном."
            )

        jerk_count = motion.get("jerk_count")
        if jerk_count is not None and jerk_count > thresholds["max_jerk_count"]:
            errors.append({
                "severity": "warning",
                "message": (
                    f"Ловля булав жёсткая: зафиксировано {jerk_count} рывков "
                    f"(допустимо не более {thresholds['max_jerk_count']})."
                ),
            })
            recommendations.append(
                "Встречайте булавы мягким движением кистей и амортизируйте ловлю "
                "сгибанием рук в локтях."
            )

    # --- Прыжок с прогибом назад ---
    elif exercise == "jump":
        y_range = trajectory.get("y_range")
        if y_range is not None and y_range < thresholds["min_jump_y_range"]:
            errors.append({
                "severity": "critical",
                "message": (
                    f"Прыжок слишком низкий: вертикальный размах центра масс "
                    f"{_fmt(y_range)} (нужно не менее {thresholds['min_jump_y_range']}). "
                    "Прогиб не успевает раскрыться в полёте."
                ),
            })
            recommendations.append(
                "Отталкивайтесь активнее обеими ногами и вытягивайтесь вверх перед прогибом, "
                "чтобы увеличить время полёта."
            )

        max_tilt = torso.get("max_deg")
        if max_tilt is not None and max_tilt < thresholds["min_backbend_tilt_deg"]:
            errors.append({
                "severity": "warning",
                "message": (
                    f"Недостаточный прогиб назад: максимальный наклон корпуса "
                    f"{_fmt(max_tilt, '°')} (нужно не менее {thresholds['min_backbend_tilt_deg']:.0f}°)."
                ),
            })
            recommendations.append(
                "В верхней точке прыжка активнее прогибайтесь в грудном отделе и тяните "
                "руки назад, раскрывая плечи."
            )

        # Прыжок: раскрытие ног (шпагат) — угол должен быть широким
        opening_max = leg_opening.get("max_deg")
        if opening_max is not None and opening_max < thresholds["min_leg_opening_deg"]:
            errors.append({
                "severity": "warning",
                "message": (
                    f"Ноги раскрыты недостаточно широко: максимальный угол раскрытия "
                    f"{_fmt(opening_max, '°')} (нужно не менее {thresholds['min_leg_opening_deg']:.0f}°)."
                ),
            })
            recommendations.append(
                "В полёте активнее разводите ноги в стороны, тяните носки, "
                "чтобы раскрыть шпагат и увеличить амплитуду прыжка."
            )

        # Прыжок: вертикальный отрыв — таз должен подниматься достаточно высоко
        hip_min_y = hip_height.get("min_y")
        if hip_min_y is not None and hip_min_y > thresholds["max_hip_lift_y"]:
            errors.append({
                "severity": "warning",
                "message": (
                    f"Недостаточный вертикальный отрыв: таз поднимается только до "
                    f"{_fmt(hip_min_y)} высоты кадра (нужно выше {thresholds['max_hip_lift_y']}). "
                    "Прыжок получается «плоским»."
                ),
            })
            recommendations.append(
                "Добавьте вертикальный импульс: активнее выталкивайтесь стопами "
                "и тянитесь макушкой вверх в момент отрыва."
            )

        # Прыжок: высота отрыва стоп — стопы должны подниматься от пола
        ankle_min_y = ankle_height.get("min_y")
        if ankle_min_y is not None and ankle_min_y > thresholds["max_ankle_lift_y"]:
            errors.append({
                "severity": "warning",
                "message": (
                    f"Стопы почти не отрываются от пола: максимальная высота стоп "
                    f"{_fmt(ankle_min_y)} (нужно выше {thresholds['max_ankle_lift_y']}). "
                    "Отрыв недостаточно высокий."
                ),
            })
            recommendations.append(
                "Выталкивайтесь активнее носками, поднимайте стопы от пола "
                "и вытягивайте носки в полёте."
            )

        jerk_count = motion.get("jerk_count")
        if jerk_count is not None and jerk_count > thresholds["max_jerk_count"]:
            errors.append({
                "severity": "warning",
                "message": (
                    f"Приземление жёсткое: зафиксировано {jerk_count} рывков "
                    f"(допустимо не более {thresholds['max_jerk_count']})."
                ),
            })
            recommendations.append(
                "Приземляйтесь мягко: с носка на полную стопу, слегка сгибая колени "
                "и сохраняя баланс."
            )

    # --- Равновесие на одной ноге ---
    elif exercise == "balance":
        x_range = trajectory.get("x_range")
        if x_range is not None and x_range > thresholds["max_com_x_range"]:
            errors.append({
                "severity": "critical",
                "message": (
                    f"Раскачивание корпуса: центр масс смещается по горизонтали на "
                    f"{_fmt(x_range)} (допустимо не более {thresholds['max_com_x_range']}). "
                    "Равновесие теряется."
                ),
            })
            recommendations.append(
                "Зафиксируйте взгляд в одной точке, напрягите мышцы пресса и удерживайте "
                "опорную ногу неподвижной."
            )

        avg_tilt = torso.get("avg_deg")
        if avg_tilt is not None and avg_tilt > thresholds["max_avg_tilt_deg"]:
            errors.append({
                "severity": "warning",
                "message": (
                    f"Корпус наклонён: средний наклон {_fmt(avg_tilt, '°')} "
                    f"(допустимо не более {thresholds['max_avg_tilt_deg']:.0f}°). "
                    "Поза выглядит неустойчивой."
                ),
            })
            recommendations.append(
                "Держите спину вертикально, подтяните живот и не опускайте плечи."
            )

        # Равновесие: выпрямленность опорной ноги — угол в колене должен быть близок к 180°
        left_avg = knee.get("left_avg_deg")
        right_avg = knee.get("right_avg_deg")
        # Определяем опорную ногу по высоте лодыжек: опорная стопа обычно ниже (стоит на полу),
        # свободная — поднята. Используем среднюю разницу высот (left - right).
        # y растёт вниз: если left_y - right_y > 0, левая лодыжка НИЖЕ -> опорная левая.
        ankle_diff = ankle_y.get("avg_diff")

        support_side = None
        if ankle_diff is not None:
            if ankle_diff > thresholds["min_ankle_height_diff"]:
                support_side = "left"
            elif ankle_diff < -thresholds["min_ankle_height_diff"]:
                support_side = "right"
        # Запасной вариант: если разница высот мала, используем более выпрямленное колено
        if support_side is None:
            if left_avg is not None and right_avg is not None:
                support_side = "left" if left_avg >= right_avg else "right"
            elif left_avg is not None:
                support_side = "left"
            elif right_avg is not None:
                support_side = "right"

        support_avg = None
        support_label = None
        if support_side == "left":
            support_avg = left_avg
            support_label = "левая"
        elif support_side == "right":
            support_avg = right_avg
            support_label = "правая"

        if support_avg is not None and support_avg < thresholds["min_knee_extension_deg"]:
            errors.append({
                "severity": "warning",
                "message": (
                    f"Опорная нога ({support_label}) согнута: угол в колене "
                    f"{_fmt(support_avg, '°')} (нужно не менее {thresholds['min_knee_extension_deg']:.0f}°). "
                    "Согнутая нога делает равновесие менее устойчивым."
                ),
            })
            recommendations.append(
                f"Выпрямляйте опорную ({support_label}) ногу в колене, тяните коленную "
                "чашечку вверх, не заваливайтесь на пятку."
            )

    # --- Поворот (пируэт) ---
    elif exercise == "rotation":
        avg_tilt = torso.get("avg_deg")
        if avg_tilt is not None and avg_tilt > thresholds["max_avg_tilt_deg"]:
            errors.append({
                "severity": "critical",
                "message": (
                    f"Ось вращения завалена: средний наклон корпуса {_fmt(avg_tilt, '°')} "
                    f"(допустимо не более {thresholds['max_avg_tilt_deg']:.0f}°). "
                    "Вращение получается неустойчивым."
                ),
            })
            recommendations.append(
                "Держите ось вращения строго вертикальной: тянитесь макушкой вверх "
                "и держите корпус собранным."
            )

        x_range = trajectory.get("x_range")
        if x_range is not None and x_range > thresholds["max_com_x_range"]:
            errors.append({
                "severity": "warning",
                "message": (
                    f"Центр масс смещается по горизонтали на {_fmt(x_range)} "
                    f"(допустимо не более {thresholds['max_com_x_range']}). "
                    "Теряется ось вращения."
                ),
            })
            recommendations.append(
                "Перед началом поворота перенесите вес точно на опорную ногу "
                "и сохраняйте проекцию центра масс над стопой."
            )

        # Пируэт: систематический завал оси в одну сторону
        signed_avg = torso_signed.get("avg_deg")
        if signed_avg is not None and abs(signed_avg) > thresholds["max_signed_tilt_deg"]:
            direction = "влево" if signed_avg > 0 else "вправо"
            errors.append({
                "severity": "warning",
                "message": (
                    f"Ось вращения систематически завалена {direction}: средний наклон "
                    f"{_fmt(signed_avg, '°')} (допустимо не более "
                    f"{thresholds['max_signed_tilt_deg']:.0f}° в любую сторону). "
                    "Вращение уходит в одну сторону."
                ),
            })
            recommendations.append(
                "Следите за симметрией: держите плечи ровно, не наклоняйте корпус "
                f"{direction}, тянитесь строго вверх."
            )

        # Пируэт: число витков (по углу поворота плечевой линии)
        turns = rotation_turns.get("turns")
        total_deg = rotation_turns.get("total_deg")
        if turns is not None and turns < thresholds["min_turns"]:
            errors.append({
                "severity": "warning",
                "message": (
                    f"Мало витков в пируэте: выполнен поворот на {_fmt(total_deg, '°')} "
                    f"(примерно {_fmt(turns, ' об.')}, нужно не менее {thresholds['min_turns']:.0f})."
                ),
            })
            recommendations.append(
                "Увеличивайте темп вращения: активнее работайте руками и не замедляйтесь "
                "между витками."
            )

        # Пируэт: симметрия рук — запястья должны быть на примерно одинаковой высоте
        arm_diff = arm_symmetry.get("avg_diff")
        if arm_diff is not None and arm_diff > thresholds["max_arm_symmetry"]:
            errors.append({
                "severity": "warning",
                "message": (
                    f"Руки асимметричны: разница высот запястий {_fmt(arm_diff)} "
                    f"(допустимо не более {thresholds['max_arm_symmetry']}). "
                    "Это нарушает баланс и ось вращения."
                ),
            })
            recommendations.append(
                "Держите руки на одной высоте, симметрично относительно оси вращения, "
                "не опускайте одну из рук."
            )

        jerk_count = motion.get("jerk_count")
        if jerk_count is not None and jerk_count > thresholds["max_jerk_count"]:
            errors.append({
                "severity": "warning",
                "message": (
                    f"Вращение неравномерное: зафиксировано {jerk_count} рывков "
                    f"(допустимо не более {thresholds['max_jerk_count']})."
                ),
            })
            recommendations.append(
                "Работайте руками равномерно, не «замирайте» между витками — "
                "вращение должно быть непрерывным."
            )

    # --- Переворот вперёд ---
    elif exercise == "walkover-front":
        # Проход через стойку на руках: корпус должен пройти через вертикаль
        max_tilt = torso.get("max_deg")
        if max_tilt is not None and max_tilt < thresholds["min_pass_tilt_deg"]:
            errors.append({
                "severity": "critical",
                "message": (
                    f"Проход через стойку на руках недостаточно вертикальный: максимальный "
                    f"наклон корпуса {_fmt(max_tilt, '°')} (нужно не менее "
                    f"{thresholds['min_pass_tilt_deg']:.0f}°). Переворот выполнен «низко»."
                ),
            })
            recommendations.append(
                "Отрабатывайте стойку на руках у стены и заход в переворот махом, "
                "чтобы корпус проходил через вертикаль."
            )

        # Руки должны опуститься до пола: опора перед проходом
        lowest_y = wrist.get("lowest_y")
        if lowest_y is not None and lowest_y < thresholds["min_hands_down_y"]:
            errors.append({
                "severity": "critical",
                "message": (
                    f"Руки не опускаются до пола: самая низкая точка запястий "
                    f"{_fmt(lowest_y)} высоты кадра (нужно не менее "
                    f"{thresholds['min_hands_down_y']}). Опора под руки недостаточно глубокая."
                ),
            })
            recommendations.append(
                "Наклоняйтесь глубже вперёд и ставьте руки на пол ближе к линии ног, "
                "чтобы перевернуться через руки, а не «провисать» в наклоне."
            )

        # Раскрытие ног (шпагат) при проходе над головой
        opening_max = leg_opening.get("max_deg")
        if opening_max is not None and opening_max < thresholds["min_split_deg"]:
            errors.append({
                "severity": "warning",
                "message": (
                    f"Ноги раскрыты недостаточно широко: м��ксимальный угол раскрытия "
                    f"{_fmt(opening_max, '°')} (нужно не менее "
                    f"{thresholds['min_split_deg']:.0f}°). Проход над головой «согнутый»."
                ),
            })
            recommendations.append(
                "В перевороте тяните ноги в шпагат: активнее разводите их в стороны "
                "и тяните носки, пока корпус идёт через стойку."
            )

        # Прямые ноги: минимальный угол в колене за всё видео
        knee_mins = [
            v for v in (knee.get("left_min_deg"), knee.get("right_min_deg"))
            if v is not None
        ]
        min_knee = min(knee_mins) if knee_mins else None
        if min_knee is not None and min_knee < thresholds["min_leg_straight_deg"]:
            errors.append({
                "severity": "warning",
                "message": (
                    f"Ноги согнуты в коленях при проходе: минимальный угол в колене "
                    f"{_fmt(min_knee, '°')} (нужно не менее "
                    f"{thresholds['min_leg_straight_deg']:.0f}°)."
                ),
            })
            recommendations.append(
                "Держите ноги прямыми на протяжении всего переворота: тяните колени "
                "и носки, не сгибайте колени при махе."
            )

        # Плавность выхода из переворота
        jerk_count = motion.get("jerk_count")
        if jerk_count is not None and jerk_count > thresholds["max_jerk_count"]:
            errors.append({
                "severity": "warning",
                "message": (
                    f"Выход из переворота резкий: зафиксировано {jerk_count} рывков "
                    f"(допустимо не более {thresholds['max_jerk_count']})."
                ),
            })
            recommendations.append(
                "Плавно завершайте переворот: приземляйтесь на маховую ногу "
                "с амортизацией в колене и сразу поднимайтесь в стойку."
            )

        # Подводящие упражнения для переворота вперёд: выводятся всегда,
        # независимо от найденных ошибок (и как развитие элемента в идеальном случае)
        recommendations.extend([
            "Колесо через одну руку с постепенным уменьшением опоры — база для захода "
            "в переворот вперёд.",
            "Перекат вперёд через стойку у стены с мягким приземлением — отработка фазы прохода.",
            "Мостик и растяжка плечевых суставов — подвижность плеч для глубокой постановки рук.",
        ])

    # --- Общая оценка и итоговая рекомендация ---
    if not errors:
        overall_score = 100
        summary = "Отличная техника! Ошибок не обнаружено, продолжайте в том же духе."
        recommendations.insert(
            0,
            "Отличная техника! Для закрепления результата повторите упражнение "
            "в медленном темпе и под контролем тренера.",
        )
    else:
        overall_score = max(
            0,
            100 - sum(SEVERITY_PENALTY[e["severity"]] for e in errors),
        )
        if overall_score >= 80:
            summary = "Техника близка к идеалу: есть небольшие недочёты, которые легко исправить."
        elif overall_score >= 50:
            summary = "Упражнение выполняется, но есть заметные ошибки — обратите внимание на рекомендации."
        else:
            summary = "Техника требует серьёзной доработки. Разберите упражнение по элементам с тренером."

    # Подбираем корректирующее упражнение для каждой ошибки (используется в PDF-отчёте)
    for e in errors:
        e["suggested_drill"] = _suggested_drill(e.get("message"))

    return {
        "overall_score": overall_score,
        "errors": errors,
        "recommendations": recommendations,
        "summary": summary,
    }
