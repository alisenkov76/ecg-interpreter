import allure
import pytest

from ecg_interpreter.ecg.interpreter import interpret_ecg
from ecg_interpreter.models.ecg_data import Rhythm, STSegment
from ecg_interpreter.models.interpretation import Diagnosis, Interpretation, Urgency


@allure.feature("Интерпретация ЭКГ")
@allure.story("interpret_ecg")
class TestInterpretEcg:

    @allure.title("Таблица решений: {reading_fixture}")
    @pytest.mark.parametrize(
        ("reading_fixture", "expected"),
        [
            pytest.param("normal_reading", Interpretation(Diagnosis.NORMAL, Urgency.NONE), id="normal"),
            pytest.param("infarction_reading", Interpretation(Diagnosis.INFARCTION, Urgency.URGENT),
                         id="infarction"),
            pytest.param("suspected_infarction_reading",
                         Interpretation(Diagnosis.SUSPECTED_INFARCTION, Urgency.URGENT), id="suspected_infarction"),
            pytest.param("ischemia_reading", Interpretation(Diagnosis.ISCHEMIA, Urgency.PLANNED), id="ischemia"),
            pytest.param("arrhythmia_reading", Interpretation(Diagnosis.ARRHYTHMIA, Urgency.PLANNED),
                         id="arrhythmia"),
            pytest.param("previous_infarction_reading",
                         Interpretation(Diagnosis.PREVIOUS_INFARCTION, Urgency.PLANNED), id="previous_infarction"),
            pytest.param("bradycardia_reading", Interpretation(Diagnosis.BRADYCARDIA, Urgency.PLANNED),
                         id="bradycardia"),
            pytest.param("tachycardia_reading", Interpretation(Diagnosis.TACHYCARDIA, Urgency.PLANNED),
                         id="tachycardia"),
        ],
    )
    def test_interpret_ecg_decision_table(self, request, reading_fixture, expected):
        reading = request.getfixturevalue(reading_fixture)

        result = interpret_ecg(reading)

        assert result == expected, f"Для {reading} ожидалась интерпретация {expected}, получено {result}"

    @allure.title("Приоритет правил при нескольких признаках")
    @pytest.mark.parametrize(
        ("reading_kwargs", "expected"),
        [
            pytest.param(
                {"st_segment": STSegment.ELEVATION, "q_wave": True, "rhythm": Rhythm.IRREGULAR, "heart_rate": 150.0},
                Interpretation(Diagnosis.INFARCTION, Urgency.URGENT),
                id="infarction_over_all",
            ),
            pytest.param(
                {"st_segment": STSegment.ELEVATION, "rhythm": Rhythm.IRREGULAR},
                Interpretation(Diagnosis.SUSPECTED_INFARCTION, Urgency.URGENT),
                id="suspected_infarction_over_arrhythmia",
            ),
            pytest.param(
                {"st_segment": STSegment.DEPRESSION, "q_wave": True, "rhythm": Rhythm.IRREGULAR},
                Interpretation(Diagnosis.ISCHEMIA, Urgency.PLANNED),
                id="ischemia_over_arrhythmia_and_q_wave",
            ),
            pytest.param(
                {"rhythm": Rhythm.IRREGULAR, "q_wave": True},
                Interpretation(Diagnosis.ARRHYTHMIA, Urgency.PLANNED),
                id="arrhythmia_over_previous_infarction",
            ),
            pytest.param(
                {"q_wave": True, "heart_rate": 40.0},
                Interpretation(Diagnosis.PREVIOUS_INFARCTION, Urgency.PLANNED),
                id="previous_infarction_over_heart_rate",
            ),
        ],
    )
    def test_interpret_ecg_rule_priority(self, make_reading, reading_kwargs, expected):
        reading = make_reading(**reading_kwargs)

        result = interpret_ecg(reading)

        assert result == expected, f"Для {reading} ожидалась интерпретация {expected}, получено {result}"

    @allure.title("Граничные значения ЧСС: {heart_rate}")
    @pytest.mark.parametrize(
        ("heart_rate", "expected"),
        [
            pytest.param(0.0, Interpretation(Diagnosis.BRADYCARDIA, Urgency.PLANNED), id="0"),
            pytest.param(59.9, Interpretation(Diagnosis.BRADYCARDIA, Urgency.PLANNED), id="59.9"),
            pytest.param(60.0, Interpretation(Diagnosis.NORMAL, Urgency.NONE), id="60"),
            pytest.param(100.0, Interpretation(Diagnosis.NORMAL, Urgency.NONE), id="100"),
            pytest.param(100.1, Interpretation(Diagnosis.TACHYCARDIA, Urgency.PLANNED), id="100.1"),
            pytest.param(300.0, Interpretation(Diagnosis.TACHYCARDIA, Urgency.PLANNED), id="300"),
        ],
    )
    def test_interpret_ecg_heart_rate_boundaries(self, make_reading, heart_rate, expected):
        reading = make_reading(heart_rate=heart_rate)

        result = interpret_ecg(reading)

        assert result == expected, f"Для ЧСС {heart_rate} ожидалась интерпретация {expected}, получено {result}"

    @allure.title("None вместо данных ЭКГ вызывает ValueError")
    def test_interpret_ecg_none_raises_error(self):
        with pytest.raises(ValueError, match="Данные ЭКГ не могут быть пустыми"):
            interpret_ecg(None)

    @allure.title("Некорректная ЧСС вызывает ValueError: {heart_rate}")
    @pytest.mark.parametrize(
        ("heart_rate", "message"),
        [
            pytest.param(None, "Значение ЧСС не может быть пустыми", id="none"),
            pytest.param(-1.0, "Значение ЧСС не может быть отрицательным", id="negative"),
            pytest.param(300.001, "Значение ЧСС физиологически невозможно", id="above_max"),
            pytest.param(float("nan"), "Значение ЧСС должно быть конечным числом", id="nan"),
            pytest.param(float("inf"), "Значение ЧСС физиологически невозможно", id="inf"),
            pytest.param(float("-inf"), "Значение ЧСС не может быть отрицательным", id="-inf"),
        ],
    )
    def test_interpret_ecg_invalid_heart_rate_raises_error(self, make_reading, heart_rate, message):
        reading = make_reading(heart_rate=heart_rate)

        with pytest.raises(ValueError, match=message):
            interpret_ecg(reading)

    @allure.title("ЧСС не проверяется, если сработало приоритетное правило")
    def test_interpret_ecg_invalid_heart_rate_ignored_by_priority_rule(self, make_reading):
        reading = make_reading(st_segment=STSegment.ELEVATION, q_wave=True, heart_rate=-10.0)

        result = interpret_ecg(reading)

        expected = Interpretation(Diagnosis.INFARCTION, Urgency.URGENT)
        assert result == expected, f"Ожидалась интерпретация {expected} без проверки ЧСС, получено {result}"
