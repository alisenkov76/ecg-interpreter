import allure
import pytest

from ecg_interpreter.ecg.risk_analyser import risk_score
from ecg_interpreter.models.ecg_data import STSegment


@allure.feature("Анализ риска")
@allure.story("risk_score")
class TestRiskScore:

    # Ожидания по формуле из docstring: (база по срочности + компонент ЧСС) * множитель по диагнозу
    @allure.title("Показатель риска: {reading_fixture}")
    @pytest.mark.parametrize(
        ("reading_fixture", "expected"),
        [
            pytest.param("normal_reading", (0.0 + 0.0) * 1.0, id="normal"),
            pytest.param("bradycardia_reading", (25.0 + 5.0) * 1.2, id="bradycardia"),
            pytest.param("tachycardia_reading", (25.0 + 5.0) * 1.2, id="tachycardia"),
            pytest.param("arrhythmia_reading", (25.0 + 0.0) * 1.4, id="arrhythmia"),
            pytest.param("previous_infarction_reading", (25.0 + 0.0) * 1.45, id="previous_infarction"),
            pytest.param("ischemia_reading", (25.0 + 0.0) * 1.5, id="ischemia"),
            pytest.param("suspected_infarction_reading", (80.0 + 0.0) * 1.8, id="suspected_infarction"),
            pytest.param("infarction_reading", (80.0 + 0.0) * 2.0, id="infarction"),
        ],
    )
    def test_risk_score_by_diagnosis(self, request, reading_fixture, expected):
        reading = request.getfixturevalue(reading_fixture)

        result = risk_score(reading)

        assert result == pytest.approx(expected), (
            f"Для {reading} ожидался риск {expected} по формуле "
            f"(база + компонент ЧСС) * множитель, получено {result}"
        )

    @allure.title("Показатель риска при патологии ЧСС и приоритетном диагнозе")
    def test_risk_score_infarction_with_bradycardia(self, make_reading):
        reading = make_reading(st_segment=STSegment.ELEVATION, q_wave=True, heart_rate=50.0)
        expected = (80.0 + 5.0) * 2.0

        result = risk_score(reading)

        assert result == pytest.approx(expected), f"Для {reading} ожидался риск {expected}, получено {result}"

    @allure.title("None вместо данных ЭКГ вызывает ValueError")
    def test_risk_score_none_raises_error(self):
        with pytest.raises(ValueError, match="Данные ЭКГ не могут быть пустыми"):
            risk_score(None)

    @allure.title("Некорректная ЧСС вызывает ValueError даже при инфаркте: {heart_rate}")
    @pytest.mark.parametrize(
        ("heart_rate", "message"),
        [
            pytest.param(None, "Значение ЧСС не может быть пустыми", id="none"),
            pytest.param(-1.0, "Значение ЧСС не может быть отрицательным", id="negative"),
            pytest.param(300.001, "Значение ЧСС физиологически невозможно", id="above_max"),
            pytest.param(float("nan"), "Значение ЧСС должно быть конечным числом", id="nan"),
        ],
    )
    def test_risk_score_invalid_heart_rate_raises_error(self, make_reading, heart_rate, message):
        reading = make_reading(st_segment=STSegment.ELEVATION, q_wave=True, heart_rate=heart_rate)

        with pytest.raises(ValueError, match=message):
            risk_score(reading)
