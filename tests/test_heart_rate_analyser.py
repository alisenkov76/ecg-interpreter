import allure
import pytest

from ecg_interpreter.ecg.heart_rate_analyser import analyze_heart_rate, average_heart_rate, heart_rate_range
from ecg_interpreter.models.interpretation import Diagnosis


@allure.feature("Анализ ЧСС")
@allure.story("analyze_heart_rate")
class TestAnalyzeHeartRate:

    @allure.title("Классификация ЧСС {heart_rate} -> {expected}")
    @pytest.mark.parametrize(
        ("heart_rate", "expected"),
        [
            pytest.param(0, Diagnosis.BRADYCARDIA, id="0"),
            pytest.param(59.9, Diagnosis.BRADYCARDIA, id="59.9"),
            pytest.param(60, Diagnosis.NORMAL, id="60"),
            pytest.param(75, Diagnosis.NORMAL, id="75"),
            pytest.param(100, Diagnosis.NORMAL, id="100"),
            pytest.param(100.1, Diagnosis.TACHYCARDIA, id="100.1"),
            pytest.param(300, Diagnosis.TACHYCARDIA, id="300"),
        ],
    )
    def test_analyze_heart_rate_returns_diagnosis(self, heart_rate, expected):
        result = analyze_heart_rate(heart_rate)

        assert result == expected, f"Для ЧСС {heart_rate} ожидался диагноз {expected}, получено {result}"

    @allure.title("Некорректная ЧСС вызывает ValueError: {heart_rate}")
    @pytest.mark.parametrize(
        ("heart_rate", "message"),
        [
            pytest.param(None, "Значение ЧСС не может быть пустыми", id="none"),
            pytest.param(-1, "Значение ЧСС не может быть отрицательным", id="-1"),
            pytest.param(-0.001, "Значение ЧСС не может быть отрицательным", id="-0.001"),
            pytest.param(300.001, "Значение ЧСС физиологически невозможно", id="300.001"),
            pytest.param(float("nan"), "Значение ЧСС должно быть конечным числом", id="nan"),
            pytest.param(float("inf"), "Значение ЧСС физиологически невозможно", id="inf"),
            pytest.param(float("-inf"), "Значение ЧСС не может быть отрицательным", id="-inf"),
        ],
    )
    def test_analyze_heart_rate_invalid_value_raises_error(self, heart_rate, message):
        with pytest.raises(ValueError, match=message):
            analyze_heart_rate(heart_rate)


@allure.feature("Анализ ЧСС")
@allure.story("average_heart_rate")
class TestAverageHeartRate:

    @allure.title("Средняя ЧСС для {heart_rates}")
    @pytest.mark.parametrize(
        ("heart_rates", "expected"),
        [
            pytest.param([75.0], 75.0, id="single"),
            pytest.param([60.0, 100.0], 80.0, id="two"),
            pytest.param([50.0, 75.0, 120.0], 81.666667, id="three"),
            pytest.param([0.0, 0.0], 0.0, id="zeros"),
        ],
    )
    def test_average_heart_rate_returns_mean(self, make_dataset, heart_rates, expected):
        dataset = make_dataset(heart_rates)

        result = average_heart_rate(dataset)

        assert result == pytest.approx(expected), f"Для ЧСС {heart_rates} ожидалось среднее {expected}, получено {result}"

    @allure.title("Пустой набор вызывает ValueError: {dataset}")
    @pytest.mark.parametrize("dataset", [pytest.param([], id="empty"), pytest.param(None, id="none")])
    def test_average_heart_rate_empty_dataset_raises_error(self, dataset):
        with pytest.raises(ValueError, match="Список измерений не может быть пустым"):
            average_heart_rate(dataset)


@allure.feature("Анализ ЧСС")
@allure.story("heart_rate_range")
class TestHeartRateRange:

    @allure.title("Диапазон ЧСС для {heart_rates}")
    @pytest.mark.parametrize(
        ("heart_rates", "expected"),
        [
            pytest.param([75.0], (75.0, 75.0), id="single"),
            pytest.param([120.0, 50.0, 75.0], (50.0, 120.0), id="unsorted"),
            pytest.param([60.0, 60.0], (60.0, 60.0), id="equal"),
            pytest.param([0.0, 300.0], (0.0, 300.0), id="limits"),
        ],
    )
    def test_heart_rate_range_returns_min_and_max(self, make_dataset, heart_rates, expected):
        dataset = make_dataset(heart_rates)

        result = heart_rate_range(dataset)

        assert result == expected, f"Для ЧСС {heart_rates} ожидался диапазон {expected}, получено {result}"

    @allure.title("Пустой набор вызывает ValueError")
    def test_heart_rate_range_empty_dataset_raises_error(self):
        with pytest.raises(ValueError):
            heart_rate_range([])

    @allure.title("None вместо набора вызывает TypeError")
    def test_heart_rate_range_none_dataset_raises_error(self):
        with pytest.raises(TypeError):
            heart_rate_range(None)
