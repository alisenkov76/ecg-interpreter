import pytest

from ecg_interpreter.models.ecg_data import ECGData, Rhythm, STSegment


@pytest.fixture
def make_reading():
    def _make(
        *,
        heart_rate: float = 75.0,
        rhythm: Rhythm = Rhythm.REGULAR,
        st_segment: STSegment = STSegment.NORMAL,
        q_wave: bool = False,
    ) -> ECGData:
        return ECGData(
            heart_rate=heart_rate,
            rhythm=rhythm,
            st_segment=st_segment,
            q_wave=q_wave,
        )

    return _make


@pytest.fixture
def normal_reading(make_reading):
    return make_reading()


@pytest.fixture
def infarction_reading(make_reading):
    return make_reading(st_segment=STSegment.ELEVATION, q_wave=True)


@pytest.fixture
def suspected_infarction_reading(make_reading):
    return make_reading(st_segment=STSegment.ELEVATION)


@pytest.fixture
def ischemia_reading(make_reading):
    return make_reading(st_segment=STSegment.DEPRESSION)


@pytest.fixture
def arrhythmia_reading(make_reading):
    return make_reading(rhythm=Rhythm.IRREGULAR)


@pytest.fixture
def previous_infarction_reading(make_reading):
    return make_reading(q_wave=True)


@pytest.fixture
def bradycardia_reading(make_reading):
    return make_reading(heart_rate=50.0)


@pytest.fixture
def tachycardia_reading(make_reading):
    return make_reading(heart_rate=120.0)


@pytest.fixture
def make_dataset(make_reading):
    def _make(heart_rates: list[float]) -> list[ECGData]:
        return [make_reading(heart_rate=hr) for hr in heart_rates]

    return _make
