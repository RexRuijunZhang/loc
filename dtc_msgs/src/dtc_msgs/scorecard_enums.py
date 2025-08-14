from enums import Enum

class SevereHemorrhage(Enum):
    PRESENT = 1
    ABSENT = 0

class RespiratoryDistress(Enum):
    PRESENT = 1
    ABSENT = 0

class TraumaHead(Enum):
    NORMAL = 0
    WOUND = 1
    NT = 2

class TraumaTorso(TraumaHead):
    pass

class TraumaUpperExtremity(Enum):
    NORMAL = 0
    WOUND = 1
    NT = 2
    AMPUTATION = 3

class TraumaLowerExtremity(TraumaUpperExtremity):
    pass

class AlertnessOcular(Enum):
    OPEN = 0
    CLOSED = 1
    NT = 3

class AlertnessVerbal(Enum):
    NORMAL = 0
    ABNORMAL = 1
    ABSENT = 2
    NT = 3

class AlertnessMotor(AlertnessVerbal):
    pass
