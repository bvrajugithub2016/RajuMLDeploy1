from pydantic import BaseModel
#enforces type hints at runtime and provides user friendly error messages

#class which represents BankNote features/measurements
class BankNote(BaseModel):
    variance: float
    skewness: float
    curtosis: float
    entropy: float