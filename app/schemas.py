from pydantic import BaseModel

class PredictionRequest(BaseModel):
    sg: float
    htn: int
    hemo: float
    dm: int
    al: int
    appet: int
    rc: float
    pc: int
    model: str
