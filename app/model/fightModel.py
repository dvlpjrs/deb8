from pydantic import BaseModel, UUID4
from decouple import config


class FightModel(BaseModel):
    Model1: str
    Model2: str
    defaultQuestion: bool = False
    question: str = None
