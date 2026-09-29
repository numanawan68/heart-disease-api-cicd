from pydantic import BaseModel


class HeartDiseaseInput(BaseModel):
    male: str
    age: float
    currentSmoker: str
    cigsPerDay: float
    BPMeds: str
    prevalentStroke: str
    prevalentHyp: str
    diabetes: str
    totChol: float
    sysBP: float
    diaBP: float
    BMI: float
    heartRate: float
    glucose: float