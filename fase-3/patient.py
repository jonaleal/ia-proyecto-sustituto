from pydantic import BaseModel


class Patient(BaseModel):
    """
    Representa un paciente con sus respectivos atributos.

    Atributos:
    - gender (str): Género del paciente (Female, Male, Other)
    - age (int): Edad del paciente
    - hypertension (int): Enfermedad hipertensión (1: si, 0: no)
    - heart_disease (int): Enfermedad cardíaca (1: si, 0: no)
    - smoking_history (str): Historial de fumado del paciente (not current, former, No Info, current, never, ever.)
    - bmi (float): Índice de masa corporal del paciente
    - HbA1c_level (float): Nivel de hemoglobina A1c del paciente
    - blood_glucose_level (int): Nivel de azúcar en la sangre del paciente
    """

    gender: str
    age: int
    hypertension: int
    heart_disease: int
    smoking_history: str
    bmi: float
    HbA1c_level: float
    blood_glucose_level: int
