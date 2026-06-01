from pydantic import BaseModel, Field
from typing import Literal


class JobFeatures(BaseModel):
    experience_level: Literal['EN', 'MI', 'SE', 'EX'] = Field(
        description="EN=Entry, MI=Mid, SE=Senior, EX=Executive"
    )
    employment_type: Literal['FT', 'PT', 'CT', 'FL'] = Field(
        description="FT=Full-time, PT=Part-time, CT=Contract, FL=Freelance"
    )
    remote_ratio: Literal[0, 50, 100] = Field(
        description="0=No remote, 50=Hybrid, 100=Full remote"
    )
    company_size: Literal['S', 'M', 'L'] = Field(
        description="S=Small, M=Medium, L=Large"
    )
    employee_residence: str = Field(
        description="Country code, e.g. US, GB, CA"
    )
    company_location: Literal['US', 'GB', 'CA', 'ES', 'DE'] = Field(
        description="Country code of company"
    )
    work_year: int = Field(
        default=2024,
        description="Year of work"
    )


class PredictionResponse(BaseModel):
    predicted_salary_usd: float
    experience_level: str
    company_location: str
    model_version: str