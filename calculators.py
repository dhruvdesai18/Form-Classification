from pydantic import BaseModel, Field, field_validator
from typing import Optional
#pydanctic is used for data validation and settings management using Python type annotations 

class EligibilityData(BaseModel):
    income: float = Field(..., description="Annual income of the applicant")
    loan_amount: float = Field(..., description="Requested loan amount")
    tenure: int = Field(..., description="Loan tenure in months")
    property_value: float = Field(..., description="Value of the property being mortgaged")

    #validator to check if income is positive
    @field_validator('tenure')
    def tenure_limit(cls, v):
        if v > 30:
            raise ValueError('Tenure cannot exceed 30 years')
        return v

    

def calculate_emi(loan_amount: float, tenure: int, annual_rate: float = 8.5) -> float:
    """Calculate the Equated Monthly Installment (EMI) for a loan."""

    monthly_rate = annual_rate / (12 * 100)  # Convert annual rate to monthly and percentage to decimal
    months = tenure * 12  # Convert years to months
    emi = (loan_amount * monthly_rate * (1 + monthly_rate) ** months) / ((1 + monthly_rate) ** months - 1)
    return emi


def calculate_foir(emi: float, income: float) -> float:
    """FOIR = (Total EMI / Monthly Income) * 100"""
    return (emi / (income / 12)) * 100  # Convert annual income to monthly income


def calculate_ltv(loan_amount: float, property_value: float) -> float:
    """LTV = (Loan Amount / Property Value) * 100"""
    return (loan_amount / property_value) * 100  # Convert to percentage




def evaluate_eligibility(data: EligibilityData) -> dict:
    """Helper to calculate all function and return a dictionary with results."""
    emi = calculate_emi(data.loan_amount, data.tenure)
    foir = calculate_foir(emi, data.income)
    ltv = calculate_ltv(data.loan_amount, data.property_value)

    return {
        "emi": emi,
        "foir": foir,
        "ltv": ltv
    }