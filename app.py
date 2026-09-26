import joblib
import pandas as pd

# Load trained ML model
model = joblib.load("loan_approval_model.pkl")

print("Loan Risk Assessment Model Loaded Successfully!")
print("Model is ready for prediction.")

# Exact feature names used during model training
feature_names = [
    'Age',
    'AnnualIncome',
    'CreditScore',
    'Experience',
    'LoanAmount',
    'LoanDuration',
    'NumberOfDependents',
    'MonthlyDebtPayments',
    'CreditCardUtilizationRate',
    'NumberOfOpenCreditLines',
    'NumberOfCreditInquiries',
    'DebtToIncomeRatio',
    'BankruptcyHistory',
    'PreviousLoanDefaults',
    'PaymentHistory',
    'LengthOfCreditHistory',
    'SavingsAccountBalance',
    'CheckingAccountBalance',
    'TotalAssets',
    'TotalLiabilities',
    'MonthlyIncome',
    'UtilityBillsPaymentHistory',
    'JobTenure',
    'NetWorth',
    'BaseInterestRate',
    'InterestRate',
    'MonthlyLoanPayment',
    'TotalDebtToIncomeRatio',
    'EmploymentStatus_Self-Employed',
    'EmploymentStatus_Unemployed',
    'EducationLevel_Bachelor',
    'EducationLevel_Doctorate',
    'EducationLevel_High School',
    'EducationLevel_Master',
    'MaritalStatus_Married',
    'MaritalStatus_Single',
    'MaritalStatus_Widowed',
    'HomeOwnershipStatus_Other',
    'HomeOwnershipStatus_Own',
    'HomeOwnershipStatus_Rent',
    'LoanPurpose_Debt Consolidation',
    'LoanPurpose_Education',
    'LoanPurpose_Home',
    'LoanPurpose_Other'
]

print("Number of features:", len(feature_names))


# Sample loan application
sample_data = [[
    30,       # Age
    60000,    # AnnualIncome
    700,      # CreditScore
    5,        # Experience
    20000,    # LoanAmount
    36,       # LoanDuration
    1,        # NumberOfDependents
    500,      # MonthlyDebtPayments
    20,       # CreditCardUtilizationRate
    5,        # NumberOfOpenCreditLines
    1,        # NumberOfCreditInquiries
    0.25,     # DebtToIncomeRatio
    0,        # BankruptcyHistory
    0,        # PreviousLoanDefaults
    90,       # PaymentHistory
    8,        # LengthOfCreditHistory
    10000,    # SavingsAccountBalance
    5000,     # CheckingAccountBalance
    100000,   # TotalAssets
    30000,    # TotalLiabilities
    5000,     # MonthlyIncome
    95,       # UtilityBillsPaymentHistory
    4,        # JobTenure
    70000,    # NetWorth
    5,        # BaseInterestRate
    6,        # InterestRate
    600,      # MonthlyLoanPayment
    0.30,     # TotalDebtToIncomeRatio

    0,        # EmploymentStatus_Self-Employed
    0,        # EmploymentStatus_Unemployed

    1,        # EducationLevel_Bachelor
    0,        # EducationLevel_Doctorate
    0,        # EducationLevel_High School
    0,        # EducationLevel_Master

    1,        # MaritalStatus_Married
    0,        # MaritalStatus_Single
    0,        # MaritalStatus_Widowed

    0,        # HomeOwnershipStatus_Other
    1,        # HomeOwnershipStatus_Own
    0,        # HomeOwnershipStatus_Rent

    1,        # LoanPurpose_Debt Consolidation
    0,        # LoanPurpose_Education
    0,        # LoanPurpose_Home
    0         # LoanPurpose_Other
]]


# Convert sample data into DataFrame
sample_df = pd.DataFrame(
    sample_data,
    columns=feature_names
)


# Make prediction
prediction = model.predict(sample_df)


# Display result
if prediction[0] == 1:
    print("Loan Approved")
else:
    print("Loan Not Approved")