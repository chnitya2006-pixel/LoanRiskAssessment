import joblib
import pandas as pd

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button


# Load trained ML model
model = joblib.load("loan_approval_model.pkl")


class LoanRiskApp(App):

    def build(self):

        layout = BoxLayout(
            orientation="vertical",
            padding=20,
            spacing=10
        )

        title = Label(
            text="INTELLIGENT LOAN RISK ASSESSMENT",
            font_size="22sp",
            size_hint_y=None,
            height=60
        )
        layout.add_widget(title)

        self.age = TextInput(
            hint_text="Enter Age",
            multiline=False,
            input_filter="float"
        )
        layout.add_widget(self.age)

        self.income = TextInput(
            hint_text="Enter Annual Income",
            multiline=False,
            input_filter="float"
        )
        layout.add_widget(self.income)

        self.credit_score = TextInput(
            hint_text="Enter Credit Score",
            multiline=False,
            input_filter="float"
        )
        layout.add_widget(self.credit_score)

        self.loan_amount = TextInput(
            hint_text="Enter Loan Amount",
            multiline=False,
            input_filter="float"
        )
        layout.add_widget(self.loan_amount)

        self.loan_duration = TextInput(
            hint_text="Enter Loan Duration (months)",
            multiline=False,
            input_filter="float"
        )
        layout.add_widget(self.loan_duration)

        button = Button(
            text="CHECK LOAN RISK",
            size_hint_y=None,
            height=55
        )

        button.bind(on_press=self.predict_loan)
        layout.add_widget(button)

        self.result = Label(
            text="Enter details and check loan risk",
            font_size="18sp"
        )
        layout.add_widget(self.result)

        return layout

    def predict_loan(self, instance):

        try:

            # Get user input
            age = float(self.age.text)
            income = float(self.income.text)
            credit_score = float(self.credit_score.text)
            loan_amount = float(self.loan_amount.text)
            loan_duration = float(self.loan_duration.text)

            # Exact 44 features used during model training
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

            # Create input data
            data = [[
                age,
                income,
                credit_score,
                5,
                loan_amount,
                loan_duration,
                1,
                500,
                20,
                5,
                1,
                0.25,
                0,
                0,
                90,
                8,
                10000,
                5000,
                100000,
                30000,
                income / 12,
                95,
                4,
                70000,
                5,
                6,
                600,
                0.30,
                0,
                0,
                1,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                1,
                0,
                1,
                0,
                0,
                0
            ]]

            # Convert to DataFrame
            input_data = pd.DataFrame(
                data,
                columns=feature_names
            )

            # Make prediction
            prediction = model.predict(input_data)

            # Display result
            if prediction[0] == 1:
                self.result.text = "LOAN APPROVED"
            else:
                self.result.text = "LOAN NOT APPROVED"

        except Exception as e:

            self.result.text = "Error: " + str(e)
            print("ERROR:", e)


if __name__ == "__main__":
    LoanRiskApp().run()