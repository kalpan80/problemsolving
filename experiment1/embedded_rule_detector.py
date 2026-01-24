import logging

from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='aml.log')

from dotenv import load_dotenv
from agents import Agent, Runner, function_tool
import asyncio
import json

load_dotenv()

MODEL = 'gpt-5-nano'

import pandas as pd

class Transaction(BaseModel):
    transaction_id: str
    risk_rating: str
    justification: str

black_money = pd.read_csv('Big_Black_Money_Dataset.csv')
df = black_money.drop(['Money Laundering Risk Score','Financial Institution','Person Involved'], axis=1)

def read_data():
    return df.sample(50).to_json(orient='records',index=False)


import pandas as pd

def classify_transaction_risk(transaction_data):
    """
    Applies money laundering risk rules to a single transaction row.
    Returns a dictionary with the Risk Level and the Suggested Score Range.
    """
    # 1. High-Risk Rules (Score 8-10)
    for row in transaction_data:
        high_risk_industries = ["Arms Trade", "Casinos"]
        high_risk_countries = ["Russia", "UAE"]
        tax_havens = ["Cayman Islands", "Bahamas", "Panama"]

        rule_1_1 = (row['Source of Money'] == 'Illegal') and (row['Reported by Authority'] == False)
        rule_1_2 = row['Shell Companies Involved'] >= 7
        rule_1_3 = (row['Country'] in high_risk_countries) and (row['Tax Haven Country'] in tax_havens) and (
                    row['Transaction Type'] == 'Cryptocurrency')
        rule_1_4 = (row['Amount (USD)'] > 4000000) and (row['Industry'] in high_risk_industries) and (
                    row['Reported by Authority'] == False)

        if rule_1_1 or rule_1_2 or rule_1_3 or rule_1_4:
            print(row['Transaction ID'],{"Risk_Level": "High", "Suggested_Score": "8-10"})

        else:
            # 2. Medium-Risk Rules (Score 4-7)
            med_risk_industries = ["Real Estate", "Construction", "Oil & Gas", "Finance"]

            rule_2_1 = (row['Source of Money'] == 'Illegal') and (row['Reported by Authority'] == True)
            rule_2_2 = (3 <= row['Shell Companies Involved'] <= 6) and (row['Industry'] in ["Real Estate", "Construction"])
            rule_2_3 = (row['Transaction Type'] in ["Offshore Transfer", "Stocks Transfer"]) and (
                pd.notnull(row['Tax Haven Country']))
            rule_2_4 = (row['Industry'] in ["Oil & Gas", "Finance"]) and (row['Shell Companies Involved'] >= 2)

            if rule_2_1 or rule_2_2 or rule_2_3 or rule_2_4:
                print(row['Transaction ID'],{"Risk_Level": "Medium", "Suggested_Score": "4-7"})
            else:
                # 3. Low-Risk Rules (Score 1-3)
                # Defaulting to low risk if no high/medium flags are triggered
                print(row['Transaction ID'],{"Risk_Level": "Low", "Suggested_Score": "1-3"})

if __name__ == "__main__":
    data = read_data()
    data = json.loads(data)
    classify_transaction_risk(data)
