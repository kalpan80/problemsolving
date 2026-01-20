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

acc_dict = {}
acc_dict['valid'] = 0
acc_dict['in_valid'] = 0

black_money = pd.read_csv('Big_Black_Money_Dataset.csv')
df = black_money.drop('Money Laundering Risk Score', axis=1)

rules = ''
with open('Rules.txt', 'r') as f:
    for line in f.readlines():
        rules += line

#print(rules)

@function_tool
def get_rules():
    return rules

rule_agent_tool = Agent(
        name="RuleAgent",
        model=MODEL,
        instructions="You are an Agent that fetches rules for AML.",
        tools=[get_rules],
    ).as_tool(tool_name='RuleAgentTool',
              tool_description='Agent that uses tools to fetch rules')

@function_tool
def read_data():
    return df.sample(10).to_json()

transaction_agent_tool = Agent(
        name="TransactionAgent",
        model=MODEL,
        instructions="You are an Agent that fetches transactions for AML.",
        tools=[read_data],
    ).as_tool(tool_name='TransactionAgentTool',
              tool_description='Agent that uses tools to fetch transaction data')

@function_tool
def alarm(transaction_data: str):
    print('High risk transaction data found ',transaction_data)

aml_agent_tool = Agent(
        name="AMLAgent",
        model=MODEL,
        instructions="You are an Agent that evaluates transactions depending on the rules in the context."
                     "Your risk rating should be either high, moderate or low.",
        output_type=Transaction
    ).as_tool(tool_name='AMLAgentTool',
              tool_description='Agent that evaluates transactions.')

aml_reviewer_agent_tool = Agent(
        name="AMLReviewerAgent",
        model=MODEL,
        instructions="You are an Agent that reviews the evaluated transactions depending on the rules in the context."
                     "If the evaluation is correct, the workflow proceeds further. Otherwise, the transaction is returned to the AMLAgent for re-evaluation.",
    ).as_tool(tool_name='AMLReviewerAgentTool',
              tool_description='Agent that reviews evaluated transactions.')

alarm_agent_tool = Agent(
        name="AlarmAgent",
        model=MODEL,
        instructions="You are an Agent that sends out alarms for high risk transactions",
        tools=[alarm],
    ).as_tool(tool_name='AlarmAgentTool',
              tool_description='Agent that uses tools to send out alarms')

@function_tool
def process_transaction(transaction_data: str):
    print('Processing transaction ',transaction_data)

processing_agent_tool = Agent(
        name="ProcessingAgent",
        model=MODEL,
        instructions="You are an Agent that processes low and moderate risk transactions",
        tools=[process_transaction],
    ).as_tool(tool_name='ProcessingAgentTool',
              tool_description='Agent that uses tools to process transactions')

planning_agent = Agent(
        name="PlanningAgent",
        model=MODEL,
        instructions="You are a orchestrating agent that leverages sub-agents to detect anti money laundering transactions. You only need to use rules outlined to classify the transaction, do not apply any other knowledge."
                     "Avoid any recommendations, follow-up questions or suggestions on the transactions. Ensure that all inter agent communication is in JSON format."
                     "The RuleAgent contains the rules for flagging a transaction as high-risk, moderate-risk or low-risk."
                     "The TransactionAgent will send incoming transactions for analysis."
                     "The AMLAgent will evaluate the transaction, and provides justification for transaction rating. The justification should refer to the applied rules."
                     "The AMLReviewerAgent will review the evaluation and justification provided."
                     "The AlarmAgent will send out notifications if any transaction is flagged high-risk."
                     "The ProcessingAgent will process the low-risk and moderate-risk transactions."
                     "Provide justification for each processed transaction in JSON format.",
        tools=[rule_agent_tool,transaction_agent_tool,aml_agent_tool,aml_reviewer_agent_tool,alarm_agent_tool,processing_agent_tool]
)


async def execute():
    result = Runner.run_streamed(starting_agent=planning_agent,
                                 input="Initiate the process to analyze transactions "
                                       "and invoke appropriate agents to proccess the data.",
                                 max_turns=50)

    print("=== Run starting ===")

    # Iterate through the event stream to capture and display events
    async for event in result.stream_events():
        logging.info(event)
        if event.type == "raw_response_event":
            if event.data.type == "response.output_text.done":
                print(event.data.text)
        if event.type == 'run_item_stream_event':
            if event.name == 'tool_called' and event.item.raw_item.status == 'completed':
                print(event.item.raw_item.arguments)
            if event.name == 'tool_output':
                print(event.item.output)

    print("\n=== Run complete ===")
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(execute())