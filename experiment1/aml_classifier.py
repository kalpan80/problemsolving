# main.py
import asyncio
import json
from agents import Runner

from fastapi import FastAPI

from aml_agents import server_planning_agent, df, Transaction, black_money

# Create a FastAPI app instance
app = FastAPI()

import logging

# Define a GET endpoint
@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

async def classify(record,acc_dict):
    result = Runner.run_streamed(starting_agent=server_planning_agent,
                                 input=json.dumps(record),
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

    if (result.is_complete):
        print("\n=== Run complete ===")
        print(result.final_output)
        transaction_data = result.final_output_as(Transaction)
        ground_truth = black_money[black_money['Transaction ID'] == transaction_data.transaction_id]['Money Laundering Risk Score'].item()
        if (ground_truth > 7 and transaction_data.risk_rating.lower() == 'high'):
            acc_dict['TP'] += 1
        elif (ground_truth <= 7 and transaction_data.risk_rating.lower() == 'moderate' or transaction_data.risk_rating.lower() == 'low'):
            acc_dict['TN'] += 1
        else:
            if (ground_truth > 7 and transaction_data.risk_rating.lower() == 'moderate' or transaction_data.risk_rating.lower() == 'low'):
                acc_dict['FN'] += 1
            else:
                acc_dict['FP'] += 1
        return transaction_data

@app.get('/classify_transactions')
def classify_transactions():
    sample = df.sample(10).to_json(orient='records', index=False)
    records = json.loads(sample)
    acc_dict = {}
    acc_dict['TP'] = 0
    acc_dict['TN'] = 0
    acc_dict['FP'] = 0
    acc_dict['FN'] = 0
    batch_results = []
    for record in records:
        result = asyncio.run(classify(record, acc_dict))
        if result is not None:
            batch_results.append(result)
    print(acc_dict)
    return {"message": "Transaction classification is complete"}