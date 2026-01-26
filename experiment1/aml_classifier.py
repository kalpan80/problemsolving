# main.py
import asyncio
import json
from agents import Runner

from fastapi import FastAPI

from detector_review import server_planning_agent, df, Transaction, black_money

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
            acc_dict['valid'] += 1
        elif (ground_truth <= 7 and transaction_data.risk_rating.lower() == 'moderate' or transaction_data.risk_rating.lower() == 'low'):
            acc_dict['valid'] += 1
        else:
            print('Mismatch ', transaction_data.risk_rating.lower(), ground_truth)
            acc_dict['in_valid'] += 1

@app.get('/classify_transactions')
def classify_transactions():
    sample = df.sample(100).to_json(orient='records', index=False)
    records = json.loads(sample)
    acc_dict = {}
    for record in records:
        asyncio.run(classify(record,acc_dict))
    return {"message": "Transaction classification is complete"}