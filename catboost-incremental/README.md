# CatBoost Incremental Learning

<img src="https://img.shields.io/badge/python-3.9.2-blue.svg" alt="python 3.9.2">

Author: [Denis Burakov @deburky](https://github.com/deburky)

## introduction

This repo contains experiments with CatBoost incremental learning. The goal is to train a model on a large dataset that does not fit into memory. The dataset is split into chunks and the model is trained on each chunk. The model is then saved and loaded to continue training on the next chunk.

[Catboost training model for huge data(~22GB) with multiple chunks](https://stackoverflow.com/questions/47019215/catboost-training-model-for-huge-data22gb-with-multiple-chunks)

## How to use API

In a Jupyter notebook, run the following code:

```python
import asyncio
import httpx
import nest_asyncio

nest_asyncio.apply()

API_URL = "http://127.0.0.1:8000/predict"

inference_results = {'num_rows': [], 'inference_time': []}

async def async_test_inference(num_rows_list):
    async with httpx.AsyncClient() as client:
        tasks = [
            client.post(API_URL, json={"num_rows": num_rows})
            for num_rows in num_rows_list
        ]
        responses = await asyncio.gather(*tasks)

        for response, num_rows in zip(responses, num_rows_list):
            if response.status_code == 200:
                result = response.json()
                inference_results['num_rows'].append(num_rows)
                inference_results['inference_time'].append(result['inference_time'])
                print(f"Success for {num_rows} inference_time: {result['inference_time']}")
            else:
                print(f"Error for {num_rows} inference_time: {response.text}")

# Run async test
asyncio.run(async_test_inference([500_000, 1_000_000, 20_000_000, 30_000_000, 40_000_000, 50_000_000]))
```