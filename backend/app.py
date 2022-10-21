import os

import numpy as np
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from src.constants import MODEL_NAME, MODEL_VERSION, URL
from src.utils import (
    connect_to_triton_inference_server,
    ogg_opus_bytes_to_numpy_array,
    postprocess,
    preprocess,
)
from tritonclient.grpc import service_pb2

grpc_stub, model_metadata, model_config = connect_to_triton_inference_server(
    url=URL,
    model_name=MODEL_NAME,
    model_version=MODEL_VERSION,
)

app = FastAPI()


def create_request(data: np.ndarray) -> service_pb2.ModelInferRequest:
    request_input = service_pb2.ModelInferRequest().InferInputTensor(
        name=model_metadata.inputs[0].name,
        datatype=model_metadata.inputs[0].datatype,
        shape=list(data.shape),
    )

    request_output = service_pb2.ModelInferRequest().InferRequestedOutputTensor(
        name=model_metadata.outputs[0].name
    )

    request = service_pb2.ModelInferRequest(
        model_name=model_metadata.name,
        model_version=model_metadata.versions[0],
        inputs=[request_input],
        outputs=[request_output],
        raw_input_contents=[data.tobytes()],
    )

    return request


def make_transcription(data: bytes) -> str:
    numpy_data = ogg_opus_bytes_to_numpy_array(data)
    request_data = preprocess(numpy_data)
    request = create_request(request_data)

    response = grpc_stub.ModelInfer(request)
    response = postprocess(response)

    return response


@app.get("/health")
def health() -> bool:
    return model_config is not None and model_metadata is not None


async def parse_body(request: Request) -> bytes:
    data: bytes = await request.body()
    return data


@app.post("/predict")
async def predict(data: bytes = Depends(parse_body)) -> str:
    if not isinstance(data, bytes):
        raise HTTPException(
            status_code=400, detail=f"Expected data of type bytes, got {type(data)}"
        )
    return make_transcription(data)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
