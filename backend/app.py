import os

import numpy as np
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel
from src.constants import MODEL_NAME, MODEL_VERSION, URL
from src.utils import connect_to_triton_inference_server
from tritonclient.grpc import service_pb2

grpc_stub, model_metadata, model_config = connect_to_triton_inference_server(
    url=URL,
    model_name=MODEL_NAME,
    model_version=MODEL_VERSION,
)

app = FastAPI()


class Data(BaseModel):
    data: str


def create_request(
    data: np.ndarray,
    input_name: str,
    output_name: str,
    model_name: str = MODEL_NAME,
    model_version: str = MODEL_VERSION,
) -> service_pb2.ModelInferRequest:
    request = service_pb2.ModelInferRequest(
        model_name=model_name,
        model_version=model_version,
    )

    input = service_pb2.ModelInferRequest().InferInputTensor(
        name=input_name,
        datatype="FP32",
        shape=[1, 64, 1],
    )

    output = service_pb2.ModelInferRequest().InferRequestedOutputTensor(
        name=output_name,
    )
    # output.parameters['classification'].int64_param = args.classes
    request.outputs.extend([output])

    # request.ClearField("inputs")
    # request.ClearField("raw_input_contents")
    input_bytes = data.tobytes()

    request.inputs.extend([input])
    request.raw_input_contents.extend([input_bytes])
    return request


def make_prediction(data: bytes) -> str:
    # data = np.asarray(data, dtype=np.float32).reshape(1, -1, 1)
    numpy_data = np.random.rand(1, 64, 1)
    request = create_request(
        data=numpy_data,
        model_name=model_metadata.name,
        model_version=model_metadata.versions[0],
        input_name=model_metadata.inputs[0].name,
        output_name=model_metadata.outputs[0].name,
    )

    # send request
    response = grpc_stub.ModelInfer(request)

    response = str(response)[:100]

    return response


# @app.get("/")
# def root() -> str:
#     return "This is the root entrypoint of our application."


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
            status_code=400, detail=f"Expected data of type bytearray, got {type(data)}"
        )
    return make_prediction(data)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
