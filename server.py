import io
import logging
from typing import List
import torch

from PIL import Image
from fastapi import FastAPI, Form
from fastapi import File, UploadFile
from pydantic import BaseModel

log = logging.getLogger("uvicorn")
app = FastAPI()

from lang_sam.models.gdino import GDINO

gdino = GDINO()
gdino.build_model()


class PanelSegmentationParameters(BaseModel):
    box_threshold: float
    text_threshold: float
    text_prompt: str
    min_width: float
    max_width: float
    min_height: float
    max_height: float
    avg_color_darkness_limit: float


class BoundingBoxModel(BaseModel):
    center_x: float
    center_y: float
    width: float
    height: float


@app.post("/segment")
async def segment_image(parameters: str = Form(...), photos: List[UploadFile] = File(...)) -> List[List[BoundingBoxModel]]:
    model_params = PanelSegmentationParameters.model_validate_json(parameters)

    images = []
    for photo in photos:
        content = await photo.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")
        images.append(image)

    torch.cuda.empty_cache()

    gdino_results = gdino.predict(
        images,
        [model_params.text_prompt] * len(images),
        model_params.box_threshold,
        model_params.text_threshold,
    )

    torch.cuda.empty_cache()

    images_bboxes = []
    for idx, image_result in enumerate(gdino_results):
        bboxes = []
        if image_result["labels"]:
            image_result["boxes"] = image_result["boxes"].cpu().numpy()

            for box in image_result["boxes"]:
                box_instance = BoundingBoxModel(
                    center_x=box[0] + (box[2] - box[0]) / 2,
                    center_y=box[1] + (box[3] - box[1]) / 2,
                    width=box[2] - box[0],
                    height=box[3] - box[1],
                )
                bboxes.append(box_instance)

        images_bboxes.append(bboxes)

    return images_bboxes
