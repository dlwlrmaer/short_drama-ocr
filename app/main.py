from io import BytesIO

import pytesseract
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

app = FastAPI(title="NAS OCR API", version="1.0.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ocr")
async def ocr(file: UploadFile = File(...)) -> dict[str, str]:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="请上传图片文件")

    try:
        image = Image.open(BytesIO(await file.read()))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="无法解析图片") from exc

    text = pytesseract.image_to_string(image, lang="chi_sim+eng")
    return {"filename": file.filename or "image", "text": text.strip()}
