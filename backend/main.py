from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from config import content_types, cache_folder_path

import image_retrieval
import image_description
import image_classify

app = FastAPI()
app.add_middleware( CORSMiddleware, allow_origins=['*'] )

@app.post("/upload")
def upload(request : Request, file_info: UploadFile = File(...)):
    if file_info.size == 0 or file_info.headers['content-type'] not in content_types:
       return {'message': f'Wront content-type: {file_info.headers["content-type"]}'}
    
    file = file_info.file
    
    if not file.readable():
        return {'message': f'Image "{file_info.filename}" is unreadable'}

    cached_image_path = cache_folder_path + file_info.filename
    with open(cached_image_path, 'wb') as cache_file:
        cache_file.write(file.read())

    file.close()

    paths = image_retrieval.get_similar_images(cached_image_path)
    description = image_description.generate_image_description(cached_image_path)
    classification = image_classify.predict_image(cached_image_path)

    requests = [str(request.base_url) + f'image?file_path=../{paths[i]}' for i in range(10)]

    return { 'image_requests': requests, 'description': description, 'category': classification }

@app.get("/image")
def image(file_path : str):
    return FileResponse(file_path)