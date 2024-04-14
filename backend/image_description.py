from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, T5ForConditionalGeneration, T5Tokenizer

device = 'cpu'

processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
model_blip = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base').to(device)

model_t5 = T5ForConditionalGeneration.from_pretrained('utrobinmv/t5_translate_en_ru_zh_small_1024').to(device)
tokenizer_t5 = T5Tokenizer.from_pretrained('utrobinmv/t5_translate_en_ru_zh_small_1024')

def generate_image_description(file_path: str):
    image = Image.open(file_path)

    inputs = processor(images=image, return_tensors="pt").to(device)
    outputs = model_blip.generate(**inputs, max_new_tokens=20)
    caption = processor.decode(outputs[0], skip_special_tokens=True)

    input_ids = tokenizer_t5('translate to ru: ' + caption, return_tensors="pt").input_ids.to(device)
    generated_tokens = model_t5.generate(input_ids)
    return tokenizer_t5.decode(generated_tokens[0], skip_special_tokens=True)