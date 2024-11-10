#Get one image from VQA Rad
#Apply transforms to match with PromptMRG
#Load PT CLIP model and pass image through it and get_image_feeatures()
#Load all embeddings in clip_text_features JSON
#Use consine similarity to get top 21 most similar

import orjson
from PIL import Image
import torch
# from transformers import CLIPProcessor

VQA_RAD_ANNOTS_FILE = r"C:\Users\agupt\OneDrive\Documents\Purdue\Junior Year\Fall Semester\ECE 570\Course Project\ECE_570_Course_Project\data\vqa_rad\VQA_RAD Dataset Public.json"
CLIP_TEXT_FEATURES_FILE = r"C:\Users\agupt\OneDrive\Documents\Purdue\Junior Year\Fall Semester\ECE 570\Course Project\ECE_570_Course_Project\data\mimic_cxr\clip_text_features.json"
CLIP_MODEL_FILE = r"C:\Users\agupt\OneDrive\Documents\Purdue\Junior Year\Fall Semester\ECE 570\Course Project\ECE_570_Course_Project\data\mimic_cxr\clip-imp-pretrained_128_6_after_4.pt"
IMAGE_FILE = r"C:\Users\agupt\OneDrive\Documents\Purdue\Junior Year\Fall Semester\ECE 570\Course Project\ECE_570_Course_Project\data\vqa_rad\synpic29265.jpg"

def test_clip_features():
    # Load large JSON file with orjson
    with open(CLIP_TEXT_FEATURES_FILE, 'rb') as f:
        data = orjson.loads(f.read())

    print(len(data))
    # sum = 0
    # for val in data:

    #     for entry in val:
    #         sum += entry
        
    #     print(sum)
    #     sum = 0

# def get_chest_images():
#     with open(VQA_RAD_ANNOTS_FILE, "rb") as f:
#         annotations = orjson.loads(f.read())
    
#     #First CHEX image
#     print(annotations[1])

#     clip_model = torch.load(CLIP_MODEL_FILE)
#     clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

#     image = Image.open(IMAGE_FILE)
#     model_input = clip_processor(images=image, return_tensors="pt")
    
#     clip_model.eval()
#     with torch.no_grad():
#         image_features = clip_model.get_image_features(**model_input)
    
#     print(image_features.shape())



if __name__ == "__main__":
    test_clip_features()
    # get_chest_images()
