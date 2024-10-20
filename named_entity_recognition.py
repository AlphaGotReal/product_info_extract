from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
import matplotlib.pyplot as plt
from transformers import pipeline

ocr = PaddleOCR(use_angle_cls=True, lang='en')

img_path = 'finaltest.jpeg'
result = ocr.ocr(img_path, cls=True)

result = result[0]
image = Image.open(img_path).convert('RGB')
boxes = [line[0] for line in result]
txts = [line[1][0] for line in result]

extracted_text = txts

print(extracted_text)

ner_model = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

ner_results = ner_model(extracted_text)
print("\nExtracted Entities:")
for entity in ner_results:
    print(f"Entity: {entity['word']}, Label: {entity['entity']}, Score: {entity['score']}")


font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'

image = Image.open(img_path).convert('RGB')
boxes = [line[0] for line in result[0]]
txts = [line[1][0] for line in result[0] if isinstance(line[1][0], str)]
scores = [line[1][1] for line in result[0] if isinstance(line[1][1], float)]

im_show = draw_ocr(image, boxes, txts, scores, font_path=font_path)
im_show = Image.fromarray(im_show)

plt.imshow(im_show)
plt.axis('off')
plt.show()

im_show.save('result.jpg')
