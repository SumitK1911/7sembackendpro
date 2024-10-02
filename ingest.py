import os
import logging
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams
import uuid

logging.basicConfig(level=logging.INFO)

class VectorIngestor:
    VECTOR_SIZE = 512

    def __init__(self, image_folder, url="http://localhost:6333"):
        self.image_folder = image_folder
        self.url = url
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.client = QdrantClient(url=self.url, prefer_grpc=False)
        self.collection_name = "image_vectors"

    def extract_features(self, image_path):
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt")
            outputs = self.model.get_image_features(pixel_values=inputs['pixel_values'])
            features = outputs.detach().numpy().flatten()  # Flatten the features
            logging.info(f"Extracted features for {image_path} with shape: {features.shape}")
            return features
        except Exception as e:
            logging.error(f"Error extracting features from {image_path}: {e}")
            return None

    def create_vector_db(self, images):
        self._create_new_collection_if_not_exists()

        points = []
        for image in images:
            points.extend(self._process_image(image))

        self._upsert_points(points)

    def _create_new_collection_if_not_exists(self):
        try:
            collection_info = self.client.get_collection(collection_name=self.collection_name)
            if collection_info:
                logging.info(f"Collection {self.collection_name} already exists.")
                return
        except Exception as e:
            logging.info(f"Collection {self.collection_name} does not exist, creating a new one.")

        try:
            vector_params = VectorParams(size=self.VECTOR_SIZE, distance="Cosine")
            self.client.create_collection(collection_name=self.collection_name, vectors_config=vector_params)
            logging.info(f"Created new collection: {self.collection_name}")
        except Exception as e:
            logging.error(f"Error creating collection: {e}")

    def _process_image(self, image):
        points = []
        image_name = image['file_name']
        image_path = os.path.join(self.image_folder, image_name)
        if os.path.isfile(image_path):
            features = self.extract_features(image_path)
            if features is not None and features.shape[0] == self.VECTOR_SIZE:
                points.append({
                    "id": str(uuid.uuid4()),
                    "vector": features.tolist(),
                    "payload": {
                        "file_name": image_name,
                        "description": image['description'],
                        "price": image['price']
                    }
                })
                logging.info(f"Added {image_name} to points.")
            else:
                logging.warning(f"Skipping {image_name}: expected {self.VECTOR_SIZE} dimensions but got {features.shape[0] if features is not None else 'None'}")
        else:
            logging.warning(f"File {image_path} does not exist or is not a file.")
        return points

    def _upsert_points(self, points):
        try:
            if points:
                self.client.upsert(collection_name=self.collection_name, points=points)
                logging.info(f"Inserted {len(points)} points into the vector database.")
            else:
                logging.info("No valid points to upsert.")
        except Exception as e:
            logging.error(f"Error upserting points: {e}")

if __name__ == "__main__":
    image_folder = "./images"
    image_metadata = {
        "th.jpg": {
            "id": 1,
            "description": "A high-quality pink summer t-shirt, designed with a long cut and crafted from premium fabric for maximum comfort and durability. Price is $100",
            "price": 100.0
        },
        "thi.jpg": {
            "id": 2,
            "description": "A vibrant black summer t-shirt, featuring a long design and made from breathable fabric, ideal for casual outings and hot weather. Price is $100",
            "price": 100.0
        },
        "pant1.jpg": {
            "id": 3,
            "description": "A well-crafted pair of quality pants, offering a perfect fit and made from durable fabric for everyday wear and comfort. Price is $120",
            "price": 120.0
        },
        "pant.jpg": {
            "id": 4,
            "description": "Comfortable summer pants, made from elastic fabric, providing a relaxed fit and easy movement for warm weather activities. Price is $121",
            "price": 121.0
        },
        "womentshirt.jpg": {
            "id": 5,
            "description": "A stylish women t-shirt with a long cut, made from soft fabric and designed for both comfort and elegance, suitable for various occasions. Price is $140",
            "price": 140.0
        },
        "womentshirt1.jpg": {
            "id": 6,
            "description": "A blue summer t-shirt for women, featuring a long design and crafted from high-quality fabric, perfect for staying cool and fashionable. Price is $150",
            "price": 150.0
        },
        "trouser.jpg": {
            "id": 7,
            "description": "A comfortable black summer trouser, designed with a long cut and made from soft fabric, ideal for casual wear and everyday use. Price is $78",
            "price": 78.0
        },
        "trouser1.jpg": {
            "id": 8,
            "description": "A versatile blue summer trouser with a long fit, made from high-quality fabric for a stylish and comfortable look. Price is $72",
            "price": 72.0
        }
    }

    images = [{
        "file_name": file_name,
        "id": metadata["id"],
        "description": metadata["description"],
        "price": metadata["price"]
    } for file_name, metadata in image_metadata.items()]

    ingestor = VectorIngestor(image_folder)
    ingestor.create_vector_db(images)
