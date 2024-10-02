from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import os
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from ingest import VectorIngestor
from app import AIVoiceAssistant
import speech_recognition as sr
from fastapi.responses import JSONResponse
import google.generativeai as genai
import json
import uuid
import logging
import requests
from urllib.parse import urlencode
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  
    
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/images", StaticFiles(directory="images"), name="images")

image_folder = "./images"
ingestor = VectorIngestor(image_folder)

vector_db_url = "http://localhost:6333"
api_key = os.getenv("API_KEY")
if not api_key:
    raise RuntimeError("API_KEY environment variable not set.")
assistant = AIVoiceAssistant(vector_db_url, api_key)

cart = []

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_message(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws/cart")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await manager.send_message(data)
    except WebSocketDisconnect:
        manager.disconnect(websocket)

class QueryRequest(BaseModel):
    query_text: str

class CartItem(BaseModel):
    id: str
    description: str
    price: float
    quantity: int = 1

class RemoveItemRequest(BaseModel):
    description: str

@app.get("/")
async def read_root():
    return {"message": "Welcome to the AI Voice Assistance API"}

@app.post("/ingest/")
async def ingest_images(files: List[UploadFile] = File(...), descriptions: List[str] = None, prices: List[float] = None):
    image_metadata = []
    images = []

    for idx, file in enumerate(files):
        file_path = os.path.join(image_folder, file.filename)
        with open(file_path, 'wb') as buffer:
            buffer.write(file.file.read())
        
        description = descriptions[idx] if descriptions and len(descriptions) > idx else "No description available."
        price = prices[idx] if prices and len(prices) > idx else 0.0

        metadata = {
            "id": str(uuid.uuid4()),
            "description": description,
            "price": price
        }
        
        images.append({
            "id": metadata["id"],
            "file_name": file.filename,
            "description": metadata["description"],
            "price": metadata["price"]
        })

    try:
        ingestor.create_vector_db(images)
    except Exception as e:
        logging.error(f"Error during image ingestion: {e}")
        raise HTTPException(status_code=500, detail="Failed to ingest images")

    return {"message": "Images ingested successfully", "images": images}

async def query_images(request: QueryRequest):
    search_results = assistant.query_vector_db(request.query_text)

@app.post("/query/")
async def query_images(request: QueryRequest):
    search_results = assistant.query_vector_db(request.query_text)

    filtered_images = []
    for result in search_results:
        payload = result.payload
        id = payload.get("id", "unknown_id")
        file_name = payload.get("file_name", "unknown.jpg")
        description = payload.get("description", "No description")
        price = payload.get("price", 0.0)

        filtered_images.append({"id": id, "filename": file_name, "description": description, "price": price})

    add_to_cart_item = filtered_images[0] if filtered_images else None
    deleted_from_cart = None
    updated_cart_item = None
    action_result = "no_action"
    payment_url=None

    if "add to cart" in request.query_text.lower():
        if filtered_images:
            add_to_cart_item = filtered_images[0]
            cart_item = CartItem(
                id=add_to_cart_item["id"],
                description=add_to_cart_item["description"],
                price=add_to_cart_item["price"],
                quantity=1,
            )
            cart.append(cart_item)
            action_result = "added_to_cart"

            notification = {"action": "add", "item": cart_item.model_dump()}
            await manager.send_message(json.dumps(notification))

    elif "delete from cart" in request.query_text.lower():
        if filtered_images:
            deleted_from_cart = filtered_images[0]
            await remove_item_from_cart_internal(deleted_from_cart["description"])
            action_result = "deleted_from_cart"

            notification = {"action": "remove", "item": deleted_from_cart["description"]}
            await manager.send_message(json.dumps(notification))

    elif "update cart" in request.query_text.lower():
        if filtered_images:
            updated_cart_item = filtered_images[0]
            cart_item = CartItem(
                id=str(updated_cart_item["id"]),
                description=updated_cart_item["description"],
                price=updated_cart_item["price"],
                quantity=1,  
            )
            await edit_cart_item(cart_item)
            action_result = "updated_cart"

            notification = {"action": "edit", "item": cart_item.model_dump()}
            await manager.send_message(json.dumps(notification))
    elif "proceed to check out" in request.query_text.lower():
     if cart:
        total_amount = sum(item.price * item.quantity for item in cart)
        
        if discount_state["bargaining_attempts"] > 0:
            discount = discount_state.get("current_discount", 0)  
            total_amount *= (1 - discount / 100)  
        
        product_ids = ', '.join(item.id for item in cart)
        payment_request = PaymentRequest(amount=total_amount, product_id=product_ids)
        
        payment_response = await esewa_payment(payment_request)
        payment_url = payment_response.get('url')  
        
        if payment_url:
            action_result = "proceed_to_checkout"
            notification = {"action": "checkout", "url": payment_url}
            await manager.send_message(json.dumps(notification))
            cart.clear()
        else:
            action_result = "payment_failed"

    
    elif "provide me discount" in request.query_text.lower():
        discount_state["bargaining_attempts"] += 1
        new_discount = handle_bargaining()

        if discount_state["bargaining_attempts"]> 0 and discount_state["current_discount"]<discount_state["max_discount"]:
          discount_state["current_discount"] +=2
          discount_state["current_discount"] = new_discount 

        total_amount = sum(item.price * item.quantity for item in cart)
        discounted_amount = total_amount * (1 - new_discount / 100)


        action_result = "discount_applied"
        notification = {"action": "discount", "discount": new_discount, "final_amount": discounted_amount}
        await manager.send_message(json.dumps(notification))



    response = assistant.generate_response(request.query_text, action_result)

    return {
        "response": response,
        "images": filtered_images,
        "addToCart": add_to_cart_item if action_result == "added_to_cart" else None,
        "deleteFromCart": deleted_from_cart if action_result == "deleted_from_cart" else None,
        "updateCart": updated_cart_item if action_result == "updated_cart" else None,
        "paymentUrl": payment_url if action_result == "proceed_to_checkout" else None,
        "discountApplied": discounted_amount if action_result == "discount_applied" else None,
    }

async def remove_item_from_cart_internal(description: str):
    global cart
    cart = [item for item in cart if item.description.lower() != description.lower()]

@app.post("/voice-query")
async def voice_query(file: UploadFile = File(...)):
    recognizer = sr.Recognizer()
    audio_file_path = os.path.join("audio", file.filename)

    try:
        with open(audio_file_path, "wb") as buffer:
            buffer.write(file.file.read())

        with sr.AudioFile(audio_file_path) as source:
            audio = recognizer.record(source)
            try:
                query_text = recognizer.recognize_google(audio)
                response = assistant.handle_user_query(query_text)
               
                return {"response": response, "query_text": query_text}
            except sr.UnknownValueError:
                raise HTTPException(status_code=400, detail="Could not understand the audio.")
            except sr.RequestError as e:
                raise HTTPException(status_code=500, detail=f"Could not request results from the speech recognition service; {e}")
    finally:
        if os.path.exists(audio_file_path):
            os.remove(audio_file_path)

@app.get("/cart/")
async def get_cart():
    return {"cart": [item.dict() for item in cart]}

@app.post("/cart/add")
async def add_cart_item(item: CartItem):
    global cart
    existing_item = next((i for i in cart if i.description.lower() == item.description.lower()), None)
    if existing_item:
        existing_item.quantity += item.quantity
    else:
        cart.append(item)

    notification = {"action": "add", "item": item.model_dump()}
    await manager.send_message(json.dumps(notification))
    
    query_request = QueryRequest(query_text=f"Added {item.description} to cart")
    response = await query_images(query_request)

    return {"message": "Item added to cart", "ai_response": response}
    

@app.post("/cart/remove")
async def delete_item_from_cart(item: RemoveItemRequest):
    global cart
    cart = [i for i in cart if i.description.lower() != item.description.lower()]

    notification = {"action": "remove", "item": item.description}
    await manager.send_message(json.dumps(notification))

    query_request = QueryRequest(query_text=f"Removed {item.description} from cart")
    response = await query_images(query_request)

    return {"message": "Item removed from cart", "ai_response": response}

@app.post("/cart/edit/")
async def edit_cart_item(updated_item: CartItem):
    global cart
    found = False
    for item in cart:
        if item.description.lower() == updated_item.description.lower():
            item.quantity = updated_item.quantity
            found = True
            break
    if not found:
        cart.append(updated_item)

    notification = {"action": "edit", "item": updated_item.model_dump()}
    await manager.send_message(json.dumps(notification))

    query_request = QueryRequest(query_text=f"Removed {updated_item.description} from cart")
    response = await query_images(query_request)

    return {"message": "Item updated from cart", "ai_response": response}

class PaymentRequest(BaseModel):
    amount: float
    product_id: str


@app.post("/esewa-payment")
async def esewa_payment(payment: PaymentRequest):
    total_amount = payment.amount
    final_discount = discount_state["current_discount"]
    discounted_amount = total_amount * (1 - final_discount / 100)

    merchant_code = 'EPAYTEST'
    success_url = 'http://localhost:3000/payment-success'
    failure_url = 'http://localhost:3000/payment-failure'

    params = {
        'amt': payment.amount,  
        'pdc': 0,
        'psc': 0,
        'txAmt': 0,
        'tAmt': payment.amount,  
        'pid': payment.product_id,
        'scd': merchant_code,
        'su': success_url,
        'fu': failure_url
    }

    logging.info(f"Sending payment request to eSewa: {params}")
    esewa_payment_url = f"https://uat.esewa.com.np/epay/main?{urlencode(params)}"

    return {
        "url": esewa_payment_url,
        "discount_amount": total_amount - discounted_amount,
        "final_amount": discounted_amount,
    }


@app.post("/esewa-verify")
async def esewa_verify(amount: float, product_id: str, ref_id: str):
    merchant_code = 'EPAYTEST'  

    params = {
        'amt': amount,
        'rid': ref_id,  
        'pid': product_id,  
        'scd': merchant_code
    }

    esewa_verification_url = 'https://uat.esewa.com.np/epay/transrec'
    
    try:
        response = requests.post(esewa_verification_url, data=params)
        if 'Success' in response.text:
            return {"message": "Payment verified successfully"}
        else:
            raise HTTPException(status_code=400, detail="Payment verification failed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

discount_state={
    "current_discount":10,
    "max_discount":20,
    "bargaining_attempts": 0
}

def handle_bargaining():
    if discount_state["bargaining_attempts"]> 0 and discount_state["current_discount"]<discount_state["max_discount"]:
        discount_state["current_discount"] +=2
    return discount_state["current_discount"]    