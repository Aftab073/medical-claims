import os
import fitz
import base64
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

load_dotenv()

def test_ocr():
    pdf_path = r"C:\Users\tambo\OneDrive\Desktop\medical-claims\final_image_protected.pdf"
    if not os.path.exists(pdf_path):
        print(f"PDF not found at {pdf_path}")
        return
        
    doc = fitz.open(pdf_path)
    page = doc[0]  # first page
    
    # Try normal text extraction
    text = page.get_text("text").strip()
    print("Normal text length:", len(text))
    
    if len(text) < 30:
        print("Falling back to Vision OCR...")
        pix = page.get_pixmap(dpi=150)
        img_bytes = pix.tobytes("png")
        b64_img = base64.b64encode(img_bytes).decode("utf-8")
        
        llm = ChatGroq(model="llama-3.2-11b-vision-preview", temperature=0.0)
        msg = HumanMessage(content=[
            {"type": "text", "text": "Extract and transcribe all text from this image exactly as written. Provide ONLY the extracted text, no introductory remarks."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}}
        ])
        
        try:
            response = llm.invoke([msg])
            print("\n--- OCR Text ---\n")
            print(response.content)
            print("\n----------------\n")
        except Exception as e:
            print("Vision API failed:", e)

if __name__ == "__main__":
    test_ocr()
