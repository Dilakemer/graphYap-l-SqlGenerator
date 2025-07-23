import subprocess
import threading
import webbrowser
import time
 
# 1. .NET Core Projesini Başlat (VS2022 Açık Olmasına Gerek YOK!)
def run_dotnet():
    # C# projenin .csproj dosyasının bulunduğu klasör yolu
    subprocess.Popen(
        ["dotnet", "run"],
        cwd=r"C:\newfile\stajdeneme",  # KENDİ PROJE YOLUNU KONTROL ET!
        shell=True
    )
 
# 2. Tarayıcıda Ana Sayfayı Aç (Portun doğru olduğundan emin ol!)
def open_browser():
    time.sleep(3)  # Dotnet'in açılması için kısa bekleme (gerekirse arttır)
    webbrowser.open("http://localhost:5296")  # Projenin kullandığı portu yaz
 
# 3. .NET Core'u thread ile başlat (bloklamasın)
threading.Thread(target=run_dotnet, daemon=True).start()
 
# 4. Ana sayfa için browser aç
threading.Thread(target=open_browser, daemon=True).start()
 
# 5. FastAPI Uygulamanı Başlat (Senin mevcut kodun aynen aşağıya gelsin!)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
from pathlib import Path
 
import time
 
# PATH ayarı
sys.path.append(str(Path(__file__).parent.parent / "src" / "nlp"))
sys.path.append(str(Path(__file__).parent.parent / "src" / "query_builder"))
 
from nlp_processor import NLPProcessor
from sql_generator import SQLGenerator
 
app = FastAPI(
    title="Turkish NLP-SQL API",
    description="Doğal dil → SQL için REST API",
    version="1.0.0"
)
 
class QueryRequest(BaseModel):
    text: str
 
nlp_processor = NLPProcessor()
sql_generator = SQLGenerator()
 
@app.post("/generate-sql")
def generate_sql(req: QueryRequest):
    """
    Türkçe doğal dil sorgusunu SQL'e çevirir.
    """
    try:
        start_time = time.time()
        nlp_result = nlp_processor.analyze(req.text)
        sql_result = sql_generator.generate_sql(nlp_result)
        elapsed = round(time.time() - start_time, 3)
 
        if sql_result.get("success"):
            return {
                "success": True,
                "sql": sql_result.get("sql"),
                "intent": sql_result.get("intent"),
                "table": sql_result.get("table"),
                "confidence": sql_result.get("confidence"),
                "has_time_filter": sql_result.get("has_time_filter"),
                "elapsed": elapsed
            }
        else:
            return {
                "success": False,
                "error": sql_result.get("error", "Bilinmeyen hata"),
                "elapsed": elapsed
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sunucu hatası: {str(e)}")
 
@app.get("/")
def root():
    return {"message": "Turkish NLP-SQL API aktif! POST /generate-sql ile kullan."}
 
# (En altta FastAPI sunucusunu başlat)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
 
 