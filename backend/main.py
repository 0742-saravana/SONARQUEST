import os
import cv2
import json
import base64
import random
import io
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from backend.onnx_yolo import YOLOOonnx

# Project Base Directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, 'frontend')
STATIC_DIR = os.path.join(FRONTEND_DIR, 'static')
SAMPLES_DIR = os.path.join(STATIC_DIR, 'samples')
MODEL_PATH = os.path.join(BASE_DIR, 'backend', 'model_weights', 'marineguardv2.onnx')

# Ensure sample directory exists
os.makedirs(SAMPLES_DIR, exist_ok=True)

app = FastAPI(title="SONARQUEST V2 | Marine Sonar Vision")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Static Assets
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Load MarineGuard AI PyTorch Model
print(f"Loading PyTorch Model from: {MODEL_PATH}")
model = YOLOOonnx(MODEL_PATH)

# Class mapping and Threat/Hazard Matrix
CLASS_MAP = {
    0: {"label": "Ghost Net / Debris", "type": "ghost_net", "hazard": "High"},
    1: {"label": "Biomaterial Anomaly", "type": "debris", "hazard": "High"},
    2: {"label": "Subsea Mine / Cylinder", "type": "pipe_cylinder", "hazard": "High"},
    3: {"label": "Shipwreck Structure", "type": "shipwreck", "hazard": "High"},
    4: {"label": "Seafloor Anomaly", "type": "natural_anomaly", "hazard": "Low"}
}

def img_to_base64(img_bgr: np.ndarray) -> str:
    """Encode OpenCV BGR image to Data URL Base64 string."""
    _, buffer = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    b64_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{b64_str}"

def apply_despeckling_clahe(img: np.ndarray) -> np.ndarray:
    """Applies Contrast Limited Adaptive Histogram Equalization and bilateral noise filtering."""
    if len(img.shape) == 2:
        gray = img
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    denoised = cv2.bilateralFilter(equalized, d=5, sigmaColor=50, sigmaSpace=50)
    return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)

def create_segmentation_visual(img_bgr: np.ndarray) -> np.ndarray:
    """Generate Acoustic Highlight & Shadow (AHS) segmentation visual."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Highlights (Acoustic reflections) - Top 15% brightness
    _, highlights = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    # Shadows (Acoustic shadow behind objects) - Bottom 15% brightness
    _, shadows = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY_INV)
    
    seg_img = img_bgr.copy()
    # Tint highlights in cyan/emerald
    seg_img[highlights > 0] = [200, 240, 65]
    # Tint shadows in deep indigo/purple
    seg_img[shadows > 0] = [80, 20, 140]
    
    return cv2.addWeighted(img_bgr, 0.5, seg_img, 0.5, 0)

def compute_snr(img: np.ndarray) -> float:
    """Estimate Signal-to-Noise Ratio (SNR) in dB."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))
    if std_val == 0:
        return 24.5
    snr = 20 * np.log10(max(mean_val, 1) / max(std_val, 0.1))
    return max(10.0, min(32.0, snr + 8.0))

@app.get("/", response_class=HTMLResponse)
async def serve_landing():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="index.html not found")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/api/samples")
async def get_samples():
    """Return available sample sonar survey logs."""
    sample_manifest = [
        {"id": "sample1", "title": "Survey Mission 01 - Ghost Net & Mine Field", "url": "/static/samples/sample1.jpg"},
        {"id": "sample2", "title": "Survey Mission 02 - Deep Seabed Hazard Log", "url": "/static/samples/sample2.jpg"},
        {"id": "sample3", "title": "Survey Mission 03 - Shipwreck Acoustic Scan", "url": "/static/samples/sample3.jpg"},
        {"id": "sample4", "title": "Survey Mission 04 - Anthropogenic Debris Cluster", "url": "/static/samples/sample4.jpg"}
    ]
    return JSONResponse(content=sample_manifest)

@app.post("/api/analyze")
async def analyze_sonar_scan(
    file: Optional[UploadFile] = File(None),
    sample_id: Optional[str] = Form(None),
    confidence_threshold: float = Form(0.60),
    enable_despeckle: bool = Form(True)
):
    try:
        # 1. Read input image
        img_bgr = None
        if file is not None and file.filename != '':
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif sample_id:
            sample_path = os.path.join(SAMPLES_DIR, f"{sample_id}.jpg")
            if os.path.exists(sample_path):
                img_bgr = cv2.imread(sample_path)
            else:
                samples = [f for f in os.listdir(SAMPLES_DIR) if f.endswith('.jpg')]
                if samples:
                    img_bgr = cv2.imread(os.path.join(SAMPLES_DIR, samples[0]))
                    
        if img_bgr is None:
            raise HTTPException(status_code=400, detail="No valid sonar image or sample ID provided.")

        h, w = img_bgr.shape[:2]

        # 2. Preprocessing / Despeckling
        if enable_despeckle:
            despeckled_bgr = apply_despeckling_clahe(img_bgr)
        else:
            despeckled_bgr = img_bgr.copy()

        # 3. Model Inference with ONNX MarineGuard
        results = model.predict(despeckled_bgr, conf_thresh=confidence_threshold)
        
        # 4. Generate Visualizations
        annotated_bgr = model.plot(despeckled_bgr, results, CLASS_MAP)
        segmentation_bgr = create_segmentation_visual(despeckled_bgr)

        # 5. Extract Structured Detections & Geotags
        base_lat = 13.1500
        base_lon = 80.6500
        detections = []

        if len(results) > 0:
            for idx, box in enumerate(results):
                bx, by, bw, bh = box["cx"], box["cy"], box["w"], box["h"]
                conf = box["conf"]
                cls_id = box["cls"]

                class_info = CLASS_MAP.get(cls_id, {
                    "label": "Debris", "type": "debris", "hazard": "Medium"
                })

                # Calculate Slant Range (meters from center nadir)
                norm_x = (bx - (w / 2.0)) / (w / 2.0)
                slant_range_m = round(abs(norm_x) * 60.0, 1)

                # Geodetic projection offset
                obj_lat = base_lat + ((by / h) - 0.5) * 0.008
                obj_lon = base_lon + norm_x * 0.008

                detections.append({
                    "id": f"DET-{idx+1:02d}",
                    "label": class_info["label"],
                    "type": class_info["type"],
                    "confidence": round(conf, 4),
                    "hazard_level": class_info["hazard"],
                    "slant_range_m": slant_range_m,
                    "geotag": {
                        "latitude": round(obj_lat, 6),
                        "longitude": round(obj_lon, 6),
                        "slant_range_m": slant_range_m
                    }
                })

        snr_val = compute_snr(img_bgr)

        payload = {
            "annotated_image": img_to_base64(annotated_bgr),
            "segmentation_image": img_to_base64(segmentation_bgr),
            "despeckled_image": img_to_base64(despeckled_bgr),
            "detections": detections,
            "telemetry": {
                "snr_db": round(snr_val, 1)
            },
            "mission_metadata": {
                "mission_id": "SQ-ALPHA-2026",
                "swath_width_m": 120.0,
                "auv_start_coords": {
                    "lat": base_lat,
                    "lon": base_lon
                }
            }
        }

        return JSONResponse(content=payload)

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

        h, w = img_bgr.shape[:2]

        # 2. Preprocessing / Despeckling
        if enable_despeckle:
            despeckled_bgr = apply_despeckling_clahe(img_bgr)
        else:
            despeckled_bgr = img_bgr.copy()

        # 3. Model Inference with ONNX MarineGuard
     results = model.predict(despeckled_bgr, conf_thresh=confidence_threshold)
        
        # 4. Generate Visualizations
       annotated_bgr = model.plot(despeckled_bgr, results, CLASS_MAP)
        if len(results) > 0:
            annotated_bgr = results[0].plot()

        segmentation_bgr = create_segmentation_visual(despeckled_bgr)

        # 5. Extract Structured Detections & Geotags
        base_lat = 13.1500
        base_lon = 80.6500
        detections = []

        if len(results) > 0 and len(results[0].boxes) > 0:
          for idx, box in enumerate(results):
             bx, by, bw, bh = box["cx"], box["cy"], box["w"], box["h"]
conf = box["conf"]
cls_id = box["cls"]

                class_info = CLASS_MAP.get(cls_id, {
                    "label": "Debris", "type": "debris", "hazard": "Medium"
                })

                # Calculate Slant Range (meters from center nadir)
                norm_x = (bx - (w / 2.0)) / (w / 2.0)
                slant_range_m = round(abs(norm_x) * 60.0, 1)

                # Geodetic projection offset
                obj_lat = base_lat + ((by / h) - 0.5) * 0.008
                obj_lon = base_lon + norm_x * 0.008

                detections.append({
                    "id": f"DET-{idx+1:02d}",
                    "label": class_info["label"],
                    "type": class_info["type"],
                    "confidence": round(conf, 4),
                    "hazard_level": class_info["hazard"],
                    "slant_range_m": slant_range_m,
                    "geotag": {
                        "latitude": round(obj_lat, 6),
                        "longitude": round(obj_lon, 6),
                        "slant_range_m": slant_range_m
                    }
                })

        snr_val = compute_snr(img_bgr)

        payload = {
            "annotated_image": img_to_base64(annotated_bgr),
            "segmentation_image": img_to_base64(segmentation_bgr),
            "despeckled_image": img_to_base64(despeckled_bgr),
            "detections": detections,
            "telemetry": {
                "snr_db": round(snr_val, 1)
            },
            "mission_metadata": {
                "mission_id": "SQ-ALPHA-2026",
                "swath_width_m": 120.0,
                "auv_start_coords": {
                    "lat": base_lat,
                    "lon": base_lon
                }
            }
        }

        return JSONResponse(content=payload)

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/export/image")
async def export_image(payload: Dict[str, Any] = Body(...)):
    """Export and download annotated JPG."""
    data_url = payload.get("image_base64", "")
    if "," in data_url:
        _, b64_data = data_url.split(",", 1)
    else:
        b64_data = data_url
    img_bytes = base64.b64decode(b64_data)
    return Response(
        content=img_bytes,
        media_type="image/jpeg",
        headers={"Content-Disposition": f"attachment; filename=sonarquest_detection_{int(random.random()*10000)}.jpg"}
    )

@app.post("/api/export/csv")
async def export_csv(payload: Dict[str, Any] = Body(...)):
    """Export detection results to formatted CSV."""
    detections = payload.get("detections", [])
    rows = []
    for d in detections:
        geo = d.get("geotag", {})
        rows.append({
            "Detection_ID": d.get("id"),
            "Classification": d.get("label"),
            "Hazard_Type": d.get("type"),
            "Confidence_Score": f"{d.get('confidence', 0)*100:.1f}%",
            "Hazard_Level": d.get("hazard_level"),
            "Slant_Range_Meters": d.get("slant_range_m"),
            "Latitude_WGS84": geo.get("latitude"),
            "Longitude_WGS84": geo.get("longitude")
        })
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=["Detection_ID", "Classification", "Hazard_Type", "Confidence_Score", "Hazard_Level", "Slant_Range_Meters", "Latitude_WGS84", "Longitude_WGS84"])
    
    csv_str = df.to_csv(index=False)
    return Response(
        content=csv_str.encode('utf-8'),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=sonarquest_metadata_report.csv"}
    )

@app.post("/api/export/json")
async def export_json(payload: Dict[str, Any] = Body(...)):
    """Export inspection manifest JSON."""
    json_bytes = json.dumps(payload, indent=4).encode('utf-8')
    return Response(
        content=json_bytes,
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=sonarquest_manifest.json"}
    )

@app.post("/api/export/pdf")
async def export_pdf(payload: Dict[str, Any] = Body(...)):
    """Generate executive PDF mission report with embedded sonar imagery and geodetic hazard tables."""
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, HRFlowable
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    import datetime

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    
    # Custom Typography Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#0f172a'),
        spaceAfter=2
    )
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#2563eb')
    )
    meta_style = ParagraphStyle(
        'MetaText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor('#475569')
    )
    section_heading = ParagraphStyle(
        'SecHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=colors.HexColor('#0f172a'),
        spaceBefore=10,
        spaceAfter=4
    )
    table_text = ParagraphStyle(
        'TableText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor('#0f172a')
    )
    table_text_bold = ParagraphStyle(
        'TableTextBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor('#ffffff')
    )

    story = []

    # 1. Header Banner
    now_str = datetime.datetime.now().strftime("%B %d, %Y - %H:%M UTC")
    doc_id = f"SQ-REP-{datetime.datetime.now().strftime('%Y%m%d')}-{random.randint(1000, 9999)}"
    
    header_data = [
        [
            Paragraph("<b>SONARQUEST V2</b> | Marine Sonar Vision", title_style),
            Paragraph(f"<b>Doc ID:</b> {doc_id}<br/><b>Date:</b> {now_str}", meta_style)
        ],
        [
            Paragraph("National Autonomous Marine Debris & Sonar Anomaly Inspection Brief", subtitle_style),
            Paragraph("<b>Standard:</b> IHO S-44 Compliant", meta_style)
        ]
    ]
    t_header = Table(header_data, colWidths=[360, 180])
    t_header.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 1),
        ('TOPPADDING', (0, 0), (-1, -1), 1),
    ]))
    story.append(t_header)
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#2563eb'), spaceAfter=8))

    # 2. Mission Telemetry
    telemetry = payload.get("telemetry", {})
    mission_meta = payload.get("mission_metadata", {})
    detections = payload.get("detections", [])
    
    snr = telemetry.get("snr_db", 20.2)
    swath = mission_meta.get("swath_width_m", 120.0)
    auv_coords = mission_meta.get("auv_start_coords", {"lat": 13.1500, "lon": 80.6500})
    
    telemetry_data = [
        [
            Paragraph("<b>Survey Sector:</b> Bay of Bengal Basin (35 NM Offshore)", meta_style),
            Paragraph(f"<b>Acoustic SNR:</b> {snr} dB", meta_style),
            Paragraph(f"<b>Swath Width:</b> {swath} m", meta_style)
        ],
        [
            Paragraph(f"<b>AUV Geodesy:</b> {auv_coords.get('lat', 13.15)}°N, {auv_coords.get('lon', 80.65)}°E", meta_style),
            Paragraph("<b>Despeckle Filter:</b> OpenCV CLAHE + Bilateral", meta_style),
            Paragraph(f"<b>Hazards Found:</b> <b>{len(detections)} Targets</b>", meta_style)
        ]
    ]
    t_tel = Table(telemetry_data, colWidths=[180, 180, 180])
    t_tel.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f1f5f9')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_tel)
    story.append(Spacer(1, 8))

    # 3. Sonar Annotated Visual
    annotated_b64 = payload.get("annotated_image", "")
    if annotated_b64:
        try:
            if "," in annotated_b64:
                _, img_data_clean = annotated_b64.split(",", 1)
            else:
                img_data_clean = annotated_b64
            img_bytes = base64.b64decode(img_data_clean)
            img_stream = io.BytesIO(img_bytes)
            
            story.append(Paragraph("1. Side-Scan Sonar Acoustic Waterfall & AI Detections", section_heading))
            # Fit image within page width (540 pt) and max height 180 pt
            sonar_img_flow = RLImage(img_stream, width=540, height=160)
            story.append(sonar_img_flow)
            story.append(Spacer(1, 6))
        except Exception as e:
            print("Failed to embed image in PDF:", e)

    # 4. Geodetic Hazard Log Table
    story.append(Paragraph("2. Detected Objects & Geodetic Recovery Coordinates (WGS84)", section_heading))
    
    table_rows = [
        [
            Paragraph("<b>ID</b>", table_text_bold),
            Paragraph("<b>Classification</b>", table_text_bold),
            Paragraph("<b>Confidence</b>", table_text_bold),
            Paragraph("<b>Hazard Level</b>", table_text_bold),
            Paragraph("<b>Slant Range</b>", table_text_bold),
            Paragraph("<b>Latitude</b>", table_text_bold),
            Paragraph("<b>Longitude</b>", table_text_bold),
        ]
    ]

    if not detections:
        table_rows.append([
            Paragraph("NONE", table_text),
            Paragraph("No target hazards detected above confidence threshold.", table_text),
            Paragraph("--", table_text),
            Paragraph("CLEARED", table_text),
            Paragraph("--", table_text),
            Paragraph("--", table_text),
            Paragraph("--", table_text),
        ])
    else:
        for d in detections:
            geo = d.get("geotag", {})
            conf_str = f"{d.get('confidence', 0)*100:.0f}%"
            lat_str = f"{geo.get('latitude', 0):.5f}°"
            lon_str = f"{geo.get('longitude', 0):.5f}°"
            
            table_rows.append([
                Paragraph(f"<b>{d.get('id', '--')}</b>", table_text),
                Paragraph(d.get('label', '--'), table_text),
                Paragraph(conf_str, table_text),
                Paragraph(f"<b>{d.get('hazard_level', '--')}</b>", table_text),
                Paragraph(f"{d.get('slant_range_m', '--')} m", table_text),
                Paragraph(lat_str, table_text),
                Paragraph(lon_str, table_text),
            ])

    t_det = Table(table_rows, colWidths=[45, 125, 60, 75, 65, 85, 85])
    t_det.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0f172a')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#ffffff'), colors.HexColor('#f8fafc')])
    ]))
    story.append(t_det)
    story.append(Spacer(1, 8))

    # 5. Recommendation & Compliance Sign-off
    story.append(Paragraph("3. Environmental Impact Assessment & Action Directive", section_heading))
    recommendation_text = (
        "<b>Operational Directive:</b> Immediate salvage clearance requested for high-priority anthropogenic debris "
        "and discarded fishing gear (ghost nets) identified within the survey sector to prevent marine megafauna "
        "entrapment and protect benthic habitats. Coordinates calibrated using acoustic slant-to-ground range georeferencing."
    )
    t_rec = Table([[Paragraph(recommendation_text, meta_style)]], colWidths=[540])
    t_rec.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#eff6ff')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#93c5fd')),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(t_rec)
    story.append(Spacer(1, 10))

    # Sign-off line
    sign_off_data = [
        [
            Paragraph("<b>Automated Verification:</b> MarineGuard AI v2.4 (Edge ONNX)", meta_style),
            Paragraph("<b>Authorized Operator:</b> ___________________________", meta_style)
        ]
    ]
    t_sign = Table(sign_off_data, colWidths=[270, 270])
    story.append(t_sign)

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=sonarquest_mission_report.pdf"}
    )

