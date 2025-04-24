import streamlit as st
import os
import json
import google.generativeai as genai
import re
from pathlib import Path
import time
import cv2
import base64
from typing import Dict, List, Union
import numpy as np

# Configuration
API_KEY = "AIzaSyAu-8UFZAx07gxgwy1aD_mgiTARy8ANgLs"
genai.configure(api_key=API_KEY)
BASE_DIR = r"D:\Home Inspection project\Json files"
OUTPUT_DIR = os.path.join(BASE_DIR, "extracted_frames")
JSON_RAW_PATH = os.path.join(BASE_DIR, "inspection_raw.json")
JSON_REPORT_PATH = os.path.join(BASE_DIR, "inspection_report.json")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize Gemini Model
system_instruction = (
    "You are an expert at analysing residential building and producing detailed inspection reports. "
    "Your job is to analyse the user provided media and produce a detailed inspection report based on Australian building standards."
)
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash-002",
    system_instruction=system_instruction,
    generation_config={
        "temperature": 0.1,
        "max_output_tokens": 8192,
        "response_mime_type": "application/json",
    }
)

# Example data
example_jsons = {
    "example1": {
        "detailedInspection": [{"area": "Roof", "mediaReference": "example1.jpg", "timestamp": "N/A", "condition": "Good", "complianceStatus": "Compliant", "issuesFound": [], "referenceDoc": "ABC", "referenceSection": "3.1.1", "recommendation": "Regular maintenance recommended."}],
        "executiveSummary": {"overallCondition": "Good", "criticalIssues": [], "recommendedActions": []},
        "maintenanceNotes": {"recurringIssues": [], "preventiveRecommendations": ["Check roof annually"], "maintenanceSchedule": [{"frequency": "Annually", "tasks": ["Inspect roof"]}], "costConsiderations": ["Low cost for annual inspection"]}
    },
    "example2": {
        "detailedInspection": [{"area": "Exterior Walls", "mediaReference": "example2.mp4", "timestamp": "0:05", "condition": "Fair", "complianceStatus": "Non-compliant", "issuesFound": ["Cracks observed"], "referenceDoc": "ABC", "referenceSection": "3.2.1", "recommendation": "Repair cracks immediately."}],
        "executiveSummary": {"overallCondition": "Fair", "criticalIssues": ["Cracks in walls"], "recommendedActions": ["Immediate repair"]},
        "maintenanceNotes": {"recurringIssues": ["Wall cracks"], "preventiveRecommendations": ["Monitor cracks"], "maintenanceSchedule": [{"frequency": "Quarterly", "tasks": ["Check walls"]}], "costConsiderations": ["Moderate repair costs"]}
    }
}

# Helper Functions
def upload_file_to_gemini(file_path: str) -> genai.__file__:
    try:
        uploaded_file = genai.upload_file(path=file_path)
        if uploaded_file.mime_type.startswith("video/"):
            while uploaded_file.state.name == "PROCESSING":
                time.sleep(5)
                uploaded_file = genai.get_file(uploaded_file.name)
            if uploaded_file.state.name == "FAILED":
                raise ValueError(f"Video processing failed for {file_path}")
        return uploaded_file
    except Exception as e:
        st.error(f"Error uploading file {file_path}: {str(e)}")
        return None

def extract_video_frames(video_path: str, timestamps: List[str]) -> Dict[str, str]:
    if not video_path or not timestamps:
        return {}
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"Error opening video file: {video_path}")
        return {}
    
    frame_paths = {}
    for timestamp in timestamps:
        if timestamp == "N/A":
            continue
        try:
            if "-" in timestamp:
                start, end = timestamp.split("-")
                start_seconds = sum(int(x) * 60 ** i for i, x in enumerate(reversed(start.split(':'))))
                end_seconds = sum(int(x) * 60 ** i for i, x in enumerate(reversed(end.split(':'))))
                for sec in range(start_seconds, end_seconds + 1, 5):
                    cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
                    ret, frame = cap.read()
                    if ret:
                        frame_filename = f"frame_{timestamp.replace(':', '_').replace('-', '_')}_{sec}.jpg"
                        frame_path = os.path.join(OUTPUT_DIR, frame_filename)
                        cv2.imwrite(frame_path, frame)
                        frame_paths[f"{timestamp}_{sec}"] = frame_path
                        st.write(f"Extracted frame saved at: {frame_path}")
            else:
                seconds = sum(int(x) * 60 ** i for i, x in enumerate(reversed(timestamp.split(':'))))
                cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
                ret, frame = cap.read()
                if ret:
                    frame_filename = f"frame_{timestamp.replace(':', '_')}.jpg"
                    frame_path = os.path.join(OUTPUT_DIR, frame_filename)
                    cv2.imwrite(frame_path, frame)
                    frame_paths[timestamp] = frame_path
                    st.write(f"Extracted frame saved at: {frame_path}")
        except ValueError as e:
            st.warning(f"Invalid timestamp format '{timestamp}': {str(e)}")
            continue
    cap.release()
    return frame_paths

def get_image_base64(image_path: str) -> str:
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")
    except Exception as e:
        st.error(f"Error reading image {image_path}: {e}")
        return None

def detect_issues_in_image(img: np.ndarray, issues: List[str]) -> List[tuple]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    issue_locations = []
    
    for issue in issues:
        issue_lower = issue.lower()
        
        if "crack" in issue_lower or "fracture" in issue_lower:
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY_INV, 11, 2)
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(thresh, kernel, iterations=2)
            eroded = cv2.erode(dilated, kernel, iterations=1)
            contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) < 150:
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                center_x = x + w // 2
                center_y = y + h // 2
                issue_locations.append((center_x, center_y))
        
        elif "water" in issue_lower or "stain" in issue_lower or "leak" in issue_lower:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lower_stain = np.array([10, 50, 50])
            upper_stain = np.array([30, 255, 255])
            mask = cv2.inRange(hsv, lower_stain, upper_stain)
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) < 200:
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                center_x = x + w // 2
                center_y = y + h // 2
                issue_locations.append((center_x, center_y))
        
        elif "mold" in issue_lower or "mould" in issue_lower:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lower_mold = np.array([0, 0, 0])
            upper_mold = np.array([180, 255, 50])
            mask = cv2.inRange(hsv, lower_mold, upper_mold)
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) < 150:
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                center_x = x + w // 2
                center_y = y + h // 2
                issue_locations.append((center_x, center_y))
    
    if not issue_locations and issues:
        height, width = img.shape[:2]
        issue_locations.append((width // 2, height // 2))
    
    return issue_locations

def display_image(file_name: str, max_width: int = 300, issues: List[str] = None) -> None:
    file_path = os.path.join(BASE_DIR, file_name)
    if os.path.exists(file_path):
        img = cv2.imread(file_path)
        if img is not None:
            ratio = max_width / img.shape[1]
            new_height = int(img.shape[0] * ratio)
            resized_img = cv2.resize(img, (max_width, new_height))
            
            if issues and len(issues) > 0:
                issue_locations = detect_issues_in_image(resized_img, issues)
                for center_x, center_y in issue_locations:
                    radius = max(20, min(resized_img.shape[0], resized_img.shape[1]) // 20)
                    color = (0, 255, 0)
                    thickness = 2
                    cv2.circle(resized_img, (center_x, center_y), radius, color, thickness)
            
            resized_img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
            st.image(resized_img_rgb, caption=file_name, use_container_width=False, width=max_width)

def display_frame(file_name: str, max_width: int = 300, issues: List[str] = None) -> None:
    frame_path = os.path.join(OUTPUT_DIR, file_name)
    if os.path.exists(frame_path):
        img = cv2.imread(frame_path)
        if img is not None:
            ratio = max_width / img.shape[1]
            new_height = int(img.shape[0] * ratio)
            resized_img = cv2.resize(img, (max_width, new_height))
            
            if issues and len(issues) > 0:
                issue_locations = detect_issues_in_image(resized_img, issues)
                for center_x, center_y in issue_locations:
                    radius = max(20, min(resized_img.shape[0], resized_img.shape[1]) // 20)
                    color = (0, 255, 0)
                    thickness = 2
                    cv2.circle(resized_img, (center_x, center_y), radius, color, thickness)
            
            resized_img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
            st.image(resized_img_rgb, caption=file_name, use_container_width=False, width=max_width)

def parse_maintenance_schedule(response_json: dict) -> List[Dict]:
    schedule_items = []
    seen_tasks = set()
    
    if 'detailedInspection' in response_json:
        for inspection in response_json['detailedInspection']:
            if inspection.get('complianceStatus') == 'Non-compliant':
                recommendation = inspection.get('recommendation', '').strip()
                issues = inspection.get('issuesFound', [])
                condition = inspection.get('condition', '').lower()
                
                severity_keywords = ['immediate', 'critical', 'urgent', 'termite', 'pest', 'poor', 'severe']
                is_immediate = any(word in ' '.join(issues + [recommendation] + [condition]).lower() 
                                 for word in severity_keywords)
                frequency = 'Immediate' if is_immediate else 'Quarterly'
                
                task = recommendation if recommendation else f"Address issues in {inspection.get('area', 'unknown area')}: {', '.join(issues) or 'general maintenance'}"
                if task and task not in seen_tasks:
                    priority = 'High' if frequency == 'Immediate' else 'Medium' if any(word in condition for word in ['fair', 'moderate']) else 'Low'
                    schedule_items.append({
                        'Task': task,
                        'Frequency': frequency,
                        'Priority': priority,
                        'Status': 'Pending'
                    })
                    seen_tasks.add(task)
    
    if 'maintenanceNotes' in response_json and 'maintenanceSchedule' in response_json['maintenanceNotes']:
        for schedule in response_json['maintenanceNotes']['maintenanceSchedule']:
            task = ', '.join(schedule.get('tasks', ['Unknown task']))
            if task and task not in seen_tasks:
                frequency = schedule.get('frequency', 'Quarterly')
                priority = 'High' if frequency == 'Immediate' else 'Medium' if frequency in ['Quarterly', 'Semi-annually'] else 'Low'
                schedule_items.append({
                    'Task': task,
                    'Frequency': frequency,
                    'Priority': priority,
                    'Status': 'Pending'
                })
                seen_tasks.add(task)
    
    standard_tasks = [
        {'Task': 'General inspection of building condition', 'Frequency': 'Annually', 'Priority': 'Low', 'Status': 'Pending'},
        {'Task': 'Check and clean gutters and drainage systems', 'Frequency': 'Quarterly', 'Priority': 'Medium', 'Status': 'Pending'},
        {'Task': 'Inspect for pest activity', 'Frequency': 'Semi-annually', 'Priority': 'Medium', 'Status': 'Pending'}
    ]
    for task in standard_tasks:
        if task['Task'] not in seen_tasks:
            schedule_items.append(task)
            seen_tasks.add(task['Task'])
    
    return schedule_items

def transform_response(response_json: Dict) -> Dict:
    if not isinstance(response_json, dict) or "detailedInspection" not in response_json:
        raise ValueError("Invalid report format: Missing 'detailedInspection'")

    transformed = {
        "detailedInspection": [],
        "executiveSummary": {"overallCondition": "Good", "criticalIssues": [], "recommendedActions": []},
        "maintenanceNotes": {"recurringIssues": [], "preventiveRecommendations": [], "maintenanceSchedule": [], "costConsiderations": []}
    }

    uploaded_media = list(document_dict["user_data"].keys())
    media_index = 0

    for item in response_json.get("detailedInspection", []):
        section_name = item.get("area", "Unknown Area")
        condition = item.get("condition", "Unknown")
        notes = item.get("issuesFound", [""])[0] if item.get("issuesFound") else ""
        recommendations = item.get("recommendation", "").split("<br>") if item.get("recommendation") else []

        compliance_status = "Compliant" if condition.lower() == "good" else "Non-compliant"
        issues_found = item.get("issuesFound", []) if item.get("issuesFound") else [notes] if notes else []

        recommendation = recommendations[0] if recommendations else ""
        if len(recommendations) > 1:
            recommendation += "<br>".join(recommendations[1:])

        timestamp = item.get("timestamp", "N/A")
        if "home_inspection.mp4" in uploaded_media[media_index] and timestamp == "N/A":
            timestamp_match = re.search(r'(\d{1,2}:\d{2}(?:-\d{1,2}:\d{2})?)', notes)
            timestamp = timestamp_match.group(1) if timestamp_match else "N/A"

        media_reference = uploaded_media[media_index] if media_index < len(uploaded_media) else "user_uploaded_media"
        if media_index < len(uploaded_media) - 1:
            media_index += 1

        transformed["detailedInspection"].append({
            "area": section_name,
            "mediaReference": media_reference,
            "timestamp": timestamp,
            "condition": f"{condition} - {notes.split('.')[0]}" if notes else condition,
            "complianceStatus": compliance_status,
            "issuesFound": issues_found,
            "referenceDoc": item.get("referenceDoc", "Australian Building Code (ABC) Housing Provisions"),
            "referenceSection": item.get("referenceSection", "Clause 3.1.1"),
            "recommendation": recommendation
        })

        if compliance_status == "Non-compliant":
            transformed["maintenanceNotes"]["recurringIssues"].append(f"{section_name} requires attention")
            transformed["maintenanceNotes"]["preventiveRecommendations"].append(f"Regular inspection of {section_name} for maintenance")
            transformed["maintenanceNotes"]["maintenanceSchedule"].append({
                "frequency": "Quarterly" if condition.lower() != "poor" else "Immediate",
                "tasks": [f"Inspect and maintain {section_name}"]
            })
            transformed["maintenanceNotes"]["costConsiderations"].append(f"Estimated cost for {section_name} repairs: Moderate")

    if "executiveSummary" in response_json:
        transformed["executiveSummary"]["overallCondition"] = response_json["executiveSummary"].get("overallCondition", "Good")
        transformed["executiveSummary"]["criticalIssues"] = response_json["executiveSummary"].get("criticalIssues", [])
        transformed["executiveSummary"]["recommendedActions"] = response_json["executiveSummary"].get("recommendedActions", [])

    if "maintenanceNotes" in response_json:
        transformed["maintenanceNotes"]["recurringIssues"].extend(response_json["maintenanceNotes"].get("recurringIssues", []))
        transformed["maintenanceNotes"]["preventiveRecommendations"].extend(response_json["maintenanceNotes"].get("preventiveRecommendations", []))
        transformed["maintenanceNotes"]["maintenanceSchedule"].extend(response_json["maintenanceNotes"].get("maintenanceSchedule", []))
        transformed["maintenanceNotes"]["costConsiderations"].extend(response_json["maintenanceNotes"].get("costConsiderations", []))

    return transformed

def get_gemini_response(question: str, context: str) -> str:
    try:
        # Parse the context JSON to extract relevant details
        context_data = json.loads(context)
        prompt = (
            f"Using the following inspection context: {context}. The user has asked: '{question}'. "
            f"Identify any relevant details from the context related to the specific issue mentioned in the question. "
            f"Provide a detailed, practical solution in plain text, focusing on step-by-step instructions to address the issue. "
            f"Include safety precautions, required tools, and any references to Australian building standards (e.g., AS 1684.1-2010) if applicable. "
            f"If professional help is needed, explicitly state so and explain why. Return the response as a numbered list of steps."
        )
        response = chat_session.send_message(prompt)
        
        text = response.text.strip()
        lines = text.split("\n")
        formatted_response = ""
        step_num = 1
        for line in lines:
            line = line.strip()
            if line and not line.startswith("**") and not line.startswith('"') and not line.startswith("'"):
                if "Disclaimer" in line:
                    formatted_response += f"<strong>Disclaimer:</strong> {line.replace('Disclaimer:', '').strip()}\n"
                elif any(keyword in line.lower() for keyword in ["professional", "technician", "contact", "safety", "dangerous"]):
                    formatted_response += f"<strong>Important Note:</strong> {line.strip()}\n"
                else:
                    formatted_response += f"{step_num}. {line.strip()}\n"
                    step_num += 1
        return formatted_response.strip() if formatted_response else text
    except Exception as e:
        return f"Error generating response: {str(e)}. Please try again or check your internet connection."

# Main Streamlit App
def main():
    st.set_page_config(page_title="Home Inspection Report Dashboard", layout="wide", page_icon="🏠")
    
    st.markdown("""
        <style>
        body {
            background-color: #f5f7fa;
            font-family: 'Arial', sans-serif;
        }
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        .report-card {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            border-left: 5px solid #2c3e50;
        }
        .header {
            color: #2c3e50;
            font-weight: bold;
            font-size: 24px;
            margin-bottom: 20px;
            text-align: center;
        }
        .subheader {
            color: #ffffff;
            font-size: 18px;
            margin-bottom: 15px;
            font-weight: bold;
        }
        .status-non-compliant {
            background-color: #ffcccc;
            color: #990000;
            padding: 6px 12px;
            border-radius: 6px;
            display: inline-block;
            font-weight: bold;
            font-size: 14px;
        }
        .status-compliant {
            background-color: #e8f5e9;
            color: #1b5e20;
            padding: 6px 12px;
            border-radius: 6px;
            display: inline-block;
            font-weight: bold;
            font-size: 14px;
        }
        .chat-container {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            max-height: 600px;
            overflow-y: auto;
        }
        .chat-message {
            margin: 10px 0;
            padding: 10px;
            border-radius: 5px;
            line-height: 1.6;
        }
        .user-message {
            background-color: #e8f5e9;
            text-align: right;
            color: #000000;
        }
        .bot-message {
            background-color: #e0e0e0;
            text-align: left;
            color: #000000;
        }
        .bot-message strong {
            color: #2c3e50;
            font-weight: bold;
        }
        .bot-message ol {
            padding-left: 20px;
            margin: 0;
        }
        .bot-message li {
            margin-bottom: 10px;
        }
        .maintenance-table {
            width: 100%;
            border-collapse: collapse;
        }
        .maintenance-table th, .maintenance-table td {
            padding: 8px 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        .maintenance-table th {
            background-color: #2c3e50;
            color: white;
            font-weight: bold;
        }
        .maintenance-table td {
            background-color: #ffffff;
            color: #000000;
        }
        .maintenance-table th:nth-child(1), .maintenance-table td:nth-child(1) { min-width: 300px; }
        .maintenance-table th:nth-child(2), .maintenance-table td:nth-child(2) { min-width: 100px; }
        .maintenance-table th:nth-child(3), .maintenance-table td:nth-child(3) { min-width: 80px; }
        .maintenance-table th:nth-child(4), .maintenance-table td:nth-child(4) { min-width: 80px; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 class='header'>🏠 Home Inspection Report Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("""
        <p style='text-align: center; color: #ffffff; font-size: 16px; margin-bottom: 30px;'>
            Upload photos or videos of your home, and let our AI analyze them against Australian building standards for a detailed, professional inspection report.
        </p>
    """, unsafe_allow_html=True)

    st.markdown("<h2 class='subheader'>Upload Your Home Media</h2>", unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload photos or videos", 
        accept_multiple_files=True, 
        type=["jpg", "jpeg", "png", "mp4"],
        help="Supported formats: JPG, JPEG, PNG, MP4"
    )

    if uploaded_files:
        with st.spinner("Analyzing your media... Please wait."):
            global document_dict
            document_dict = {
                "examples": {
                    "example1": {f"example1_{ext}": genai.upload_file(os.path.join(BASE_DIR, f"example1.{ext}")) for ext in ["jpg"] if os.path.exists(os.path.join(BASE_DIR, f"example1.{ext}"))},
                    "example2": {f"example2_{ext}": genai.upload_file(os.path.join(BASE_DIR, f"example2.{ext}")) for ext in ["mp4"] if os.path.exists(os.path.join(BASE_DIR, f"example2.{ext}"))}
                },
                "user_data": {}
            }
            video_path = None
            
            for uploaded_file in uploaded_files:
                file_path = os.path.join(BASE_DIR, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                uploaded_file_obj = upload_file_to_gemini(file_path)
                if uploaded_file_obj:
                    document_dict["user_data"][uploaded_file.name] = uploaded_file_obj
                    if uploaded_file.type.startswith("video/"):
                        video_path = file_path

            if not document_dict["user_data"]:
                st.error("No files were successfully uploaded. Please try again.")
                return

            prompt = """
You have been supplied with a set of building standards and manufacturer specifications to evaluate the photos and videos against.
Please be specific about any violations of building codes or manufacturer specifications found in the documentation.

Analyze the uploaded photos and videos of the building and generate a detailed inspection report in JSON format.
Be exhaustive in your inspection and cover all aspects of the building shown in the media.

The response should be a valid JSON object with the following structure:

{
  "detailedInspection": [
    {
      "area": "string",
      "mediaReference": "string",
      "timestamp": "string",
      "condition": "string",
      "complianceStatus": "string",
      "issuesFound": ["string"],
      "referenceDoc": "string",
      "referenceSection": "string",
      "recommendation": "string"
    }
  ],
  "executiveSummary": {
    "overallCondition": "string",
    "criticalIssues": ["string"],
    "recommendedActions": ["string"]
  },
  "maintenanceNotes": {
    "recurringIssues": ["string"],
    "preventiveRecommendations": ["string"],
    "maintenanceSchedule": [
      {
        "frequency": "string",
        "tasks": ["string"]
      }
    ],
    "costConsiderations": ["string"]
  }
}

Ensure the response is a valid JSON object that can be parsed.
"""

            content = []
            content.append({'text': prompt})
            content.append({'text': 'Here are some examples of analysed building reports:'})
            content.append({'text': 'Example 1 Media and report (purely for reference):'})
            for name, doc in document_dict['examples']['example1'].items():
                content.append({'text': f"Example 1 Document: {name}"})
                content.append(doc)
            content.append({'text': json.dumps(example_jsons['example1'])})
            content.append({'text': 'Example 2 Media and report (purely for reference):'})
            for name, doc in document_dict['examples']['example2'].items():
                content.append({'text': f"Example 2 Document: {name}"})
                content.append(doc)
            content.append({'text': json.dumps(example_jsons['example2'])})
            content.append({'text': 'Now analyse the user provided media and provide a detailed inspection report. Analyse only the user provided images and video. Do not analyse either example provided earlier. You should analyse the entire video file (home_inspection.mp4) and consider approximately every 5 seconds as a unique timepoint to analyse as well as each image provided:'})
            content.append({'text': 'User provided media:'})
            for name, doc in document_dict['user_data'].items():
                content.append({'text': f"User Document: {name}"})
                content.append(doc)

            global chat_session
            chat_session = model.start_chat(history=[{"role": "user", "parts": content}])
            try:
                response = chat_session.send_message("Please generate a detailed building report. Please provide a detailed answer with elaboration on the report and reference material.")
                st.write("Raw response:", response.text)
                try:
                    response_json = json.loads(response.text)
                except json.JSONDecodeError as json_err:
                    st.error(f"Failed to parse raw response as JSON: {str(json_err)}")
                    st.write("Raw response content:", response.text)
                    return
                with open(JSON_RAW_PATH, "w") as f:
                    json.dump(response_json, f, indent=4)
                if not isinstance(response_json, dict) or "detailedInspection" not in response_json:
                    st.error("Invalid report format received from the API: Missing 'detailedInspection' key or not a dictionary.")
                    return
                transformed_response = transform_response(response_json)
                with open(JSON_REPORT_PATH, "w") as f:
                    json.dump(transformed_response, f, indent=4)
            except Exception as e:
                st.error(f"Error generating report: {str(e)}")
                return

            timestamps = []
            for item in transformed_response["detailedInspection"]:
                if "timestamp" in item and "home_inspection.mp4" in item["mediaReference"] and item["timestamp"] != "N/A":
                    timestamps.extend([ts.strip() for ts in item["timestamp"].split(",")])
            frame_paths = extract_video_frames(video_path, timestamps) if video_path and timestamps else {}

        if os.path.exists(JSON_REPORT_PATH):
            try:
                with open(JSON_REPORT_PATH, "r") as f:
                    transformed_response = json.load(f)
            except Exception as e:
                st.error(f"Error loading transformed report: {str(e)}")
                return
        else:
            st.error(f"Transformed report file not found at {JSON_REPORT_PATH}")
            return

        st.markdown("<h2 class='subheader'>Inspection Report</h2>", unsafe_allow_html=True)
        uploaded_images = [f for f in document_dict["user_data"].keys() if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for item in transformed_response["detailedInspection"]:
            with st.container():
                st.markdown(f"<div class='report-card'>", unsafe_allow_html=True)
                
                if item["mediaReference"] in uploaded_images and item["complianceStatus"] == "Non-compliant":
                    st.markdown(f"<strong>Image with Issues:</strong>", unsafe_allow_html=True)
                    display_image(item["mediaReference"], issues=item["issuesFound"])
                elif item["mediaReference"] in uploaded_images:
                    st.markdown(f"<strong>Image:</strong>", unsafe_allow_html=True)
                    display_image(item["mediaReference"])
                elif "home_inspection.mp4" in item["mediaReference"] and item["timestamp"] != "N/A" and frame_paths:
                    frame_key = [k for k in frame_paths.keys() if item["timestamp"] in k]
                    if frame_key:
                        frame_file = os.path.basename(frame_paths[frame_key[0]])
                        if item["complianceStatus"] == "Non-compliant":
                            st.markdown(f"<strong>Frame with Issues:</strong>", unsafe_allow_html=True)
                            display_frame(frame_file, issues=item["issuesFound"])
                        else:
                            st.markdown(f"<strong>Frame:</strong>", unsafe_allow_html=True)
                            display_frame(frame_file)

                status_class = "status-non-compliant" if item["complianceStatus"] == "Non-compliant" else "status-compliant"
                st.markdown(f"""
                    <h3 class='header' style='margin-bottom: 15px;'>{item['area']}</h3>
                    <span class='{status_class}' style='margin-bottom: 15px;'>{item['complianceStatus']}</span>
                """, unsafe_allow_html=True)

                st.markdown(f"<p style='color: #ffffff; line-height: 1.6;'><strong>Condition:</strong> {item['condition']}</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='color: #ffffff; line-height: 1.6;'><strong>Reference:</strong> {item.get('referenceDoc', 'N/A')} - {item.get('referenceSection', 'N/A')}</p>", unsafe_allow_html=True)
                if item.get("issuesFound"):
                    st.markdown("<p style='color: #ffffff; line-height: 1.6;'><strong>Issues Found:</strong></p>", unsafe_allow_html=True)
                    for issue in item["issuesFound"]:
                        st.markdown(f"<p style='color: #ffffff; line-height: 1.6; margin-left: 20px;'>- {issue}</p>", unsafe_allow_html=True)
                if item.get("recommendation"):
                    st.markdown(f"<p style='color: #ffffff; line-height: 1.6;'><strong>Recommendation:</strong> {item['recommendation']}</p>", unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<h2 class='subheader'>Executive Summary</h2>", unsafe_allow_html=True)
        summary = transformed_response["executiveSummary"]
        st.markdown(f"<p style='color: #ffffff; font-size: 18px; line-height: 1.6;'><strong>Overall Condition:</strong> {summary['overallCondition']}</p>", unsafe_allow_html=True)
        st.markdown("<p style='color: #ffffff; font-size: 18px; line-height: 1.6;'><strong>Critical Issues:</strong></p>", unsafe_allow_html=True)
        for issue in summary["criticalIssues"]:
            st.markdown(f"<p style='color: #ffffff; font-size: 16px; line-height: 1.6; margin-left: 20px;'>- {issue}</p>", unsafe_allow_html=True)
        st.markdown("<p style='color: #ffffff; font-size: 18px; line-height: 1.6;'><strong>Recommended Actions:</strong></p>", unsafe_allow_html=True)
        for action in summary["recommendedActions"]:
            st.markdown(f"<p style='color: #ffffff; font-size: 16px; line-height: 1.6; margin-left: 20px;'>- {action}</p>", unsafe_allow_html=True)

        st.markdown("<h2 class='subheader'>Maintenance Schedule</h2>", unsafe_allow_html=True)
        maintenance_schedule = parse_maintenance_schedule(transformed_response)
        if maintenance_schedule:
            import pandas as pd
            df = pd.DataFrame(maintenance_schedule)
            st.markdown('<div class="maintenance-table">', unsafe_allow_html=True)
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("No maintenance tasks generated for this report.")

        st.markdown("<h2 class='subheader'>Maintenance Notes</h2>", unsafe_allow_html=True)
        notes = transformed_response["maintenanceNotes"]
        st.markdown("<p style='color: #ffffff; font-size: 18px; line-height: 1.6;'><strong>Recurring Issues:</strong></p>", unsafe_allow_html=True)
        for issue in notes["recurringIssues"]:
            st.markdown(f"<p style='color: #ffffff; font-size: 16px; line-height: 1.6; margin-left: 20px;'>- {issue}</p>", unsafe_allow_html=True)
        st.markdown("<p style='color: #ffffff; font-size: 18px; line-height: 1.6;'><strong>Preventive Recommendations:</strong></p>", unsafe_allow_html=True)
        for rec in notes["preventiveRecommendations"]:
            st.markdown(f"<p style='color: #ffffff; font-size: 16px; line-height: 1.6; margin-left: 20px;'>- {rec}</p>", unsafe_allow_html=True)
        st.markdown("<p style='color: #ffffff; font-size: 18px; line-height: 1.6;'><strong>Maintenance Schedule:</strong></p>", unsafe_allow_html=True)
        for schedule in notes["maintenanceSchedule"]:
            st.markdown(f"<p style='color: #ffffff; font-size: 16px; line-height: 1.6; margin-left: 20px;'>- Frequency: {schedule['frequency']}, Tasks: {', '.join(schedule['tasks'])}</p>", unsafe_allow_html=True)
        st.markdown("<p style='color: #ffffff; font-size: 18px; line-height: 1.6;'><strong>Cost Considerations:</strong></p>", unsafe_allow_html=True)
        for cost in notes["costConsiderations"]:
            st.markdown(f"<p style='color: #ffffff; font-size: 16px; line-height: 1.6; margin-left: 20px;'>- {cost}</p>", unsafe_allow_html=True)

        # Q&A Section
        st.markdown("<h2 class='subheader'>Ask Questions About the Inspection</h2>", unsafe_allow_html=True)
        col1, col2 = st.columns([4, 1])
        with col1:
            question = st.text_input("Ask a question about the inspection...", key="chat_input")
        with col2:
            send_button = st.button("Send", key="send_button")

        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        if send_button and question:
            with st.spinner("Generating response..."):
                context = json.dumps(transformed_response, indent=2)
                response = get_gemini_response(question, context)
                st.session_state.chat_history.append({"role": "user", "content": question})
                st.session_state.chat_history.append({"role": "bot", "content": response})

        if st.session_state.chat_history:
            st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
            for message in st.session_state.chat_history:
                message_class = "user-message" if message["role"] == "user" else "bot-message"
                if message["role"] == "bot":
                    if any(str(i) + "." in message["content"] for i in range(1, 10)):
                        html_content = "<ol>" + "".join(f"<li>{line.strip()}</li>" for line in message["content"].split("\n") if line.strip()) + "</ol>"
                        st.markdown(f"<div class='chat-message {message_class}'>{html_content}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div class='chat-message {message_class}'><p>{message['content']}</p></div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='chat-message {message_class}'><p>{message['content']}</p></div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()