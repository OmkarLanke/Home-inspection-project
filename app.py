import streamlit as st
import os
import json
import google.generativeai as genai
import re  # Added for regular expressions
from pathlib import Path
import time
import cv2
import base64
from typing import Dict, List, Union
!pip install -r requirements.txt


# Configuration
API_KEY = "AIzaSyAu-8UFZAx07gxgwy1aD_mgiTARy8ANgLs"  # Updated API key as provided
genai.configure(api_key=API_KEY)
BASE_DIR = r"D:\Home Inspection project\Json files"  # Updated to your specified directory
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

# Helper Functions
def upload_file_to_gemini(file_path: str) -> genai.__file__:
    """Upload a file to Gemini and wait for processing if it's a video."""
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
    """Extract frames from video at specified timestamps, skipping invalid ones like 'N/A'."""
    if not video_path or not timestamps:
        return {}
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"Error opening video file: {video_path}")
        return {}
    
    frame_paths = {}
    for timestamp in timestamps:
        if timestamp == "N/A":  # Skip invalid timestamps
            continue
        try:
            # Handle timestamp ranges (e.g., "0:00-0:21") or single timestamps (e.g., "0:22")
            if "-" in timestamp:
                start, end = timestamp.split("-")
                start_seconds = sum(int(x) * 60 ** i for i, x in enumerate(reversed(start.split(':'))))
                end_seconds = sum(int(x) * 60 ** i for i, x in enumerate(reversed(end.split(':'))))
                for sec in range(start_seconds, end_seconds + 1, 5):  # Extract every 5 seconds in range
                    cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
                    ret, frame = cap.read()
                    if ret:
                        frame_filename = f"frame_{timestamp.replace(':', '_').replace('-', '_')}_{sec}.jpg"
                        frame_path = os.path.join(OUTPUT_DIR, frame_filename)
                        cv2.imwrite(frame_path, frame)
                        frame_paths[f"{timestamp}_{sec}"] = frame_path
                        st.write(f"Extracted frame saved at: {frame_path}")  # Debug log
            else:
                # Handle single timestamp (e.g., "0:22")
                seconds = sum(int(x) * 60 ** i for i, x in enumerate(reversed(timestamp.split(':'))))
                cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
                ret, frame = cap.read()
                if ret:
                    frame_filename = f"frame_{timestamp.replace(':', '_')}.jpg"
                    frame_path = os.path.join(OUTPUT_DIR, frame_filename)
                    cv2.imwrite(frame_path, frame)
                    frame_paths[timestamp] = frame_path
                    st.write(f"Extracted frame saved at: {frame_path}")  # Debug log
        except ValueError as e:
            st.warning(f"Invalid timestamp format '{timestamp}': {str(e)}")
            continue
    cap.release()
    return frame_paths

def get_image_base64(image_path: str) -> str:
    """Convert image to base64 string."""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")
    except Exception as e:
        st.error(f"Error reading image {image_path}: {e}")
        return None

def parse_maintenance_schedule(response_json: dict) -> List[Dict]:
    """Extract and generate maintenance schedule from inspection findings"""
    schedule_items = []
    
    # Generate maintenance tasks from inspection findings
    for inspection in response_json['detailedInspection']:
        if inspection['complianceStatus'] == 'Non-compliant':
            # Get the recommendation and issues
            recommendation = inspection.get('recommendation', '')
            issues = inspection.get('issuesFound', [])
            
            # Determine frequency based on severity
            frequency = 'Immediate' if any(word in ' '.join(issues).lower() 
                                        for word in ['immediate', 'critical', 'urgent', 'termite', 'pest']) \
                       else 'Quarterly'
            
            # Add the task
            schedule_items.append({
                'Task': recommendation,
                'Frequency': frequency,
                'Priority': 'High' if frequency == 'Immediate' else 'Medium',
                'Status': 'Pending'
            })
    
    # Add standard maintenance tasks
    standard_tasks = [
        {
            'Task': 'General inspection of building condition',
            'Frequency': 'Annually',
            'Priority': 'Medium',
            'Status': 'Pending'
        },
        {
            'Task': 'Check and clean gutters and drainage systems',
            'Frequency': 'Quarterly',
            'Priority': 'Medium',
            'Status': 'Pending'
        },
        {
            'Task': 'Inspect for pest activity',
            'Frequency': 'Semi-annually',
            'Priority': 'Medium',
            'Status': 'Pending'
        }
    ]
    
    schedule_items.extend(standard_tasks)
    
    return schedule_items

def transform_response(response_json: Dict) -> Dict:
    """Transform the Gemini response into the expected format for new images/videos."""
    if not isinstance(response_json, dict) or "sections" not in response_json:
        raise ValueError("Invalid report format: Missing 'sections'")

    transformed = {
        "detailedInspection": [],
        "executiveSummary": {
            "overallCondition": "Good",
            "criticalIssues": [],
            "recommendedActions": []
        },
        "maintenanceNotes": {
            "recurringIssues": [],
            "preventiveRecommendations": [],
            "maintenanceSchedule": []
        }
    }

    # Process sections into detailedInspection
    uploaded_media = list(document_dict["user_data"].keys())
    media_reference = uploaded_media[0] if uploaded_media else "user_uploaded_media"

    for section in response_json.get("sections", []):
        section_name = section["section_name"]
        details_list = section.get("details", [])  # Handle list of details

        for details in details_list:  # Process each detail in the list
            condition = details.get("condition", "Unknown")
            notes = details.get("notes", "")

            # Determine compliance status and issues
            compliance_status = "Compliant" if condition.lower() == "good" else "Non-compliant"
            issues_found = [notes] if "further inspection" in notes.lower() or condition.lower() != "good" else []
            recommendation = notes.split(". Further inspection")[0] + ". Conduct a detailed inspection by a professional." if "further inspection" in notes.lower() else "No action required."

            # Set timestamp only if it's a video; otherwise, use "N/A"
            timestamp = "N/A"
            if "home_inspection.mp4" in media_reference:
                # Try to extract timestamp from notes or mediaReference if available
                timestamp_match = re.search(r'(\d{1,2}:\d{2}(?:-\d{1,2}:\d{2})?)', notes)
                if timestamp_match:
                    timestamp = timestamp_match.group(1)
                else:
                    # Fallback: Check mediaReference for timestamp (e.g., "home_inspection.mp4 0:22-0:32")
                    media_match = re.search(r'home_inspection.mp4 (\d{1,2}:\d{2}(?:-\d{1,2}:\d{2})?)', media_reference)
                    timestamp = media_match.group(1) if media_match else "N/A"
                st.write(f"Extracted timestamp for {section_name}: {timestamp}")  # Debug log

            transformed["detailedInspection"].append({
                "area": section_name,
                "mediaReference": media_reference,  # Single media reference for all items
                "timestamp": timestamp,  # Use actual timestamp if found, otherwise "N/A"
                "condition": f"{condition} - {notes.split('.')[0]}" if notes else condition,
                "complianceStatus": compliance_status,
                "issuesFound": issues_found,
                "referenceDoc": "Australian Building Code (ABC) Housing Provisions",  # Default reference
                "referenceSection": "Clause 3.1.1",  # Default section
                "recommendation": recommendation
            })

            # Add to maintenance notes if non-compliant
            if compliance_status == "Non-compliant":
                transformed["maintenanceNotes"]["recurringIssues"].append(f"{section_name} requires attention")
                transformed["maintenanceNotes"]["preventiveRecommendations"].append(f"Regular inspection of {section_name} for maintenance")
                transformed["maintenanceNotes"]["maintenanceSchedule"].append({
                    "frequency": "Quarterly" if condition.lower() != "poor" else "Immediate",
                    "tasks": [f"Inspect and maintain {section_name}"]
                })

    # Update executive summary from Overall Condition section
    for section in response_json.get("sections", []):
        if section["section_name"] == "Overall Condition":
            details_list = section.get("details", [])
            if details_list:
                details = details_list[0]  # Take the first detail for overall condition (if multiple exist)
                overall_notes = details.get("notes", "")
                transformed["executiveSummary"]["overallCondition"] = details.get("condition", "Good")
                if "recommended" in overall_notes.lower():
                    transformed["executiveSummary"]["recommendedActions"].append(overall_notes.split("recommended")[1].strip())
                if "requires" in overall_notes.lower() or "issues" in overall_notes.lower():
                    transformed["executiveSummary"]["criticalIssues"].append(overall_notes.split("requires")[1].strip() if "requires" in overall_notes.lower() else overall_notes.split("issues")[1].strip())

    return transformed

# Main Streamlit App
def main():
    st.set_page_config(page_title="Home Inspection Report Dashboard", layout="wide", page_icon="🏠")
    
    # Custom CSS for a professional design (replacing #34495e with #ffffff)
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
            color: #ffffff;  /* Changed from #34495e to #ffffff */
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
        </style>
    """, unsafe_allow_html=True)

    # Title and Description
    st.markdown("<h1 class='header'>🏠 Home Inspection Report Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("""
        <p style='text-align: center; color: #ffffff; font-size: 16px; margin-bottom: 30px;'>
            Upload photos or videos of your home, and let our AI analyze them against Australian building standards for a detailed, professional inspection report.
        </p>
    """, unsafe_allow_html=True)

    # File Upload Section
    st.markdown("<h2 class='subheader'>Upload Your Home Media</h2>", unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload photos or videos", 
        accept_multiple_files=True, 
        type=["jpg", "jpeg", "png", "mp4"],
        help="Supported formats: JPG, JPEG, PNG, MP4"
    )

    if uploaded_files:
        with st.spinner("Analyzing your media... Please wait."):
            # Reset document_dict for each new upload to avoid caching issues
            global document_dict
            document_dict = {"user_data": {}}
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

            # Generate Inspection Report
            prompt = (
                "Analyze the uploaded photos and videos of the building and generate a detailed inspection report in JSON format. "
                "Return the response as a valid JSON object with 'report_details' and 'sections', including section_name and details with condition and notes."
            )
            content = [{"text": prompt}]
            for name, doc in document_dict["user_data"].items():
                content.append({"text": f"User Document: {name}"})
                content.append(doc)

            chat_session = model.start_chat(history=[{"role": "user", "parts": content}])
            try:
                response = chat_session.send_message("Please generate a detailed building report.")
                st.write("Raw response:", response.text)  # Debug output
                # Save raw response to JSON file
                with open(JSON_RAW_PATH, "w") as f:
                    json.dump(json.loads(response.text), f, indent=4)
                response_json = json.loads(response.text)
                if not isinstance(response_json, dict) or "sections" not in response_json:
                    st.error("Invalid report format received from the API.")
                    return
                # Transform the response to match the expected structure
                transformed_response = transform_response(response_json)
                # Save transformed response to JSON file
                with open(JSON_REPORT_PATH, "w") as f:
                    json.dump(transformed_response, f, indent=4)
            except json.JSONDecodeError as e:
                st.error(f"Failed to parse response as JSON: {str(e)}")
                st.write("Response content:", response.text)
                return
            except Exception as e:
                st.error(f"Error generating report: {str(e)}")
                return

            # Extract timestamps for video frames (if applicable)
            timestamps = []
            for item in transformed_response["detailedInspection"]:
                if "timestamp" in item and "home_inspection.mp4" in item["mediaReference"] and item["timestamp"] != "N/A":
                    timestamps.extend([ts.strip() for ts in item["timestamp"].split(",")])
            frame_paths = extract_video_frames(video_path, timestamps) if video_path and timestamps else {}

        # Load and display the transformed report from JSON file
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

        # Display Report
        st.markdown("<h2 class='subheader'>Inspection Report</h2>", unsafe_allow_html=True)
        for item in transformed_response["detailedInspection"]:
            with st.container():
                st.markdown(f"<div class='report-card'>", unsafe_allow_html=True)
                
                # Area and Compliance Status
                status_class = "status-non-compliant" if item["complianceStatus"] == "Non-compliant" else "status-compliant"
                st.markdown(f"""
                    <h3 class='header' style='margin-bottom: 15px;'>{item['area']}</h3>
                    <span class='{status_class}' style='margin-bottom: 15px;'>{item['complianceStatus']}</span>
                """, unsafe_allow_html=True)

                # No image display in report cards (as per previous request)
                # Details
                st.markdown(f"<p style='color: #ffffff; line-height: 1.6;'><strong>Condition:</strong> {item['condition']}</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='color: #ffffff; line-height: 1.6;'><strong>Reference:</strong> {item.get('referenceDoc', 'N/A')} - {item.get('referenceSection', 'N/A')}</p>", unsafe_allow_html=True)
                if item.get("issuesFound"):
                    st.markdown("<p style='color: #ffffff; line-height: 1.6;'><strong>Issues Found:</strong></p>", unsafe_allow_html=True)
                    for issue in item["issuesFound"]:
                        st.markdown(f"<p style='color: #ffffff; line-height: 1.6; margin-left: 20px;'>- {issue}</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='color: #ffffff; line-height: 1.6;'><strong>Recommendation:</strong> {item.get('recommendation', 'N/A')}</p>", unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)

        # Executive Summary
        st.markdown("<h2 class='subheader'>Executive Summary</h2>", unsafe_allow_html=True)
        summary = transformed_response["executiveSummary"]
        st.markdown(f"<p style='color: #ffffff; font-size: 18px; line-height: 1.6;'><strong>Overall Condition:</strong> {summary['overallCondition']}</p>", unsafe_allow_html=True)
        st.markdown("<p style='color: #ffffff; font-size: 18px; line-height: 1.6;'><strong>Critical Issues:</strong></p>", unsafe_allow_html=True)
        for issue in summary["criticalIssues"]:
            st.markdown(f"<p style='color: #ffffff; font-size: 16px; line-height: 1.6; margin-left: 20px;'>- {issue}</p>", unsafe_allow_html=True)
        st.markdown("<p style='color: #ffffff; font-size: 18px; line-height: 1.6;'><strong>Recommended Actions:</strong></p>", unsafe_allow_html=True)
        for action in summary["recommendedActions"]:
            st.markdown(f"<p style='color: #ffffff; font-size: 16px; line-height: 1.6; margin-left: 20px;'>- {action}</p>", unsafe_allow_html=True)

        # Maintenance Schedule (reverted to st.table for simplicity)
        st.markdown("<h2 class='subheader'>Maintenance Schedule</h2>", unsafe_allow_html=True)
        maintenance_schedule = parse_maintenance_schedule(transformed_response)
        if maintenance_schedule:
            st.table(maintenance_schedule)
        else:
            st.warning("No maintenance tasks generated for this report.")

        # Maintenance Notes
        st.markdown("<h2 class='subheader'>Maintenance Notes</h2>", unsafe_allow_html=True)
        notes = transformed_response["maintenanceNotes"]
        st.markdown("<p style='color: #ffffff; font-size: 18px; line-height: 1.6;'><strong>Recurring Issues:</strong></p>", unsafe_allow_html=True)
        for issue in notes["recurringIssues"]:
            st.markdown(f"<p style='color: #ffffff; font-size: 16px; line-height: 1.6; margin-left: 20px;'>- {issue}</p>", unsafe_allow_html=True)
        st.markdown("<p style='color: #ffffff; font-size: 18px; line-height: 1.6;'><strong>Preventive Recommendations:</strong></p>", unsafe_allow_html=True)
        for rec in notes["preventiveRecommendations"]:
            st.markdown(f"<p style='color: #ffffff; font-size: 16px; line-height: 1.6; margin-left: 20px;'>- {rec}</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
