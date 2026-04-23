from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
import asyncio
import subprocess
from typing import Dict, Any
from pathlib import Path
from typing import Dict, Any, Optional
from websocket import WebSocket
from src.multi_agents_orch import trigger_orchestrator
from src.utils.agents_tools import run_resolution_scripts, run_resolution_scripts_api
from src.utils.common_utils import check_resolution_found, find_relevant_scripts, resolve_incident, get_actions, \
    format_resolution_to_render, get_script_content
from datetime import datetime
import re
import uuid

from src.utils.data_pipeline import sync_data_from_app, test_connection_to_app
from src.utils.rag_utils import fine_tune_user_prompt
from src.utils.zephyr_upload import upload_test_cases_to_zephyr
from src.utils.jira_upload import upload_user_stories_to_jira
import csv
import json
from fastapi import Query
from fastapi.responses import JSONResponse
from typing import List
import time,random
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Load Confluence environment variables
try:
    # Try to load from confluence directory first
    confluence_env_path = Path(__file__).parent.parent / "confluence" / "env"
    if confluence_env_path.exists():
        load_dotenv(confluence_env_path)
        print(f"Loaded Confluence config from: {confluence_env_path}")
    else:
        # Fallback to default .env
        load_dotenv()
        print("Loaded default environment config")

    CONFLUENCE_URL = os.getenv("CONFLUENCE_URL")
    CONFLUENCE_USERNAME = os.getenv("CONFLUENCE_USERNAME")
    CONFLUENCE_API_TOKEN = os.getenv("CONFLUENCE_API_TOKEN")

    print(f"Confluence config loaded - URL: {CONFLUENCE_URL}, Username: {CONFLUENCE_USERNAME}")
except Exception as e:
    print(f"Error loading Confluence config: {e}")
    CONFLUENCE_URL = None
    CONFLUENCE_USERNAME = None
    CONFLUENCE_API_TOKEN = None


# Confluence API Functions
def fetch_all_pages_paginated(space_key):
    """Fetch all pages in a Confluence space (paginated)"""
    if not all([CONFLUENCE_URL, CONFLUENCE_USERNAME, CONFLUENCE_API_TOKEN]):
        raise Exception("Confluence credentials not configured")

    url = f"{CONFLUENCE_URL}/rest/api/content"
    params = {
        "type": "page",
        "spaceKey": space_key,
        "expand": "title,version,body.view",
        "limit": 25  # Fetch 25 pages per request
    }

    pages = []
    while url:
        response = requests.get(
            url,
            params=params,
            auth=(CONFLUENCE_USERNAME, CONFLUENCE_API_TOKEN)
        )

        if response.status_code == 200:
            data = response.json()
            pages.extend(data['results'])  # Append the fetched pages

            # Check for the next page of results
            url = data['_links'].get('next', None)
            params = {}  # Reset params after the first request
        else:
            raise Exception(f"Failed to fetch pages from space '{space_key}': {response.status_code} - {response.text}")

    return pages


def fetch_confluence_page(page_id):
    """Fetch content from a specific Confluence page"""
    if not all([CONFLUENCE_URL, CONFLUENCE_USERNAME, CONFLUENCE_API_TOKEN]):
        raise Exception("Confluence credentials not configured")

    url = f"{CONFLUENCE_URL}/rest/api/content/{page_id}?expand=body.view"
    response = requests.get(
        url,
        auth=(CONFLUENCE_USERNAME, CONFLUENCE_API_TOKEN)
    )

    if response.status_code == 200:
        data = response.json()
        if 'body' in data and 'view' in data['body']:
            html_content = data['body']['view']['value']
            return html_content  # Return the rendered HTML content
        else:
            print("Error: Body content not found in the response.")
            return None
    else:
        raise Exception(f"Failed to fetch page {page_id}: {response.status_code} - {response.text}")


def format_html_to_text(html_content):
    """Clean and format HTML content into plain text"""
    soup = BeautifulSoup(html_content, 'html.parser')
    # Remove any unwanted tags (like scripts, styles, etc.)
    for script in soup(['script', 'style']):
        script.decompose()

    # Get the text content
    text_content = soup.get_text(separator='\n', strip=True)
    return text_content


def normalize_test_case(tc: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize various incoming test case shapes into a canonical dictionary.

    The canonical shape contains these keys (best-effort):
    - id, title, description, preconditions, steps (list), testData, expectedResult,
      passFailCriteria, notes, priority, status, labels, customFields, estimatedTime
    """
    if tc is None:
        return {}

    # Helper to safely get values from nested customFields dict
    custom = tc.get('customFields') if isinstance(tc.get('customFields'), dict) else {}

    # Ensure an id exists
    tc_id = tc.get('id') or tc.get('testId') or tc.get('name') or str(uuid.uuid4())

    # Steps: prefer list, otherwise split on newlines; also accept common custom field names
    raw_steps = tc.get('steps') or tc.get('testSteps') or custom.get('Test Steps') or custom.get('Test Steps (list)') or tc.get('steps_text') or tc.get('stepsString')
    steps: list = []
    if raw_steps:
        if isinstance(raw_steps, list):
            steps = [str(s).strip() for s in raw_steps if str(s).strip()]
        else:
            # split on newlines or numbered lists
            if isinstance(raw_steps, str):
                lines = [l.strip() for l in raw_steps.splitlines() if l.strip()]
                steps = lines
            else:
                steps = [str(raw_steps)]

    normalized = {
        'id': tc_id,
        'title': tc.get('title') or tc.get('name') or tc.get('summary') or '',
        'description': tc.get('description') or tc.get('objective') or tc.get('expectedResult') or '',
        'preconditions': tc.get('preconditions') or tc.get('precondition') or tc.get('preconditions_text') or '',
        'steps': steps,
        'testData': tc.get('testData') or custom.get('Test Data') or '',
        'expectedResult': tc.get('expectedResult') or custom.get('Expected Result') or '',
        'actualResult': tc.get('actualResult') or custom.get('Actual Result') or '',
        'passFailCriteria': tc.get('passFailCriteria') or custom.get('Pass/Fail Criteria') or '',
        'notes': tc.get('notes') or custom.get('Notes') or '',
        'postconditions': tc.get('postconditions') or custom.get('Postconditions') or '',
        'testedBy': tc.get('testedBy') or custom.get('Tested By') or '',
        'testDate': tc.get('testDate') or custom.get('Test Date') or '',
        'priority': tc.get('priority') or tc.get('priorityName') or '',
        'status': tc.get('status') or tc.get('statusName') or '',
        'labels': tc.get('labels') or [],
        'customFields': custom,
        'estimatedTime': tc.get('estimatedTime') or '',
    }

    # Preserve any other top-level fields that may be useful
    for k, v in tc.items():
        if k not in normalized:
            normalized.setdefault('extra', {})
            normalized['extra'][k] = v

    return normalized


def save_confluence_articles_to_files(space_key, output_dir="confluence_articles"):
    """Fetch all pages and save them as separate .txt files"""
    try:
        # Create output directory if it doesn't exist
        output_path = Path(__file__).parent / output_dir
        output_path.mkdir(exist_ok=True)

        pages = fetch_all_pages_paginated(space_key)
        saved_articles = []

        for page in pages:
            page_id = page['id']
            page_title = page['title']
            print(f"Fetching content for page: {page_title} (ID: {page_id})")

            # Get content from the API response if available, otherwise fetch separately
            if 'body' in page and 'view' in page['body']:
                html_content = page['body']['view']['value']
            else:
                html_content = fetch_confluence_page(page_id)

            if html_content:
                # Format the content to plain text
                formatted_content = format_html_to_text(html_content)

                # Create safe filename
                safe_title = "".join(c for c in page_title if c.isalnum() or c in (' ', '-', '_')).rstrip()
                file_name = f"{safe_title}.txt"
                file_path = output_path / file_name

                # Save the content to a text file
                with file_path.open('w', encoding='utf-8') as file:
                    file.write(formatted_content)

                print(f"Content saved to '{file_path}'")

                saved_articles.append({
                    "id": page_id,
                    "title": page_title,
                    "file_path": str(file_path),
                    "content_preview": formatted_content[:200] + "..." if len(
                        formatted_content) > 200 else formatted_content
                })
            else:
                print(f"Failed to retrieve content for page '{page_title}'")

        return saved_articles
    except Exception as e:
        print(f"Error fetching or saving pages: {e}")
        raise



# Helper functions for sync operations
async def sync_servicenow_data():
    """
    Sync data from ServiceNow using MCP client or fallback to CSV
    """
    try:
        # Try to use MCP client first
        from src.mcp.servicenow_mcp_client import call_tool_with_params
        import asyncio
        
        print("Attempting ServiceNow MCP sync...")
        # Call the MCP function
        response = await call_tool_with_params("list_incidents", {"limit": 200})
        
        return {
            "success": True,
            "connected": True,
            "records_count": getattr(response, 'records_count', 200),
            "message": "Successfully synced from ServiceNow MCP",
            "errors": []
        }
    except Exception as mcp_error:
        print(f"MCP sync failed: {mcp_error}, falling back to CSV validation")
        
        # Fallback: Validate CSV file exists and return its record count
        try:
            csv_path = Path(__file__).parent / "data" / "csv" / "servicenow_incidents_list.csv"
            if csv_path.exists():
                # Count records in CSV
                with csv_path.open(newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    record_count = sum(1 for _ in reader)
                
                return {
                    "success": True,
                    "connected": True,  # CSV is available
                    "records_count": record_count,
                    "message": f"Using local CSV data with {record_count} records",
                    "errors": [f"MCP connection failed: {str(mcp_error)}"]
                }
            else:
                return {
                    "success": False,
                    "connected": False,
                    "records_count": 0,
                    "message": "No data source available",
                    "errors": [f"MCP failed: {str(mcp_error)}", "CSV file not found"]
                }
        except Exception as csv_error:
            return {
                "success": False,
                "connected": False,
                "records_count": 0,
                "message": "Sync failed completely",
                "errors": [f"MCP error: {str(mcp_error)}", f"CSV error: {str(csv_error)}"]
            }


async def sync_confluence_data():
    """
    Sync data from Confluence using real API integration
    """
    try:
        # Check if Confluence is configured
        if not all([CONFLUENCE_URL, CONFLUENCE_USERNAME, CONFLUENCE_API_TOKEN]):
            return {
                "success": False,
                "connected": False,
                "records_count": 0,
                "message": "Confluence credentials not configured",
                "errors": ["Missing CONFLUENCE_URL, CONFLUENCE_USERNAME, or CONFLUENCE_API_TOKEN"]
            }

        print("Attempting Confluence sync...")

        # Test connection first
        test_url = f"{CONFLUENCE_URL}/rest/api/space"
        response = requests.get(
            test_url,
            auth=(CONFLUENCE_USERNAME, CONFLUENCE_API_TOKEN),
            timeout=10
        )

        if response.status_code != 200:
            return {
                "success": False,
                "connected": False,
                "records_count": 0,
                "message": "Failed to connect to Confluence",
                "errors": [f"HTTP {response.status_code}: {response.text}"]
            }

        # Fetch articles from default space (KB)
        try:
            pages = fetch_all_pages_paginated("KB")
            records_count = len(pages)

            return {
                "success": True,
                "connected": True,
                "records_count": records_count,
                "message": f"Successfully connected to Confluence and found {records_count} articles",
                "errors": []
            }
        except Exception as fetch_error:
            return {
                "success": False,
                "connected": True,  # Connection works but fetch failed
                "records_count": 0,
                "message": "Connected to Confluence but failed to fetch articles",
                "errors": [f"Fetch error: {str(fetch_error)}"]
            }

    except Exception as e:
        return {
            "success": False,
            "connected": False,
            "records_count": 0,
            "message": "Confluence sync failed completely",
            "errors": [f"Error: {str(e)}"]
        }

# ---------- Jira integration (similar style to Confluence) ----------
# Load Jira environment variables
try:
    # Try to load from jira directory first
    jira_env_path = Path(__file__).parent.parent / "jira" / "env"
    if jira_env_path.exists():
        load_dotenv(jira_env_path)
        print(f"Loaded Jira config from: {jira_env_path}")
    else:
        # Fallback to default .env (already loaded)
        print("Using default environment config for Jira")

    JIRA_URL = os.getenv("JIRA_URL")
    JIRA_USERNAME = os.getenv("JIRA_USERNAME")
    JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")

    print(f"Jira config loaded - URL: {JIRA_URL}, Username: {JIRA_USERNAME}")
except Exception as e:
    print(f"Error loading Jira config: {e}")
    JIRA_URL = None
    JIRA_USERNAME = None
    JIRA_API_TOKEN = None


def fetch_user_stories_from_project(project_key: str, max_results: int = 50):
    """Fetch user stories (issuetype = Story) from Jira using the JQL search endpoint.
    Returns a list of issue objects as returned by Jira REST API.
    """
    if not all([JIRA_URL, JIRA_USERNAME, JIRA_API_TOKEN]):
        raise Exception("Jira credentials not configured")

    url = f"{JIRA_URL}/rest/api/3/search/jql"

    # Define the JQL query in the body of the request
    jql_query = {
        "jql": f"project = {project_key} AND issuetype = Story",
        "fields": ["summary", "description", "status", "priority", "assignee"],
        "maxResults": max_results
    }

    user_stories = []

    while url:
        response = requests.post(
            url,
            json=jql_query,
            auth=(JIRA_USERNAME, JIRA_API_TOKEN)
        )

        if response.status_code == 200:
            data = response.json()
            user_stories.extend(data['issues'])

            # Check for the next page of results
            url = data.get('nextPage', None)
        else:
            print(f"Error response: {response.text}")
            raise Exception(f"Failed to fetch user stories from project '{project_key}': {response.status_code}")

    print(f"Total issues fetched: {len(user_stories)}")
    return user_stories


def extract_text_from_description(description):
    """Extract a plain-text string from Jira rich-text description payload or return string as-is."""
    if not description:
        return 'No description available'

    if isinstance(description, dict):
        parts = []

        def walk(content):
            if not content:
                return
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get('type') == 'text':
                    parts.append(item.get('text', ''))
                elif item.get('content'):
                    walk(item['content'])

        if 'content' in description:
            walk(description['content'])

        return ' '.join(parts).strip()

    return str(description)


def save_user_stories_to_csv(file_path: str, user_stories):
    """Write a list of Jira issues to CSV (Story Key, Summary, Status, Priority, Assignee, Description)."""
    fieldnames = ['Story Key', 'Summary', 'Status', 'Priority', 'Assignee', 'Description']
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for s in user_stories:
            key = s.get('key', '')
            fields = s.get('fields', {})
            summary = fields.get('summary', '')
            status = fields.get('status', {}).get('name', '')
            priority = fields.get('priority', {}).get('name', '') if isinstance(fields.get('priority', {}), dict) else str(fields.get('priority', ''))
            assignee = fields.get('assignee', {}).get('displayName', 'Unassigned') if isinstance(fields.get('assignee', {}), dict) else str(fields.get('assignee', ''))
            description = extract_text_from_description(fields.get('description'))

            writer.writerow({
                'Story Key': key,
                'Summary': summary,
                'Status': status,
                'Priority': priority,
                'Assignee': assignee,
                'Description': description
            })

def save_user_stories_to_json(file_path: str, user_stories):
    """Write a list of Jira issues to JSON file."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    user_stories_data = []

    for story in user_stories:
        description_raw = story['fields'].get('description', 'No description available')
        description_text = extract_text_from_description(description_raw)

        # Get priority
        priority = story['fields'].get('priority', {})
        priority_name = priority.get('name', 'No Priority') if isinstance(priority, dict) else str(priority)

        # Get assignee
        assignee = story['fields'].get('assignee', {})
        assignee_name = assignee.get('displayName', 'Unassigned') if isinstance(assignee, dict) else str(assignee)

        user_stories_data.append({
            'Story Key': story['key'],
            'Summary': story['fields']['summary'],
            'Status': story['fields']['status']['name'],
            'Priority': priority_name,
            'Assignee': assignee_name,
            'Description': description_text
        })

    # Write to JSON file
    with open(file_path, 'w', encoding='utf-8') as jsonfile:
        json.dump(user_stories_data, jsonfile, ensure_ascii=False, indent=4)

    print(f"JSON file saved: {file_path}")


def get_jira_user_stories_impl(project_key: Optional[str] = Query("HB", description="Jira project key"),
                         max_results: int = Query(50, description="Max results per page"),
                         save_csv: bool = Query(True, description="Save results to CSV in ./data")):
    """Return user stories for a Jira project and optionally save to CSV.
    Example: /api/jira/user-stories?project_key=HB&max_results=50&save_csv=true
    """
    try:
        if not all([JIRA_URL, JIRA_USERNAME, JIRA_API_TOKEN]):
            return JSONResponse({"success": False, "message": "Jira credentials not configured"}, status_code=400)

        issues = fetch_user_stories_from_project(project_key, max_results)

        simplified = []
        for s in issues:
            f = s.get('fields', {})
            simplified.append({
                'key': s.get('key', ''),
                'summary': f.get('summary', ''),
                'status': f.get('status', {}).get('name', ''),
                'priority': f.get('priority', {}).get('name', '') if isinstance(f.get('priority', {}), dict) else str(f.get('priority', '')),
                'assignee': f.get('assignee', {}).get('displayName', 'Unassigned') if isinstance(f.get('assignee', {}), dict) else str(f.get('assignee', '')),
                'description': extract_text_from_description(f.get('description'))
            })

        csv_path = None
        if save_csv:
            out_dir = Path(__file__).parent / 'data'
            out_dir.mkdir(exist_ok=True)
            csv_path = out_dir / f"{project_key}_user_stories.csv"
            save_user_stories_to_csv(str(csv_path), issues)

        return JSONResponse({"success": True, "count": len(simplified), "stories": simplified, "csv_path": str(csv_path) if csv_path else None})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch Jira stories: {e}")


def sync_jira_stories_impl(project_key: Optional[str] = Query("HB", description="Jira project key"),
                     output_dir: Optional[str] = Query("jira_user_stories", description="Directory under ./data to save stories")):
    """Fetch user stories and save to CSV and individual text files under `./data/{output_dir}`."""
    try:
        print(f"Starting Jira sync for project: {project_key}, output_dir: {output_dir}")
        print(f"Jira credentials - URL: {JIRA_URL}, Username: {JIRA_USERNAME}, Token: {'***' if JIRA_API_TOKEN else 'None'}")

        if not all([JIRA_URL, JIRA_USERNAME, JIRA_API_TOKEN]):
            error_msg = "Jira credentials not configured"
            print(f"ERROR: {error_msg}")
            return JSONResponse({"success": False, "message": error_msg}, status_code=400)

        print(f"Fetching user stories from Jira project: {project_key}")
        issues = fetch_user_stories_from_project(project_key, 50)
        print(f"Found {len(issues)} issues")

        base_dir = Path(__file__).parent / 'data' / output_dir
        base_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created output directory: {base_dir}")

        # Save CSV
        csv_file = base_dir / f"{project_key}_user_stories.csv"
        print(f"Saving CSV to: {csv_file}")
        save_user_stories_to_csv(str(csv_file), issues)

        # Save JSON
        json_file = base_dir / f"{project_key}_user_stories.json"
        print(f"Saving JSON to: {json_file}")
        save_user_stories_to_json(str(json_file), issues)

        # Save individual story text files (key.txt)
        print(f"Saving individual story files...")
        for s in issues:
            key = s.get('key', 'unknown')
            fields = s.get('fields', {})
            desc = extract_text_from_description(fields.get('description'))
            summary = fields.get('summary', '')
            content = f"{summary}\n\n{desc}"
            file_path = base_dir / f"{key}.txt"
            with file_path.open('w', encoding='utf-8') as fh:
                fh.write(content)

        print(f"Jira sync completed successfully: {len(issues)} stories synced")
        return JSONResponse({"success": True, "count": len(issues), "csv": str(csv_file), "json": str(json_file), "dir": str(base_dir)})
    except Exception as e:
        error_msg = f"Failed to sync Jira stories: {str(e)}"
        print(f"ERROR in sync_jira_stories_impl: {error_msg}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=error_msg)


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/api/jira/user-stories")
def get_jira_user_stories(project_key: Optional[str] = Query("HB", description="Jira project key"),
                         max_results: int = Query(50, description="Max results per page"),
                         save_csv: bool = Query(True, description="Save results to CSV in ./data")):
    return get_jira_user_stories_impl(project_key, max_results, save_csv)


@app.post("/api/jira/sync")
def sync_jira_stories(project_key: Optional[str] = Query("HB", description="Jira project key"),
                     output_dir: Optional[str] = Query("jira_user_stories", description="Directory under ./data to save stories")):
    return sync_jira_stories_impl(project_key, output_dir)


class QueryRequest(BaseModel):
    query: str
    role: str
    testCases: Optional[list] = None

class RecommendScriptsRequest(BaseModel):
    incidentId: Optional[str] = None
    incidentType: Optional[str] = None
    aiMessage: Optional[str] = None

class ScriptExecutionRequest(BaseModel):
    scriptToExecute: str

class UpdateRequest(BaseModel):
    resolutionComment: str

class AppDataRequest(BaseModel):
    appId: str

class ZephyrUploadRequest(BaseModel):
    testCases: list
    projectKey: Optional[str] = "HB"

class JiraUploadRequest(BaseModel):
    userStories: list
    projectKey: Optional[str] = "HB"

@app.get("/")
def on_startup():
    msg = "FastAPI Server Started"
    print(msg)
    return msg

@app.get("/api/incidents")
def list_incidents(state: Optional[str] = Query(default=None, description="Filter by state, e.g., 'open' (new+in progress), 'new', 'progress'")):
    """Return incidents from CSV, optionally filtered by state (case-insensitive substring match)."""
    try:
        csv_path = Path(__file__).parent / "data" / "csv" / "servicenow_incidents_list.csv"
        if not csv_path.exists():
            raise HTTPException(status_code=404, detail=f"CSV not found at {csv_path}")

        incidents: List[dict] = []
        with csv_path.open(newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Skip empty rows
                if not row.get('Number'):
                    continue

                # Extract priority level and description
                priority_text = row.get('Priority', 'Low')
                if ' - ' in priority_text:
                    priority_level, priority_desc = priority_text.split(' - ', 1)
                    # Map priority levels to standard format
                    priority_map = {
                        '1': 'Critical',
                        '2': 'High',
                        '3': 'Medium',
                        '4': 'Low',
                        '5': 'Low'
                    }
                    priority = priority_map.get(priority_level, priority_desc)
                else:
                    priority = priority_text

                # Get state for filtering
                row_state = row.get('State', '').strip()

                # Filter by state if provided (substring match, case-insensitive)
                if state:
                    # Special handling for "open" incidents - include both New and In Progress
                    if state.lower() == "open":
                        is_open = (
                            "new" in row_state.lower() or
                            "progress" in row_state.lower() or
                            row_state.lower() == "in progress"
                        )
                        if not is_open:
                            continue
                    else:
                        if state.lower() not in row_state.lower():
                            continue

                incidents.append({
                    "id": row.get('Number', ''),
                    "openedDate": row.get('Opened', ''),
                    "priority": priority,
                    "shortDescription": row.get('Short Description', ''),
                    "description": row.get('Description', ''),
                    "status": row_state,
                    "category": row.get('Category', ''),
                    "caller": row.get('Caller', ''),
                    "assignmentGroup": row.get('Assignment Group', ''),
                    "assignedTo": row.get('Assigned To', ''),
                    "updatedBy": row.get('Updated by', ''),
                    "resolutionNotes": row.get('Resolution notes', ''),
                    "actualStart": row.get('Actual start', ''),
                    "actualEnd": row.get('Actual end', '')
                })

        return JSONResponse({"incidents": incidents})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read incidents: {e}")

@app.get("/api/notifications")
def get_notifications():
    """Return notifications from the JSON file."""
    try:
        notifications_path = Path(__file__).parent / "data" / "notifications.json"
        if not notifications_path.exists():
            raise HTTPException(status_code=404, detail=f"Notifications file not found at {notifications_path}")
        
        with notifications_path.open(encoding='utf-8') as f:
            notifications_data = json.loads(f.read())
        
        return JSONResponse(notifications_data)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read notifications: {e}")


# DEPRECATED: This is an old version - see line 1498 for the active endpoint
# @app.post("/get_chat_response")
# def fetch_agent_response_old(req: QueryRequest):
    global DEFAULT_TEST_CASES
    actions = []
    user_prompt = req.query
    current_role = req.role
    response = ""
    test_cases = None

    # If updated test cases are provided in the request, normalize and persist them
    if req.testCases is not None:
        try:
            # Normalize each incoming test case into canonical shape
            DEFAULT_TEST_CASES = [normalize_test_case(tc) for tc in req.testCases]
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid testCases payload: {e}")

    #excluding pre-defined messages
    # if "Get Resolution for Incident" != user_prompt:
    #     user_prompt=fine_tune_user_prompt(user_prompt)

    response=trigger_orchestrator(user_prompt,current_role)

    if "Test Manager" in current_role:
        # Always use the latest test cases
        try:
            test_cases = DEFAULT_TEST_CASES
        except NameError:
            test_cases = []

            if test_cases:
                response = "Test cases have been created:\n\n"
                for idx, tc in enumerate(test_cases, 1):
                    response += f"### TC- {idx}  \n"
                    # Support multiple possible field names coming from frontend/backends
                    name = tc.get('name') or tc.get('title') or tc.get('id')
                    if name:
                        response += f"**Name:** {name}  \n"

                    objective = tc.get('objective') or tc.get('description') or tc.get('expectedResult')
                    if objective:
                        response += f"**Objective/Description:** {objective}  \n"

                    precond = tc.get('precondition') or tc.get('preconditions') or tc.get('preconditions_text') or tc.get('preconditions')
                    if precond:
                        response += f"**Preconditions:** {precond}  \n"

                    if tc.get('estimatedTime'):
                        response += f"**Estimated Time:** {tc['estimatedTime']}  \n"

                    priority = tc.get('priorityName') or tc.get('priority')
                    if priority:
                        response += f"**Priority:** {priority}  \n"

                    status = tc.get('statusName') or tc.get('status')
                    if status:
                        response += f"**Status:** {status}  \n"

                    if tc.get('labels'):
                        response += f"**Labels:** {', '.join(tc['labels'])}  \n"

                    # Render steps if present
                    steps = tc.get('steps') or tc.get('testSteps') or (tc.get('customFields') or {}).get('Test Steps') if isinstance(tc.get('customFields'), dict) else None
                    if steps:
                        if isinstance(steps, list):
                            response += "**Steps:**  \n"
                            for sidx, s in enumerate(steps, 1):
                                # Render each step as a numbered line without trailing two-spaces
                                # to avoid forcing Markdown line breaks that may continue the list.
                                response += f"{sidx}. {s}\n"
                            # Add an extra blank line after the numbered list to stop Markdown list continuation
                            response += "\n"
                        else:
                            # If steps is a string, split into lines and render as a numbered list
                            if isinstance(steps, str):
                                lines = [l.strip() for l in steps.splitlines() if l.strip()]
                                if lines:
                                    response += "**Steps:**  \n"
                                    for sidx, s in enumerate(lines, 1):
                                        # remove any existing leading numbering to avoid duplication
                                        cleaned = re.sub(r'^\s*\d+\.?\s*', '', s)
                                        response += f"{sidx}. {cleaned}\n"
                                    # Add an extra blank line after the numbered list to stop Markdown list continuation
                                    response += "\n"
                                else:
                                    response += f"**Steps:** {steps}  \n\n"
                            else:
                                response += f"**Steps:** {steps}  \n\n"

                    # Test data, expected result, pass/fail, notes
                    test_data = tc.get('testData') or (tc.get('customFields') or {}).get('Test Data') if isinstance(tc.get('customFields'), dict) else None
                    if test_data:
                        response += f"**Test Data:** {test_data}  \n"

                    expected = tc.get('expectedResult') or (tc.get('customFields') or {}).get('Expected Result') if isinstance(tc.get('customFields'), dict) else None
                    if expected:
                        response += f"**Expected Result:** {expected}  \n"

                    # Actual Result
                    actual = tc.get('actualResult') or (tc.get('customFields') or {}).get('Actual Result') if isinstance(tc.get('customFields'), dict) else None
                    if actual:
                        response += f"**Actual Result:** {actual}  \n"

                    # Postconditions, Tested By, Test Date
                    postcond = tc.get('postconditions') or (tc.get('customFields') or {}).get('Postconditions') if isinstance(tc.get('customFields'), dict) else None
                    if postcond:
                        response += f"**Postconditions:** {postcond}  \n"

                    tested_by = tc.get('testedBy') or (tc.get('customFields') or {}).get('Tested By') if isinstance(tc.get('customFields'), dict) else None
                    if tested_by:
                        response += f"**Tested By:** {tested_by}  \n"

                    test_date = tc.get('testDate') or (tc.get('customFields') or {}).get('Test Date') if isinstance(tc.get('customFields'), dict) else None
                    if test_date:
                        response += f"**Test Date:** {test_date}  \n"

                    passfail = tc.get('passFailCriteria') or (tc.get('customFields') or {}).get('Pass/Fail Criteria') if isinstance(tc.get('customFields'), dict) else None
                    if passfail:
                        response += f"**Pass/Fail Criteria:** {passfail}  \n"

                    notes = tc.get('notes') or (tc.get('customFields') or {}).get('Notes') if isinstance(tc.get('customFields'), dict) else None
                    if notes:
                        response += f"**Notes:** {notes}  \n"

                    # Custom fields as bullet points, no label, with a blank line before
                    if tc.get('customFields') and isinstance(tc.get('customFields'), dict):
                        response += "\n"
                        for k, v in tc['customFields'].items():
                            # skip fields already shown
                            if k in ['Test Data', 'Test Steps', 'Expected Result', 'Pass/Fail Criteria', 'Notes', 'Actual Result', 'Postconditions', 'Tested By', 'Test Date']:
                                continue
                            response += f"- **{k}:** {v}  \n"
                response += "\n"
        else:
            response = "No test cases found."

        actions = [
            {"label": "Review TC", "query": "Review TC"},
            {"label": "Review TD", "query": "Review TD"},
            {"label": "Load Dataset", "query": "Load Dataset"},
            {"label": "Export", "query": "Export"}
        ]

    if "L1/L2" in current_role:
        if user_prompt !="Investigate Further" and "close ticket" not in user_prompt:
            pass

    result = {"answer": response, "actions": actions}
    if test_cases is not None:
        print(f"[Backend] Returning {len(test_cases)} test cases to frontend")
        print(f"[Backend] First test case returned: id={test_cases[0].get('id')}, title={test_cases[0].get('title') or test_cases[0].get('name')}")
        result["testCases"] = test_cases
    else:
        print(f"[Backend] No test cases to return")

    return result

    responses = []
    print(f"Reading recently used file from: {csv_path}")

    try:
        with csv_path.open(newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            row_count = 0
            for row in reader:
                row_count += 1
                #print(f"Processing row {row_count}: {row}")

                # Skip empty rows
                if not any(row.values()):
                    continue

                # Clean up the data
                cleaned_row = {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in row.items()}

                # Get the response text
                response_text = cleaned_row.get("Recently_Used") or cleaned_row.get("recently_used") or ""

                # Skip rows with empty response text
                if not response_text:
                    continue

                responses.append({
                    "id": len(responses) + 1,  # Generate simple ID
                    "response": response_text,
                    "dateTime": cleaned_row.get("Date_Time") or cleaned_row.get("date_time") or ""
                })

        print(f"Found {len(responses)} valid recently used responses")

        # Sort by date/time (most recent first) if dateTime is available
        responses.sort(key=lambda x: x["dateTime"], reverse=True)

        # Apply limit if specified
        if limit is not None and limit > 0:
            responses = responses[:limit]

        return JSONResponse({
            "responses": responses,
            "total": len(responses)
        })
    except HTTPException:
        raise
    except Exception as e:
        # print(f"Error in get_recently_used: {e}")
        # raise HTTPException(status_code=500, detail=f"Failed to read recently used responses: {e}")
        pass


@app.get("/api/incidents/category-distribution")
def get_category_distribution():
    """Return category distribution counts from the CSV file."""
    try:
        csv_path = Path(__file__).parent / "data" / "csv" / "servicenow_incidents_list.csv"
        if not csv_path.exists():
            raise HTTPException(status_code=404, detail=f"CSV not found at {csv_path}")

        category_counts = {}
        total_count = 0
        
        with csv_path.open(newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Skip empty rows
                if not row.get('Number'):
                    continue

                # Get category from the new CSV structure
                category = row.get('Category', 'Unknown').strip()
                
                if category:
                    category_counts[category] = category_counts.get(category, 0) + 1
                    total_count += 1

        # Sort categories by count (descending) for better visualization
        sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Define colors for different categories
        category_colors = {
            "Software": "#818cf8",      # indigo
            "Security": "#fb923c",      # orange
            "Infrastructure": "#4ade80", # green
            "Performance": "#f472b6",    # pink
            "Hardware": "#94a3b8",       # slate
            "Network": "#06b6d4",        # cyan
            "Database": "#8b5cf6",       # violet
            "Unknown": "#9ca3af"         # gray
        }
        
        # Create distribution data for pie chart
        distribution = []
        for category, count in sorted_categories:
            color = category_colors.get(category, "#94a3b8")  # Default to slate
            distribution.append({
                "name": category,
                "value": count,
                "color": color
            })

        return JSONResponse({"distribution": distribution, "total": total_count})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read category distribution: {e}")

@app.get("/api/incidents/status-distribution")
def get_status_distribution():
    """Return status distribution counts from the CSV file."""
    try:
        csv_path = Path(__file__).parent / "data" / "csv" / "servicenow_incidents_list.csv"
        if not csv_path.exists():
            raise HTTPException(status_code=404, detail=f"CSV not found at {csv_path}")

        status_counts = {}
        total_count = 0
        
        with csv_path.open(newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Skip empty rows
                if not row.get('Number'):
                    continue

                # Get state from the new CSV structure
                row_state = row.get('State', 'Unknown').strip()
                
                # Use actual state values from CSV
                if row_state:
                    status_counts[row_state] = status_counts.get(row_state, 0) + 1
                    total_count += 1

        # Sort states by count (descending) for better visualization
        sorted_states = sorted(status_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Create distribution with actual state names from CSV
        distribution = [{"label": "Total", "count": total_count, "color": "bg-indigo-500", "colorHex": "#6366f1"}]
        
        # Define colors for different states using hex values for better compatibility
        state_colors = {
            "1 - New": {"class": "bg-yellow-400", "hex": "#fbbf24"},
            "2 - In Progress": {"class": "bg-blue-500", "hex": "#3b82f6"}, 
            "3 - On Hold": {"class": "bg-orange-400", "hex": "#fb923c"},
            "4 - Pending": {"class": "bg-purple-400", "hex": "#a855f7"},
            "5 - Planning": {"class": "bg-pink-400", "hex": "#f472b6"},
            "6 - Resolved": {"class": "bg-green-500", "hex": "#10b981"},
            "7 - Closed": {"class": "bg-gray-400", "hex": "#9ca3af"},
            "Closed": {"class": "bg-gray-500", "hex": "#6b7280"},
            "Resolved": {"class": "bg-green-500", "hex": "#10b981"},
            "Open": {"class": "bg-yellow-400", "hex": "#fbbf24"},
            "In Progress": {"class": "bg-blue-500", "hex": "#3b82f6"}
        }
        
        # Consolidate similar states and add to distribution
        consolidated_states = {}

        for state, count in sorted_states:
            # Normalize and consolidate similar states
            if state == "1 - New":
                display_label = "New"
                color_key = "1 - New"
            elif state == "2 - In Progress" or state == "In Progress":
                display_label = "In Progress"
                color_key = "2 - In Progress"  # Use the numbered version for color consistency
            elif state == "Resolved":
                display_label = "Resolved"
                color_key = "6 - Resolved"  # Map to numbered version
            else:
                display_label = state  # Keep original for other states
                color_key = state

            # Consolidate counts for similar states
            if display_label in consolidated_states:
                consolidated_states[display_label]["count"] += count
            else:
                color_info = state_colors.get(color_key, {"class": "bg-slate-500", "hex": "#64748b"})
                consolidated_states[display_label] = {
                    "label": display_label,
                    "count": count,
                    "color": color_info["class"],
                    "colorHex": color_info["hex"]
                }

        # Sort consolidated states by count (descending) and add to distribution
        sorted_consolidated = sorted(consolidated_states.values(), key=lambda x: x["count"], reverse=True)
        distribution.extend(sorted_consolidated)

        return JSONResponse({"distribution": distribution})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read status distribution: {e}")

@app.post("/api/recommended-scripts")
async def get_recommended_scripts(request: RecommendScriptsRequest):
    """Returns recommended scripts based on the current incident context."""
    print("In get recommended scripts api")
    incident_id = request.incidentId
    incident_type = request.incidentType
    aiResolution = request.aiMessage

    print("paarms",incident_id,incident_type,aiResolution)
    #recommended_scripts={"recommendedScripts":["reset-password", "unlock-account"]}
    #resolution_desc="Password Reset"
    script_ids=find_relevant_scripts(aiResolution)
    recommended_scripts = {"recommendedScripts": script_ids}
    return recommended_scripts

@app.post("/api/execute-scripts")
async def run_selected_script(execution_request:ScriptExecutionRequest):
    """Returns recommended scripts based on the current incident context."""
    print("In execute recommended scripts api")
    print(f"Received request: {execution_request}")
    status=run_resolution_scripts_api(execution_request.scriptToExecute)
    return status


@app.post("/api/sync-app")
def sync_data_from_tool(req: AppDataRequest):
    print("In sync data backend api, appId:", req.appId)
    sync_info= sync_data_from_app(req.appId.lower())
    return sync_info

@app.post("/api/test-connection")
def test_app_connection(req: AppDataRequest):
    print("In test connection backend api",req.appId)
    connection_info=test_connection_to_app(req.appId.lower())
    #connection_info = {"connected": True, "lastSynced": "Today, 10:45 AM"}
    return connection_info

@app.post("/api/update-incident-status")
def update_incident_status(req: UpdateRequest):
    print("In update status backend api",req.resolutionComment)
    status=resolve_incident(req.resolutionComment)
    return status


@app.post("/api/get-script-content")
def update_incident_status(req: ScriptExecutionRequest):
    print("In preview script content api",req.scriptToExecute)
    content=get_script_content(req.scriptToExecute)
    return content


class CreateArticleRequest(BaseModel):
    title: str
    space: str
    body: str
    visibility: str

@app.get("/api/recently-used")
def get_recently_used(limit: Optional[int] = Query(default=None, description="Limit number of recently used responses returned")):
    """Return recently used responses from the Recently_used.csv file, optionally limited to recent entries."""
    try:
        csv_path = Path(__file__).parent / "data" / "Recently_used.csv"
        if not csv_path.exists():
            print(f"Recently used file not found at {csv_path}")
            raise HTTPException(status_code=404, detail=f"Recently used file not found at {csv_path}")

        responses = []
        print(f"Reading recently used file from: {csv_path}")
        
        with csv_path.open(newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            row_count = 0
            for row in reader:
                row_count += 1
                print(f"Processing row {row_count}: {row}")
                
                # Skip empty rows
                if not any(row.values()):
                    continue
                    
                # Clean up the data
                cleaned_row = {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
                
                # Get the response text
                response_text = cleaned_row.get("Recently_Used") or cleaned_row.get("recently_used") or ""
                
                # Skip rows with empty response text
                if not response_text:
                    continue
                
                responses.append({
                    "id": len(responses) + 1,  # Generate simple ID
                    "response": response_text,
                    "dateTime": cleaned_row.get("Date_Time") or cleaned_row.get("date_time") or ""
                })

        print(f"Found {len(responses)} valid recently used responses")
        
        # Sort by date/time (most recent first) if dateTime is available
        responses.sort(key=lambda x: x["dateTime"], reverse=True)
        
        # Apply limit if specified
        if limit is not None and limit > 0:
            responses = responses[:limit]

        return JSONResponse({
            "responses": responses,
            "total": len(responses)
        })
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in get_recently_used: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to read recently used responses: {e}")

@app.get("/api/confluence/articles")
def get_confluence_articles(space_key: Optional[str] = Query("KB", description="Confluence space key"),
                          limit: Optional[int] = Query(default=None, description="Limit number of articles returned")):
    """Return Confluence articles from real API or local cache."""
    try:
        # Check if Confluence is configured
        if not all([CONFLUENCE_URL, CONFLUENCE_USERNAME, CONFLUENCE_API_TOKEN]):
            print("Confluence not configured, returning dummy data")
            # Fallback to dummy data if Confluence is not configured
            dummy_articles = [
                {
                    "id": "123456",
                    "title": "Password Reset Troubleshooting Guide",
                    "created_by": "john.doe@company.com",
                    "created_date": "2025-11-01T10:30:00Z",
                    "body": "This comprehensive guide covers step-by-step instructions for password reset procedures...",
                    "space": "IT_SUPPORT",
                    "url": f"{CONFLUENCE_URL or 'https://company.atlassian.net'}/spaces/IT_SUPPORT/pages/123456/Password+Reset+Troubleshooting+Guide"
                }
            ]

            if limit is not None and limit > 0:
                dummy_articles = dummy_articles[:limit]

            return JSONResponse({
                "articles": dummy_articles,
                "total": len(dummy_articles),
                "source": "dummy_data"
            })

        # Fetch real data from Confluence
        print(f"Fetching articles from Confluence space: {space_key}")
        pages = fetch_all_pages_paginated(space_key)

        articles = []
        for page in pages:
            # Extract body content if available
            body_text = ""
            if 'body' in page and 'view' in page['body']:
                html_content = page['body']['view']['value']
                body_text = format_html_to_text(html_content)[:500] + "..."  # Truncate for listing

            article = {
                "id": page['id'],
                "title": page['title'],
                "created_by": page.get('version', {}).get('by', {}).get('displayName', 'Unknown'),
                "created_date": page.get('version', {}).get('when', datetime.now().isoformat() + "Z"),
                "body": body_text,
                "space": space_key,
                "url": f"{CONFLUENCE_URL}/spaces/{space_key}/pages/{page['id']}/{page['title'].replace(' ', '+')}"
            }
            articles.append(article)
        
        # Sort by created date (most recent first)
        articles = sorted(articles, key=lambda x: x["created_date"], reverse=True)
        
        # Apply limit if specified
        if limit is not None and limit > 0:
            articles = articles[:limit]

        return JSONResponse({
            "articles": articles,
            "total": len(articles),
            "source": "confluence_api",
            "space_key": space_key
        })

    except Exception as e:
        print(f"Error fetching Confluence articles: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to read Confluence articles: {e}")

@app.get("/api/confluence/articles/{article_id}")
def get_confluence_article(article_id: str):
    """Get a specific Confluence article by ID."""
    try:
        # Check if Confluence is configured
        if not all([CONFLUENCE_URL, CONFLUENCE_USERNAME, CONFLUENCE_API_TOKEN]):
            print("Confluence not configured, returning dummy data")
            # Fallback to dummy data if Confluence is not configured
            dummy_articles = {
                "123456": {
                    "id": "123456",
                    "title": "Password Reset Troubleshooting Guide",
                    "created_by": "john.doe@company.com",
                    "created_date": "2025-11-01T10:30:00Z",
                    "body": "# Password Reset Troubleshooting Guide\n\nThis comprehensive guide covers step-by-step instructions for password reset procedures.\n\n## Common Issues\n1. User forgot password\n2. Account locked due to multiple failed attempts\n3. Password expired\n\n## Resolution Steps\n1. Verify user identity through security questions\n2. Check account status in Active Directory\n3. Reset password using approved tools\n4. Notify user of new temporary password\n5. Ensure user changes password on first login\n\n## Prevention Tips\n- Educate users about strong password practices\n- Implement regular password change policies\n- Use multi-factor authentication where possible",
                    "space": "IT_SUPPORT",
                    "url": f"{CONFLUENCE_URL or 'https://company.atlassian.net'}/spaces/IT_SUPPORT/pages/123456/Password+Reset+Troubleshooting+Guide"
                }
            }

            article = dummy_articles.get(article_id)
            if not article:
                raise HTTPException(status_code=404, detail="Article not found")

            return JSONResponse(article)

        # Fetch real data from Confluence
        print(f"Fetching article {article_id} from Confluence")

        # Get the article content
        html_content = fetch_confluence_page(article_id)
        if not html_content:
            raise HTTPException(status_code=404, detail="Article not found")
            
        # Convert HTML to text
        body_text = format_html_to_text(html_content)

        # Get article metadata by fetching the page info
        url = f"{CONFLUENCE_URL}/rest/api/content/{article_id}?expand=version,space"
        response = requests.get(
            url,
            auth=(CONFLUENCE_USERNAME, CONFLUENCE_API_TOKEN)
        )

        if response.status_code == 200:
            page_data = response.json()
            article = {
                "id": page_data['id'],
                "title": page_data['title'],
                "created_by": page_data.get('version', {}).get('by', {}).get('displayName', 'Unknown'),
                "created_date": page_data.get('version', {}).get('when', datetime.now().isoformat() + "Z"),
                "body": body_text,
                "space": page_data.get('space', {}).get('key', 'Unknown'),
                "url": f"{CONFLUENCE_URL}/spaces/{page_data.get('space', {}).get('key', 'Unknown')}/pages/{article_id}/{page_data['title'].replace(' ', '+')}"
            }
            return JSONResponse(article)
        else:
            raise HTTPException(status_code=404, detail="Article not found")

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error fetching article {article_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to read article: {e}")

@app.post("/confluence/sync")
def sync_confluence_articles(space_key: Optional[str] = Query("KB", description="Confluence space key to sync")):
    """Sync Confluence articles from remote and save to local files."""
    try:
        # Check if Confluence is configured
        if not all([CONFLUENCE_URL, CONFLUENCE_USERNAME, CONFLUENCE_API_TOKEN]):
            print("Confluence not configured, simulating sync")
            # Fallback to simulation if Confluence is not configured
            time.sleep(1)  # Shorter delay for simulation
            synced_count = random.randint(5, 15)
            return JSONResponse({
                "success": True,
                "message": f"Simulated sync of {synced_count} articles (Confluence not configured)",
                "synced_count": synced_count,
                "last_sync": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "source": "simulation"
            })

        print(f"Starting real Confluence sync for space: {space_key}")

        # Perform real sync with Confluence
        saved_articles = save_confluence_articles_to_files(space_key)
        synced_count = len(saved_articles)

        return JSONResponse({
            "success": True,
            "message": f"Successfully synced {synced_count} articles from Confluence space '{space_key}'",
            "synced_count": synced_count,
            "last_sync": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "space_key": space_key,
            "articles": [{"id": art["id"], "title": art["title"]} for art in saved_articles[:5]],  # Show first 5
            "source": "confluence_api"
        })

    except Exception as e:
        print(f"Confluence sync failed: {e}")
        return JSONResponse({
            "success": False,
            "message": f"Failed to sync articles: {str(e)}",
            "synced_count": 0,
            "last_sync": None
        }, status_code=500)

@app.post("/confluence/fetch_and_save_pages")
def fetch_and_save_confluence_pages(space_key: Optional[str] = Query("KB", description="Confluence space key (default: KB)")):
    """
    Fetch and save all Confluence pages in the given space as text files.
    """
    try:
        if not all([CONFLUENCE_URL, CONFLUENCE_USERNAME, CONFLUENCE_API_TOKEN]):
            raise HTTPException(status_code=400, detail="Confluence credentials not configured")

        saved_articles = save_confluence_articles_to_files(space_key)

        return JSONResponse({
            "success": True,
            "message": f"Successfully fetched and saved {len(saved_articles)} pages from space '{space_key}'.",
            "articles_count": len(saved_articles),
            "space_key": space_key,
            "articles": saved_articles
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching or saving Confluence pages: {e}")

@app.post("/confluence/articles")
def create_confluence_article(req: CreateArticleRequest):
    """Create a new article in Confluence."""
    try:
        # Simulate article creation - will be replaced with actual Confluence API integration
        import uuid
        
        # Generate dummy article data
        new_article = {
            "id": str(uuid.uuid4())[:8],
            "title": req.title,
            "created_by": "current.user@company.com",  # Would be from auth context
            "created_date": datetime.now().isoformat() + "Z",
            "body": req.body,
            "space": req.space,
            "visibility": req.visibility,
            "url": f"https://company.atlassian.net/spaces/{req.space}/pages/{str(uuid.uuid4())[:8]}/{req.title.replace(' ', '+')}"
        }
        
        # In real implementation, this would:
        # 1. Call Confluence API to create the page
        # 2. Save to local cache/database
        # 3. Return the actual created page metadata
        
        return JSONResponse({
            "success": True,
            "message": "Article created successfully",
            "article": new_article
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create article: {e}")

class TestCase(BaseModel):
    id: Optional[str] = None
    name: Optional[str] = None
    objective: Optional[str] = None
    preconditions: Optional[str] = None
    
    priorityName: Optional[str] = None
    
    statusName: Optional[str] = None
    estimatedTime: Optional[int] = None
    labels: Optional[List[str]] = None
    customFields: Optional[Dict[str, Any]] = None
    
# Global variable to store generated test cases (initially empty)
DEFAULT_TEST_CASES = []


def parse_test_cases_from_ai_response(ai_response: str) -> list:
    """Parse AI-generated test cases from text response into structured format"""
    import re
    
    print(f"\n=== PARSER DEBUG ===")
    print(f"Response length: {len(ai_response)} chars")
    
    test_cases = []
    
    # Check if response contains test case markers
    has_test_case_title = "Test Case Title:" in ai_response or "**Test Case Title:**" in ai_response
    has_numbered_tc = re.search(r'\d+\.\s+\*\*Test Case', ai_response) is not None
    
    print(f"Has 'Test Case Title:' marker: {has_test_case_title}")
    print(f"Has numbered test case pattern: {has_numbered_tc}")
    
    # Split by "---" or "Test Case" markers to identify individual test cases
    # Try splitting by --- first
    sections = re.split(r'\n---+\n|\n---+$|^---+\n', ai_response, flags=re.MULTILINE)
    
    print(f"Split by '---' resulted in {len(sections)} sections")
    
    # If no --- separators, try splitting by test case patterns
    if len(sections) <= 1:
        # Try to split by patterns like "**Test Case Title:**" or numbered test cases
        sections = re.split(r'(?=\*\*Test Case Title:\*\*|\d+\.\s+\*\*Test Case Title:\*\*|Test Case \d+)', ai_response)
        print(f"Split by test case patterns resulted in {len(sections)} sections")
    
    tc_counter = 1
    
    for idx, section in enumerate(sections):
        if not section.strip():
            continue
        
        print(f"\n--- Processing section {idx + 1} (length: {len(section)} chars) ---")
        print(f"First 200 chars: {section[:200]}")
            
        # Skip sections that don't contain test case data
        if "Test Case Title:" not in section and "Test Description:" not in section and "Title:" not in section:
            print(f"Skipping section - no test case markers found")
            continue
        
        # Extract fields using regex patterns
        test_case = {
            "id": f"TC-{tc_counter:03d}",
            "name": "",
            "objective": "",
            "precondition": "",
            "estimatedTime": 0,
            "priorityName": "Medium",
            "statusName": "Draft",
            "labels": [],
            "customFields": {
                "Test Data": "",
                "Test Steps": "",
                "Expected Result": "",
                "Actual Result": "",
                "Pass/Fail Criteria": "",
                "Postconditions": "",
                "Tested By": "",
                "Test Date": "",
                "Notes": ""
            }
        }
        
        # Extract Test Case Title (try multiple patterns)
        title_match = re.search(r'\*\*Test Case Title:\*\*\s*\n?(.+?)(?=\n\*\*|\n\n|$)', section, re.DOTALL)
        if not title_match:
            title_match = re.search(r'Test Case Title:\s*\n?(.+?)(?=\n\*\*|\nTest|\n\n|$)', section, re.DOTALL)
        if not title_match:
            title_match = re.search(r'\*\*Title:\*\*\s*\n?(.+?)(?=\n\*\*|\n\n|$)', section, re.DOTALL)
        
        if title_match:
            test_case["name"] = title_match.group(1).strip()
            print(f"✓ Extracted title: {test_case['name'][:50]}...")
        else:
            print(f"✗ Could not extract title")
        
        # Extract Test Description (maps to objective) - try multiple patterns
        desc_match = re.search(r'\*\*Test Description:\*\*\s*\n?(.+?)(?=\n\*\*|\n\n|$)', section, re.DOTALL)
        if not desc_match:
            desc_match = re.search(r'Test Description:\s*\n?(.+?)(?=\n\*\*|\nTest|\n\n|$)', section, re.DOTALL)
        if not desc_match:
            desc_match = re.search(r'\*\*Description:\*\*\s*\n?(.+?)(?=\n\*\*|\n\n|$)', section, re.DOTALL)
        
        if desc_match:
            test_case["objective"] = desc_match.group(1).strip()
            print(f"✓ Extracted description")
        
        # Extract Preconditions - try multiple patterns
        precond_match = re.search(r'\*\*Preconditions?:\*\*\s*\n?(.+?)(?=\n\*\*|\n\n|$)', section, re.DOTALL)
        if not precond_match:
            precond_match = re.search(r'Preconditions?:\s*\n?(.+?)(?=\n\*\*|\nTest|\n\n|$)', section, re.DOTALL)
        
        if precond_match:
            test_case["precondition"] = precond_match.group(1).strip()
            print(f"✓ Extracted preconditions")
        
        # Extract Priority - try multiple patterns
        priority_match = re.search(r'\*\*Priority:\*\*\s*\n?(.+?)(?=\n\*\*|\n\n|$)', section, re.DOTALL)
        if not priority_match:
            priority_match = re.search(r'Priority:\s*\n?(.+?)(?=\n\*\*|\nTest|\n\n|$)', section, re.DOTALL)
        
        if priority_match:
            priority = priority_match.group(1).strip()
            test_case["priorityName"] = priority if priority in ["High", "Medium", "Low"] else "Medium"
            print(f"✓ Extracted priority: {test_case['priorityName']}")
        
        # Extract labels from the test case (look for keywords to auto-tag)
        # You can enhance this to extract from a Labels field if AI provides it
        labels = []
        if "dark mode" in test_case["name"].lower() or "dark mode" in test_case["objective"].lower():
            labels.extend(["UI", "Dark Mode", "Theme"])
        if "registration" in test_case["name"].lower() or "registration" in test_case["objective"].lower():
            labels.extend(["Registration", "Validation"])
        if "email" in test_case["name"].lower() or "email" in test_case["objective"].lower():
            labels.append("Email")
        if "authentication" in test_case["name"].lower() or "auth" in test_case["objective"].lower():
            labels.append("Authentication")
        
        # Remove duplicates and set labels
        test_case["labels"] = list(set(labels)) if labels else []
        
        # Extract Test Data
        test_data_match = re.search(r'\*\*Test Data:\*\*\s*\n?(.+?)(?=\n\*\*|\n\n|$)', section, re.DOTALL)
        if not test_data_match:
            test_data_match = re.search(r'Test Data:\s*\n?(.+?)(?=\n\*\*|\nTest|\n\n|$)', section, re.DOTALL)
        
        if test_data_match:
            test_data = test_data_match.group(1).strip()
            # Remove any accidentally captured field labels
            test_data = re.sub(r'\*\*[^*]+:\*\*.*$', '', test_data, flags=re.DOTALL).strip()
            test_case["customFields"]["Test Data"] = test_data if test_data else ""
            print(f"✓ Extracted test data")
        
        # Extract Test Steps
        steps_match = re.search(r'\*\*Test Steps:\*\*\s*\n(.+?)(?=\n\*\*(?!.*\d+\.)|\n\n(?!\d))', section, re.DOTALL)
        if not steps_match:
            steps_match = re.search(r'Test Steps:\s*\n(.+?)(?=\n\*\*|\nTest|\n\n(?!\d))', section, re.DOTALL)
        
        if steps_match:
            steps = steps_match.group(1).strip()
            # Remove any accidentally captured field labels
            steps = re.sub(r'\n\*\*[^*]+:\*\*.*$', '', steps, flags=re.DOTALL).strip()
            test_case["customFields"]["Test Steps"] = steps if steps else ""
            print(f"✓ Extracted test steps")
        
        # Extract Expected Result
        expected_match = re.search(r'\*\*Expected Result:\*\*\s*\n?(.+?)(?=\n\*\*|\n\n|$)', section, re.DOTALL)
        if not expected_match:
            expected_match = re.search(r'Expected Result:\s*\n?(.+?)(?=\n\*\*|\nTest|\n\n|$)', section, re.DOTALL)
        
        if expected_match:
            expected = expected_match.group(1).strip()
            # Remove any accidentally captured field labels
            expected = re.sub(r'\*\*[^*]+:\*\*.*$', '', expected, flags=re.DOTALL).strip()
            test_case["customFields"]["Expected Result"] = expected if expected else ""
            print(f"✓ Extracted expected result")
        
        # Extract Actual Result
        actual_match = re.search(r'\*\*Actual Result:\*\*\s*\n?(.+?)(?=\n\*\*|\n\n|$)', section, re.DOTALL)
        if actual_match:
            actual_result = actual_match.group(1).strip()
            # Remove any accidentally captured field labels
            actual_result = re.sub(r'\*\*[^*]+:\*\*.*$', '', actual_result, flags=re.DOTALL).strip()
            test_case["customFields"]["Actual Result"] = actual_result if actual_result else ""
        else:
            test_case["customFields"]["Actual Result"] = ""
        
        # Extract Pass/Fail Criteria
        criteria_match = re.search(r'\*\*Pass/Fail Criteria:\*\*\s*\n?(.+?)(?=\n\*\*|\n\n|$)', section, re.DOTALL)
        if criteria_match:
            criteria = criteria_match.group(1).strip()
            # Remove any accidentally captured field labels
            criteria = re.sub(r'\*\*[^*]+:\*\*.*$', '', criteria, flags=re.DOTALL).strip()
            test_case["customFields"]["Pass/Fail Criteria"] = criteria if criteria else ""
        else:
            test_case["customFields"]["Pass/Fail Criteria"] = ""
        
        # Extract Postconditions
        postcond_match = re.search(r'\*\*Postconditions?:\*\*\s*\n?(.+?)(?=\n\*\*|\n\n|$)', section, re.DOTALL)
        if postcond_match:
            postcond = postcond_match.group(1).strip()
            # Remove any accidentally captured field labels
            postcond = re.sub(r'\*\*[^*]+:\*\*.*$', '', postcond, flags=re.DOTALL).strip()
            test_case["customFields"]["Postconditions"] = postcond if postcond else ""
        else:
            test_case["customFields"]["Postconditions"] = ""
        
        # Extract Tested By
        tester_match = re.search(r'\*\*Tested By:\*\*\s*\n?(.+?)(?=\n\*\*|\n\n|$)', section, re.DOTALL)
        if tester_match:
            tester = tester_match.group(1).strip()
            # Remove any accidentally captured field labels
            tester = re.sub(r'\*\*[^*]+:\*\*.*$', '', tester, flags=re.DOTALL).strip()
            test_case["customFields"]["Tested By"] = tester if tester else ""
        else:
            test_case["customFields"]["Tested By"] = ""
        
        # Extract Test Date
        date_match = re.search(r'\*\*Test Date:\*\*\s*\n?(.+?)(?=\n\*\*|\n\n|$)', section, re.DOTALL)
        if date_match:
            test_date = date_match.group(1).strip()
            # Remove any accidentally captured field labels
            test_date = re.sub(r'\*\*[^*]+:\*\*.*$', '', test_date, flags=re.DOTALL).strip()
            test_case["customFields"]["Test Date"] = test_date if test_date else ""
        else:
            test_case["customFields"]["Test Date"] = ""
        
        # Extract Notes
        notes_match = re.search(r'\*\*Notes?:\*\*\s*\n?(.+?)(?=\n\*\*|\n\n---|\Z)', section, re.DOTALL)
        if notes_match:
            notes = notes_match.group(1).strip()
            # Remove any accidentally captured field labels
            notes = re.sub(r'\*\*[^*]+:\*\*.*$', '', notes, flags=re.DOTALL).strip()
            test_case["customFields"]["Notes"] = notes if notes else ""
        else:
            test_case["customFields"]["Notes"] = ""
        
        # Only add test case if it has at least a name
        if test_case["name"]:
            test_cases.append(test_case)
            print(f"✓ Added test case #{tc_counter}: {test_case['name']}")
            tc_counter += 1
        else:
            print(f"✗ Skipping section - no valid test case name found")
    
    print(f"\n=== PARSER SUMMARY ===")
    print(f"Total test cases parsed: {len(test_cases)}")
    print(f"=== END PARSER DEBUG ===\n")
    
    return test_cases


@app.post("/get_chat_response")  # Multi-agent orchestrator: Routes to appropriate agent (LLM-based for Test Manager & Product Owner)
def fetch_agent_response(req: QueryRequest):
    global DEFAULT_TEST_CASES
    actions = []
    user_prompt = req.query
    current_role = req.role
    response = ""
    test_cases = None
    dataset = None

    # If updated test cases are provided in the request, normalize and persist them
    if req.testCases is not None:
        try:
            # Normalize each incoming test case into canonical shape
            DEFAULT_TEST_CASES = [normalize_test_case(tc) for tc in req.testCases]
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid testCases payload: {e}")

    #excluding pre-defined messages
    # if "Get Resolution for Incident" != user_prompt:
    #     user_prompt=fine_tune_user_prompt(user_prompt)
    response=trigger_orchestrator(user_prompt,current_role)

    if "Test Manager" in current_role:
        # Check if this is a Load Dataset request
        if "load dataset" in user_prompt.lower():
            try:
                data_file = Path(__file__).parent.parent / "data.json"
                if data_file.exists():
                    with open(data_file, 'r', encoding='utf-8') as f:
                        dataset = json.load(f)
                    response = f"Dataset loaded successfully with {len(dataset)} records."
                    actions = [
                        {"label": "Close", "query": "Close"}
                    ]
                else:
                    response = "Error: data.json file not found."
                    actions = []
            except Exception as e:
                response = f"Error loading dataset: {str(e)}"
                actions = []
        
        # Check if this is a new test case generation request (user story clicked or generate request)
        else:
            is_generate_request = "generate test case" in user_prompt.lower() or "create test case" in user_prompt.lower() or "user story" in user_prompt.lower() or "HB-" in user_prompt.upper()
            
            # If it's a generate request, parse the AI response to extract test cases
            if is_generate_request:
                # Parse the AI-generated test cases from the response
                print("=== Parsing AI-generated test cases from response ===")
                print(f"Response content (first 1000 chars): {response[:1000]}...")  # Print first 1000 chars for debugging
                print(f"Full response length: {len(response)} characters")
                
                # Try to parse test cases from the AI response
                test_cases = parse_test_cases_from_ai_response(response)
                
                # If parsing succeeded, use the parsed test cases
                if test_cases and len(test_cases) > 0:
                    print(f"✓ Successfully parsed {len(test_cases)} test cases from AI response")
                    for i, tc in enumerate(test_cases, 1):
                        print(f"  TC {i}: {tc.get('name', 'Unnamed')}")
                    DEFAULT_TEST_CASES = test_cases
                else:
                    print("✗ Failed to parse test cases from AI response")
                    print(f"  Checking if response contains test case markers...")
                    if "Test Case Title:" in response or "**Test Case" in response:
                        print(f"  Response DOES contain test case markers but parsing failed!")
                    else:
                        print(f"  Response does NOT contain expected test case format")
                    
                    # For user story requests, don't fall back to defaults - let it show empty
                    # Only use defaults for explicit "create test case" button
                    if "create test case" in user_prompt.lower():
                        test_cases = []  # Will load defaults below
                    else:
                        # User story clicked but parsing failed - show empty instead of defaults
                        test_cases = []
            else:
                # Return existing test cases (for update operations like Review TC)
                try:
                    test_cases = DEFAULT_TEST_CASES if DEFAULT_TEST_CASES else []
                    print(f"Returning existing {len(test_cases)} test cases from DEFAULT_TEST_CASES")
                except NameError:
                    test_cases = []
            
            # Always generate test cases from LLM - no hardcoded defaults
            print(f"\n[Test Case Generation] Generating test cases from AI...")
            print(f"[Test Case Generation] Parsing AI response...")
            test_cases = parse_test_cases_from_ai_response(response)
            
            if test_cases:
                print(f"[Test Case Generation] Successfully parsed {len(test_cases)} test cases")
                # Update DEFAULT_TEST_CASES with the newly generated ones
                DEFAULT_TEST_CASES = test_cases
            else:
                print(f"[Test Case Generation] Warning: No test cases found in AI response")
                # If parsing fails and no cached test cases, provide instruction
                if not DEFAULT_TEST_CASES:
                    print(f"[Test Case Generation] No cached test cases available. Requesting user input...")
                    response = "Please provide a user story or feature description to generate test cases."
                    test_cases = []
                else:
                    print(f"[Test Case Generation] Using previously generated test cases from cache")
                    test_cases = DEFAULT_TEST_CASES

        # Format test cases for display in the response text
        # The detailed test cases will also be sent in the testCases field as structured JSON
        if test_cases:
            response = "Test cases have been created:\n\n"
            for idx, tc in enumerate(test_cases, 1):
                response += f"**Test Case ID:** {tc.get('id', f'TC-{idx:03d}')}  \n"
                response += f"**Name:** {tc.get('name', 'Unnamed Test Case')}  \n"
                response += f"**Objective:** {tc.get('objective', 'N/A')}  \n"
                
                # Format preconditions on single line (replace newlines with semicolons)
                precond = tc.get('precondition') or tc.get('preconditions', 'N/A')
                precond_single_line = precond.replace('\n', '; ').replace('- ', '').strip()
                response += f"**Preconditions:** {precond_single_line}  \n"
                
                response += f"**Estimated Time:** {tc.get('estimatedTime', 0)}  \n"
                response += f"**Priority:** {tc.get('priorityName', 'Medium')}  \n"
                response += f"**Status:** {tc.get('statusName', 'Draft')}  \n"
                
                labels = tc.get('labels', [])
                response += f"**Labels:** {', '.join(labels) if labels else 'None'}  \n"
                
                # Custom fields - show all fields in order
                if tc.get('customFields'):
                    custom = tc['customFields']
                    # Format test data on single line (replace newlines with commas)
                    test_data = custom.get('Test Data', 'N/A')
                    test_data_single_line = test_data.replace('\n', ', ').strip()
                    response += f"\n**Test Data:** {test_data_single_line}  \n"
                    
                    # Test Steps - format each step on new line
                    steps_text = custom.get('Test Steps', 'N/A')
                    if steps_text and steps_text != 'N/A':
                        steps = [step.strip() for step in steps_text.split('\n') if step.strip()]
                        response += f"**Test Steps:**  \n"
                        for step in steps:
                            response += f"{step}  \n"
                    else:
                        response += f"**Test Steps:** N/A  \n"
                    
                    response += f"\n**Expected Result:** {custom.get('Expected Result', 'N/A')}  \n"
                    response += f"**Actual Result:** {custom.get('Actual Result', 'N/A')}  \n"
                    response += f"**Pass/Fail Criteria:** {custom.get('Pass/Fail Criteria', 'N/A')}  \n"
                    response += f"**Postconditions:** {custom.get('Postconditions', 'N/A')}  \n"
                    response += f"**Tested By:** {custom.get('Tested By', 'N/A')}  \n"
                    response += f"**Test Date:** {custom.get('Test Date', 'N/A')}  \n"
                    response += f"**Notes:** {custom.get('Notes', 'N/A')}  \n"
                
                response += "\n---\n\n"  # Separator between test cases
        else:
            response = "No test cases found."

        actions = [
            {"label": "Review TC", "query": "Review TC"},
            {"label": "Review TD", "query": "Review TD"},
            {"label": "Load Dataset", "query": "Load Dataset"},
            {"label": "Export", "query": "Export"}
        ]


    if "L1/L2" in current_role:
        if user_prompt !="Investigate Further" and "close ticket" not in user_prompt:
            resolution_type=check_resolution_found(response)
            print("---Resolution Type--",resolution_type)
            # format resolution as list to render in ui, only for resolution response
            if "Resolution Found" in resolution_type:
                response=format_resolution_to_render(response)

            actions=get_actions(response,resolution_type)

    result = {"answer": response, "actions": actions}
    if test_cases is not None:
        result["testCases"] = test_cases
    if dataset is not None:
        result["dataset"] = dataset

    return result


@app.post("/api/upload-to-zephyr")
def upload_to_zephyr(req: ZephyrUploadRequest):
    """Upload test cases to Zephyr Squad"""
    try:
        print(f"Uploading {len(req.testCases)} test cases to Zephyr...")
        
        # Upload test cases to Zephyr
        results = upload_test_cases_to_zephyr(req.testCases, req.projectKey)
        
        return {
            "success": len(results["failed"]) == 0,
            "message": f"Successfully uploaded {len(results['success'])} out of {results['total']} test cases",
            "results": results
        }
    except Exception as e:
        print(f"Error uploading to Zephyr: {e}")
        raise HTTPException(status_code=500, detail=f"Error uploading to Zephyr: {str(e)}")



@app.post("/api/upload-to-jira")  # Upload user stories (can be LLM-generated) to Jira
def upload_to_jira(req: JiraUploadRequest):
    """Upload user stories to Jira"""
    try:
        print(f"Uploading {len(req.userStories)} user stories to Jira project {req.projectKey}...")
        
        # Upload user stories to Jira
        result = upload_user_stories_to_jira(req.projectKey, req.userStories)
        
        return result
    except Exception as e:
        print(f"Error uploading to Jira: {e}")
        raise HTTPException(status_code=500, detail=f"Error uploading to Jira: {str(e)}")


if __name__=="__main__":
    uvicorn.run(app,host="127.0.0.1",port=8000)