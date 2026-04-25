import datetime
import os.path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# If modifying these SCOPES, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/calendar.readonly']

def get_calendar_events():
    """Fetches the next 10 events from the user's Google Calendar."""
    creds = None
    # The file token.json stores the user's access and refresh tokens.
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    service = build('calendar', 'v3', credentials=creds)

    # Call the Calendar API
    now = datetime.datetime.utcnow().isoformat() + 'Z'  # 'Z' indicates UTC time
    print('📅 Fetching upcoming events...')
    events_result = service.events().list(calendarId='primary', timeMin=now,
                                              maxResults=10, singleEvents=True,
                                              orderBy='startTime').execute()
    events = events_result.get('items', [])

    if not events:
        return "No upcoming events found."

    event_summary = ""
    for event in events:
        start = event['start'].get('dateTime', event['start'].get('date'))
        event_summary += f"- {event['summary']} at {start}\n"
    
    return event_summary

# --- AGENT INTEGRATION ---
def resolve_calendar_conflicts(model_pipe):
    # 1. Get real data from Google
    real_schedule = get_calendar_events()
    print(f"\n📋 Current Schedule from Google:\n{real_schedule}")
    
    # 2. Ask Model to Assign Priorities
    prompt = f"""<|im_start|>system
    You are an Elite Executive Assistant. Here is the user's real schedule from Google Calendar.
    Identify any time overlaps and assign Priority Levels (Critical, High, Medium, Low).
    <|im_end|>
    <|im_start|>user
    {real_schedule}
    <|im_end|>
    <|im_start|>assistant
    <thought>
    """
    
    # Run the model
    print("\n🧠 Agent is analyzing your Google Calendar...")
    outputs = model_pipe(prompt, max_new_tokens=512, do_sample=False)
    print(outputs[0]["generated_text"])

# Note: You need credentials.json in your folder for this to work!
