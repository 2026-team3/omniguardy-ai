# Keypad trigger test for `vision_api.py`

Start the existing Vision API from the project root:

```powershell
.\.venv\Scripts\python.exe api\vision_api.py
```

It listens on port 8000. The desktop does not need a camera, a keypad, or a
Raspberry Pi for this test. From a LAN machine, replace `127.0.0.1` with the
desktop IPv4 address obtained from `ipconfig`.

```powershell
curl.exe -X POST http://127.0.0.1:8000/triggers/keypad `
  -H "Content-Type: application/json" `
  -d '{"triggerId":"keypad-test-001","triggeredAt":"2026-09-30T10:00:00+09:00","securityEventId":"event-001","deviceId":"pi-01"}'
```

Successful requests return HTTP `202` and `status: accepted`. Confirm the
payload that arrived at the desktop:

```powershell
curl.exe http://127.0.0.1:8000/triggers/keypad/latest
```

Only the latest 100 trigger metadata records are retained in memory and they
disappear when the service restarts. Never send a keypad PIN in the request.

When the Raspberry Pi and its USB webcam are available, the backend can record
the video after this trigger and upload it to the existing `POST /analyze/vision`
endpoint. Video-inference models now load only when that endpoint is first used,
so keypad testing remains independent of camera/model initialization.
