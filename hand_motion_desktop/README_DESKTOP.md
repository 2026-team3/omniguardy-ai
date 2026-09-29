# Desktop Silent Signal API

The desktop runs the FastAPI service. It does not need a webcam, keypad, or
Raspberry Pi to receive and verify keypad triggers.

## Start the service

```powershell
cd hand_motion_desktop
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
cd src
python -m uvicorn api:app --host 0.0.0.0 --port 8001
```

Open `http://127.0.0.1:8001/docs` on the desktop to use Swagger UI. For a
notebook or backend machine on the same LAN, replace `127.0.0.1` with the
desktop's IPv4 address from `ipconfig`. Allow inbound TCP port 8001 on the
desktop's private-network Windows Firewall profile if needed.

## Test keypad trigger without hardware

From the desktop or another LAN machine:

```powershell
curl.exe -X POST http://127.0.0.1:8001/triggers/keypad `
  -H "Content-Type: application/json" `
  -d '{"triggerId":"keypad-test-001","triggeredAt":"2026-09-30T10:00:00+09:00","securityEventId":"event-001","deviceId":"pi-01"}'
```

Expected response is HTTP `202` with `status: accepted` and
`nextAction: UPLOAD_VIDEO`. Verify what the server received:

```powershell
curl.exe http://127.0.0.1:8001/triggers/keypad/latest
```

The service stores only the latest 100 trigger metadata records in memory; they
are intentionally cleared when the service restarts. Never send the keypad PIN.

## Later hardware flow

1. Raspberry Pi/backend validates the keypad input and sends JSON to
   `POST /triggers/keypad`.
2. Pi records video using its USB webcam.
3. Backend uploads the resulting MP4 to `POST /analyze/silent-signal` using
   `multipart/form-data` and the same `triggerId`.

Required upload fields: `file` (MP4), `triggerId`, and `triggeredAt`.
`triggerType` defaults to `KEYPAD`; `securityEventId` is optional.

```powershell
curl.exe -X POST http://127.0.0.1:8001/analyze/silent-signal `
  -F "file=@C:\path\test.mp4;type=video/mp4" `
  -F "triggerId=keypad-test-001" `
  -F "triggeredAt=2026-09-30T10:00:00+09:00"
```

If `visionEvents` includes `SILENT_SIGNAL`, the hand-motion rule detected the
signal. The current rule detects repeated fist-to-open-palm motion.
