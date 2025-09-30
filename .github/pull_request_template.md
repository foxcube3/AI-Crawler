## Summary

Fix PyInstaller build failure by:
- Adding `pyinstaller` to `requirements.txt`
- Correcting module invocation in `scripts/build_windows.bat` to use `python -m PyInstaller` (proper casing)
- Updating README troubleshooting guidance for the specific error

## Changes

- requirements.txt: add `pyinstaller>=6.16.0`
- scripts/build_windows.bat: use `-m PyInstaller` instead of `-m pyinstaller`
- README.md: expand troubleshooting and step-by-step fix

## Motivation

Windows build was failing with:
```
python.exe: No module named pyinstaller
PyInstaller build failed.
```
This occurs when PyInstaller is not installed in the active environment or when invoking the wrong module name. The PR ensures reliable builds via the provided script.

## Testing

- Run: `scripts\build_windows.bat`
- Verify output:
  - Executable created at `dist\AI_Crawler_Assistant_Server.exe`
  - Launch executable and confirm server starts on PORT (default 8000)
  - Open `http://localhost:8000/ui`

## Risk and Impact

- Low risk. Added dependency is standard for packaging and only used for Windows build flow.
- No runtime changes to the application logic.

## Checklist

- [ ] Ran `scripts\build_windows.bat` successfully
- [ ] Verified executable starts and serves UI
- [ ] README troubleshooting instructions render correctly
- [ ] No unintended changes to existing dependencies order/versions

## Related

- Troubleshooting section in README
- Windows build instructions