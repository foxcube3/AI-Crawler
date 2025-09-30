; Inno Setup Script for AI Crawler Assistant
; Requires Inno Setup 6: https://jrsoftware.org/isdl.php

#define MyAppName "AI Crawler Assistant"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "Your Company"
#define MyAppExeName "AI_Crawler_Assistant_Server.exe"

[Setup]
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\AI Crawler Assistant
DefaultGroupName=AI Crawler Assistant
DisableDirPage=no
DisableProgramGroupPage=no
OutputDir=.
OutputBaseFilename=AI_Crawler_Assistant_Installer
Compression=lzma
SolidCompression=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Files]
; Include the built executable from PyInstaller
Source: "..\..\dist\{#MyAppExeName}"; DestDir: "{app}"; Flags: ignoreversion
; Bundle example templates so users can customize UI and reports
Source: "..\..\templates\*"; DestDir: "{app}\templates"; Flags: ignoreversion recursesubdirs createallsubdirs
; Bundle pre-configured data directory with sample files
Source: "..\..\data\*"; DestDir: "{app}\data"; Flags: ignoreversion recursesubdirs createallsubdirs

[Dirs]
; Ensure runtime subfolders exist
Name: "{app}\data\jobs"; Flags: uninsalwaysuninstall
Name: "{app}\data\reports"; Flags: uninsalwaysuninstall

[Icons]
; Create shortcuts
Name: "{group}\AI Crawler Assistant Server"; Filename: "{app}\{#MyAppExeName}"
Name: "{userdesktop}\AI Crawler Assistant Server"; Filename: "{app}\{#MyAppExeName}"

[Run]
; Offer to run server after installation
Filename: "{app}\{#MyAppExeName}"; Description: "Start AI Crawler Assistant Server"; Flags: postinstall nowait skipifsilent