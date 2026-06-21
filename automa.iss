[Setup]
AppId={{196E198B-B90A-49F2-B131-E20A8785372C}
AppName=automa
AppVersion=1.0
AppPublisher=geltz
DefaultDirName={autopf}\automa
SetupIconFile=automa.ico
DefaultGroupName=geltz
Compression=lzma2/ultra64
SolidCompression=yes
OutputDir=.
OutputBaseFilename=automa_setup_1.0
WizardStyle=modern
UninstallDisplayIcon={app}\automa.exe

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; Copy the main executable
Source: "dist\automa\automa.exe"; DestDir: "{app}"; Flags: ignoreversion

; Copy the main application icon next to the exe in the install folder
Source: "automa.ico"; DestDir: "{app}"; Flags: ignoreversion

; Copy the internal dependency libraries
Source: "dist\automa\_internal\*"; DestDir: "{app}\_internal"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
; Set the WorkingDir so the application can resolve relative paths and load DLLs correctly
Name: "{group}\automa"; Filename: "{app}\automa.exe"; WorkingDir: "{app}"
Name: "{autodesktop}\automa"; Filename: "{app}\automa.exe"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\automa.exe"; Description: "{cm:LaunchProgram,automa}"; Flags: nowait postinstall skipifsilent