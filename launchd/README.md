# launchd jobs

Copies of the two LaunchAgents that keep the server up. They live in
`~/Library/LaunchAgents/` at runtime; these are the versioned originals, so a
rebuilt machine can be brought back without reconstructing them from memory.

Both hardcode `/Users/scifablab/...` paths — launchd does no variable expansion
in `ProgramArguments`, so a different user or checkout location means editing
these files.

## Install

```sh
cp launchd/*.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$UID ~/Library/LaunchAgents/com.trellis.webserver.plist
launchctl bootstrap gui/$UID ~/Library/LaunchAgents/com.trellis.watchdog.plist
launchctl list | grep trellis          # both should appear
```

`bootout` to remove, `kickstart -k gui/$UID/com.trellis.webserver` to restart.

## com.trellis.webserver

Runs `run_server.sh` under `caffeinate -i -s` so the machine cannot idle- or
system-sleep while the server is up. `RunAtLoad` + `KeepAlive` restart it
whenever the process exits, throttled to one restart per 15s so a boot-time
failure cannot busy-loop the GPU.

## com.trellis.watchdog

Runs `watchdog.sh` every 60s. Covers the failure `KeepAlive` cannot see: a
process that is alive but wedged — the macOS GPU watchdog kills a long-running
Metal kernel, the decoder holds a dead command buffer, and uvicorn keeps its
socket open while answering nothing. To launchd that server looks healthy.

Restarts only after **three** consecutive failed health checks, because a
generation job holds the single worker thread for minutes and one slow response
is normal load, not a fault.

`RunAtLoad` is false: at login the server is still loading a ~90s pipeline, and
a watchdog counting strikes immediately would fight the boot it protects.

## Not handled by launchd

These need doing once, by hand, with admin rights:

```sh
sudo pmset -a autorestart 1     # power back on after a power cut
sudo pmset -a disksleep 0
```

Automatic login (System Settings → Users & Groups) is also required for an
unattended reboot to bring the server back, because a LaunchAgent only starts
once a GUI session exists. Trade-off: physical access to the Mac then yields a
logged-in desktop.
