"""Run the API or container tests with Docker Desktop open only when needed."""

import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STARTUP_TIMEOUT = 120


class TaskError(Exception):
    """A Docker task could not start or finish normally."""


class Interrupted(KeyboardInterrupt):
    def __init__(self, signum):
        self.signum = signum


def interrupt(signum, _frame):
    raise Interrupted(signum)


def find_docker():
    executable = shutil.which("docker")
    if executable:
        return executable
    bundled = Path("/Applications/Docker.app/Contents/Resources/bin/docker")
    if bundled.is_file() and os.access(bundled, os.X_OK):
        return str(bundled)
    raise TaskError("Docker was not found. Install Docker Desktop for macOS first.")


class DockerTask:
    def __init__(self, executable):
        self.docker = executable
        self.owns_desktop = False
        self.api_started = False
        self.test_container = None

    def capture(self, *arguments, timeout=5):
        return subprocess.run(
            [self.docker, *arguments],
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

    def compose(self, *arguments):
        return [
            "compose",
            "--project-directory",
            str(ROOT),
            "--file",
            str(ROOT / "compose.yaml"),
            *arguments,
        ]

    def engine_ready(self, timeout=5):
        try:
            return self.capture(
                "info", "--format", "{{.ServerVersion}}", timeout=timeout
            ).returncode == 0
        except subprocess.TimeoutExpired:
            return False

    def ensure_engine(self):
        if self.engine_ready():
            print("Using the running Docker engine; Desktop will stay open.", flush=True)
            return
        if sys.platform != "darwin":
            raise TaskError(
                "Start your Docker engine first; automatic startup requires macOS."
            )

        # The Desktop status command can hang while the app is closed. Check
        # its macOS backend instead; an uncertain result never grants ownership.
        probe = subprocess.run(
            [
                "/usr/bin/pgrep",
                "-f",
                r"/Docker\.app/Contents/MacOS/(com\.docker\.backend|Docker Desktop)",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if probe.returncode != 1:
            raise TaskError(
                "Docker Desktop may already be running, but its engine is unavailable. "
                "Resume it if paused, wait for it to be ready, and retry. "
                "The helper will not close a Desktop session it cannot identify."
            )

        print("Starting Docker Desktop (up to 120 seconds)...", flush=True)
        deadline = time.monotonic() + STARTUP_TIMEOUT
        started = self.capture(
            "desktop", "start", "--detach", "--timeout", "15", timeout=20
        )
        if started.returncode:
            detail = (started.stderr or started.stdout).strip()
            raise TaskError(f"Docker Desktop could not start: {detail}")
        self.owns_desktop = True
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            if self.engine_ready(timeout=min(5, remaining)):
                return
            time.sleep(min(1, max(0, deadline - time.monotonic())))
        raise TaskError(
            "Docker engine was not ready within 120 seconds. Check Docker Desktop."
        )

    def stream(self, arguments):
        process = subprocess.Popen(
            [self.docker, *arguments],
            cwd=ROOT,
            env={**os.environ, "COMPOSE_MENU": "false"},
            start_new_session=True,
        )
        try:
            return process.wait()
        except KeyboardInterrupt as error:
            # Forward the terminal signal to Compose and let it stop gracefully.
            if process.poll() is None:
                os.killpg(process.pid, getattr(error, "signum", signal.SIGINT))
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
            raise

    def run(self, task):
        self.ensure_engine()
        if task == "up":
            self.api_started = True
            print("API: http://localhost:8000/docs — press Ctrl+C to stop.", flush=True)
            return self.stream(self.compose("up", "--build", "api"))
        self.test_container = f"teleradiology-tests-{os.getpid()}"
        return self.stream(
            self.compose(
                "run", "--build", "--rm", "--name", self.test_container, "tests"
            )
        )

    def cleanup(self):
        # A cleanup failure must not replace the service/test result.
        if self.api_started:
            try:
                stopped = self.capture(
                    *self.compose("stop", "--timeout", "10", "api"), timeout=25
                )
                if stopped.returncode:
                    print("Cleanup: could not stop the API service.", file=sys.stderr)
            except (OSError, subprocess.TimeoutExpired) as error:
                print(
                    f"Cleanup: could not stop the API service: {error}", file=sys.stderr
                )
        if self.test_container:
            try:
                # Only this helper's uniquely named test container can be removed.
                found = self.capture("container", "inspect", self.test_container)
                if found.returncode == 0:
                    removed = self.capture(
                        "rm", "--force", self.test_container, timeout=15
                    )
                    if removed.returncode:
                        print("Cleanup: could not remove the test container.", file=sys.stderr)
            except (OSError, subprocess.TimeoutExpired) as error:
                print(
                    f"Cleanup: could not remove the test container: {error}",
                    file=sys.stderr,
                )
        if not self.owns_desktop:
            return
        try:
            running = self.capture("ps", "--quiet")
            if running.returncode:
                print(
                    "Cannot verify running containers; leaving Docker Desktop open.",
                    file=sys.stderr,
                )
            elif running.stdout.strip():
                print(
                    "Other containers are running; leaving Docker Desktop open.",
                    flush=True,
                )
            else:
                print("Stopping Docker Desktop...", flush=True)
                stopped = self.capture("desktop", "stop", "--timeout", "30", timeout=35)
                if stopped.returncode:
                    print(
                        "Docker Desktop did not stop; use 'docker desktop stop'.",
                        file=sys.stderr,
                    )
        except (OSError, subprocess.TimeoutExpired) as error:
            print(
                f"Leaving Docker Desktop open after cleanup error: {error}",
                file=sys.stderr,
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("up", "test"))
    arguments = parser.parse_args()
    task = None
    handlers = {
        sig: signal.signal(sig, interrupt) for sig in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        task = DockerTask(find_docker())
        result = task.run(arguments.task)
        return result if result >= 0 else 128 - result
    except KeyboardInterrupt as error:
        print("\nStopping the Docker task...", flush=True)
        return 128 + getattr(error, "signum", signal.SIGINT)
    except (TaskError, OSError, subprocess.TimeoutExpired) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    finally:
        for sig in handlers:
            signal.signal(sig, signal.SIG_IGN)
        try:
            if task is not None:
                task.cleanup()
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)


if __name__ == "__main__":
    sys.exit(main())
