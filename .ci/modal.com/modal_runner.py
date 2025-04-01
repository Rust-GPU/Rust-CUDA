import modal
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

APP_PREFIX = "rust-cuda-test-runner"
RUNNER_LOG_PREFIX = "[runner]"
REMOTE_LOG_PREFIX = "[remote]"

TestResult = Dict[str, Any]


@dataclass(frozen=True)  # Immutable config object
class Config:
    """Holds validated configuration for the runner."""

    local_test_binary_path: Path
    test_binary_args: List[str]
    modal_gpu_type: str
    os_name: str  # Original GHA OS name
    cuda_version: str
    modal_token_id: str
    modal_token_secret: str
    nvidia_os_tag: str
    gpu_config: str


def log_runner(message: str) -> None:
    """Logs a message to stderr with the Runner prefix."""
    print(f"{RUNNER_LOG_PREFIX} {message}", file=sys.stderr)


def log_remote(message: str) -> None:
    """Logs a message to stderr with the Remote prefix (used inside Modal function)."""
    print(f"{REMOTE_LOG_PREFIX} {message}", file=sys.stderr)


# --- Configuration Loading and Validation ---

# Mapping from GH Actions OS name to Nvidia registry tags
OS_NAME_TO_NVIDIA_TAG: Dict[str, str] = {
    "ubuntu-20.04": "ubuntu20.04",
    "ubuntu-24.04": "ubuntu24.04",
}


def load_and_validate_config(argv: List[str], env: Dict[str, str]) -> Config:
    """Loads config from args/env, validates, and returns a Config object or raises ValueError."""
    log_runner("Loading and validating configuration...")
    if len(argv) < 2:
        raise ValueError(
            "Usage: python modal_test_runner.py <path_to_test_binary> [test_args...]"
        )

    local_test_binary_path = Path(argv[1]).resolve()
    test_binary_args = argv[2:]

    # Read from env dictionary
    modal_gpu_type = env.get("MODAL_GPU_TYPE")
    os_name = env.get("RUNNER_OS_NAME")
    cuda_version = env.get("RUNNER_CUDA_VERSION")
    modal_token_id = env.get("MODAL_TOKEN_ID")
    modal_token_secret = env.get("MODAL_TOKEN_SECRET")

    required_vars: Dict[str, Optional[str]] = {
        "MODAL_GPU_TYPE": modal_gpu_type,
        "RUNNER_OS_NAME": os_name,
        "RUNNER_CUDA_VERSION": cuda_version,
        "MODAL_TOKEN_ID": modal_token_id,
        "MODAL_TOKEN_SECRET": modal_token_secret,
    }
    missing = [k for k, v in required_vars.items() if not v]
    if missing:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing)}"
        )

    assert modal_gpu_type is not None
    assert os_name is not None
    assert cuda_version is not None
    assert modal_token_id is not None
    assert modal_token_secret is not None

    # Validate OS and get Nvidia Tag
    if os_name not in OS_NAME_TO_NVIDIA_TAG:
        supported = list(OS_NAME_TO_NVIDIA_TAG.keys())
        raise ValueError(
            f"OS '{os_name}' not supported Linux target. Supported: {supported}"
        )
    nvidia_os_tag = OS_NAME_TO_NVIDIA_TAG[os_name]

    log_runner("Configuration loaded successfully.")
    return Config(
        local_test_binary_path=local_test_binary_path,
        test_binary_args=test_binary_args,
        modal_gpu_type=modal_gpu_type,
        os_name=os_name,
        cuda_version=cuda_version,
        modal_token_id=modal_token_id,
        modal_token_secret=modal_token_secret,
        nvidia_os_tag=nvidia_os_tag,
        gpu_config=modal_gpu_type,
    )


# --- Image Creation (Refined based on Modal guidance) ---
def create_runtime_image(
    os_tag: str, cuda_version: str, binary_path_to_add: Path
) -> modal.Image:
    """Creates the Modal runtime image, adding the specific test binary at runtime
    as the LAST step for better caching."""
    registry_image_tag = f"nvidia/cuda:{cuda_version}-runtime-{os_tag}"
    log_runner(f"Defining image from registry: {registry_image_tag}")
    remote_bin_path = "/app/test_executable"  # Define remote path

    try:
        # Define the base image and all steps *before* adding the local file
        image_base = (
            modal.Image.from_registry(registry_image_tag)
            .env({"RUST_BACKTRACE": "1"})
            .run_commands(
                # These commands modify the filesystem *before* the file is added
                "mkdir -p /app",
                "echo '[Remote] Container environment baseline ready.'",  # Use log_remote style
            )
            # Add any other apt_install, pip_install, run_commands etc. HERE if needed later
        )

        # *** Apply add_local_file as the VERY LAST step. ***
        image_with_file = image_base.add_local_file(
            local_path=binary_path_to_add,
            remote_path=remote_bin_path,
            copy=False,  # IMPORTANT: Adds file at runtime. This is the default but we want to be extra sure.
        )

        log_runner(
            f"Configured to add binary '{binary_path_to_add.name}' to '{remote_bin_path}' at runtime (last image step)."
        )
        return image_with_file

    except FileNotFoundError:
        log_runner(f"Error: Local binary path not found: {binary_path_to_add}")
        raise ValueError(f"Test binary not found at {binary_path_to_add}")
    except Exception as e:
        log_runner(
            f"Error defining Modal image from registry '{registry_image_tag}': {e}"
        )
        raise ValueError("Failed to define Modal image. Verify CUDA/OS tag.") from e


# --- Command Preparation ---
def prepare_remote_command(base_args: List[str]) -> List[str]:
    """Adds --nocapture to test args if appropriate."""
    remote_command = ["/app/test_executable"] + base_args
    is_list_command = any(arg == "--list" for arg in base_args)
    # Check if separator '--' exists
    try:
        separator_index = remote_command.index("--")
        test_runner_args = remote_command[separator_index + 1 :]
        has_nocapture = "--nocapture" in test_runner_args
    except ValueError:  # '--' not found
        separator_index = -1
        has_nocapture = False

    if not is_list_command:
        if separator_index == -1:
            # Add separator if not present
            remote_command.append("--")
        if not has_nocapture:
            # Add --nocapture after separator (or at end if separator was just added)
            remote_command.append("--nocapture")

    return remote_command


app = modal.App()  # Name will be set dynamically later


@app.function()
def execute_test_binary(command: List[str]) -> TestResult:
    """Executes the test binary within the Modal container."""
    start_time = time.monotonic()
    try:
        # Ensure the binary is executable after mounting
        subprocess.run(["chmod", "+x", command[0]], check=True, cwd="/app")
    except Exception as e:
        log_remote(f"Error setting executable permission on {command[0]}: {e}")
        # Still try to run, maybe permissions were already ok

    log_remote(f"Executing command: {' '.join(command)}")
    env: Dict[str, str] = os.environ.copy()  # Inherit container env
    process: Optional[subprocess.CompletedProcess] = None
    status: str = "failed"
    stdout: str = ""
    stderr: str = ""
    retcode: int = -99

    try:
        process = subprocess.run(
            command,
            cwd="/app",
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        status = "ok" if process.returncode == 0 else "failed"
        stdout, stderr, retcode = process.stdout, process.stderr, process.returncode
    except subprocess.TimeoutExpired as e:
        stderr = (
            e.stderr or ""
        ) + "\nTimeoutException: Test execution exceeded timeout."
        retcode = -1
        log_remote("Test execution TIMED OUT.")
    except Exception as e:
        stderr = f"[Remote] Runner Exception: Failed to execute binary: {e}\n"
        retcode = -2
        log_remote(stderr.strip())

    end_time = time.monotonic()
    duration_ms: int = int((end_time - start_time) * 1000)
    log_remote(
        f"Execution finished. Status: {status}, Return Code: {retcode}, Duration: {duration_ms}ms"
    )

    return {
        "status": status,
        "stdout": stdout,
        "stderr": stderr,
        "returncode": retcode,
        "duration_ms": duration_ms,
    }


# --- Main Execution Flow ---
def run_tests_on_modal(config: Config) -> int:
    """Orchestrates Modal setup, execution, and result processing. Returns exit code."""
    log_runner(f"Preparing Modal execution for {config.local_test_binary_path.name}")

    # Dynamically configure app name based on runtime config
    app.name = f"{APP_PREFIX}-{config.os_name.replace('.', '-')}-{config.cuda_version.replace('.', '-')}"
    log_runner(f"Using Modal App name: {app.name}")

    try:
        # Create image definition, including the specific binary to add at runtime
        runtime_image = create_runtime_image(
            config.nvidia_os_tag, config.cuda_version, config.local_test_binary_path
        )

        # Prepare remote command
        remote_command = prepare_remote_command(config.test_binary_args)
        log_runner(f"Prepared remote command: {' '.join(remote_command)}")

        # Configure the function call with dynamic resources
        # The image definition now includes the instruction to add the specific binary
        configured_function = execute_test_binary.options(
            image=runtime_image,
            gpu=config.gpu_config,
            timeout=600,
        )

        # Invoke Modal Function
        log_runner(f"Invoking Modal function '{app.name}/execute_test_binary'...")
        result: TestResult = configured_function.remote(remote_command)
        log_runner(
            f"Modal execution finished for {config.local_test_binary_path.name}."
        )

        # --- Process Results ---
        print(result.get("stdout", ""), end="")  # Print remote stdout to local stdout
        remote_stderr: str = result.get("stderr", "")
        if remote_stderr:
            log_runner("--- Remote Stderr ---")
            print(remote_stderr.strip(), file=sys.stderr)
            log_runner("--- End Remote Stderr ---")

        return result.get("returncode", 1)  # Return the exit code

    except modal.exception.AuthError as e:
        log_runner(f"Modal Authentication Failed: {e}")
        log_runner(
            "Ensure MODAL_TOKEN_ID and MODAL_TOKEN_SECRET env vars are set correctly."
        )
        return 1  # Return error code
    except ValueError as e:  # Catch config/setup errors (incl. image definition)
        log_runner(f"Setup Error: {e}")
        return 1
    except Exception as e:
        log_runner(f"An unexpected error occurred during Modal orchestration: {e}")
        import traceback

        traceback.print_exc(file=sys.stderr)
        return 1  # Return error code


# --- Entry Point ---
# This will be called by cargo with thet test binary path and args
if __name__ == "__main__":
    exit_code = 1  # Default exit code
    try:
        # Pass actual env vars and command line args from execution context
        config = load_and_validate_config(sys.argv, os.environ.copy())
        exit_code = run_tests_on_modal(config)
    except ValueError as e:
        log_runner(f"Configuration Error: {e}")
        sys.exit(1)
    except (
        Exception
    ) as e:  # Catch unexpected errors during initial setup before run_tests_on_modal
        log_runner(f"Unhandled script error: {e}")
        import traceback

        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    log_runner(f"Exiting with code {exit_code}")
    sys.exit(exit_code)
