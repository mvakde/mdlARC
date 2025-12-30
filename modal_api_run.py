import modal
import os
import subprocess
import sys

# 1. Define the App and Volume
app = modal.App("mdlARC-training")
volume = modal.Volume.from_name("mithil-arc", create_if_missing=True)

# 2. Define Paths (Moved up so they can be used in the Image definition)
# Your script logic explicitly checks for a folder named 'mdlARC', so we must preserve that name.
REMOTE_ROOT = "/root/mdlARC"
REMOTE_ARCHIVE_PATH = f"{REMOTE_ROOT}/archives"

# 3. Define the Image
# We use the official python image, install requirements, and add the local directory directly here.
image = (
    modal.Image.debian_slim()
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir(".", remote_path=REMOTE_ROOT)
)


# 4. The Training Function
@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=86400,  # 24 hours
    cpu=4.0,  # 8 physical cores (approx 16 vCPUs)
    memory=32768,  # 32 GiB RAM
    # A. Mounts are now handled by image.add_local_dir() above.
    # B. Mount the Volume specifically to where full-run.py tries to save archives.
    #    This "overlays" the volume on top of the 'archives' folder in the code mount.
    volumes={REMOTE_ARCHIVE_PATH: volume},
)
def train_entrypoint():
    print(f"🚀 Starting run in {REMOTE_ROOT}...")

    # Switch to the project directory so imports work relative to CWD
    os.chdir(REMOTE_ROOT)

    # Ensure src is in python path (just like _prepare_environment in full-run.py)
    sys.path.append(f"{REMOTE_ROOT}/src")

    # Run the existing full-run.py script
    try:
        # We run as a subprocess to ensure a fresh process state, similar to running from terminal
        subprocess.run([sys.executable, "full-run.py"], check=True)
        print("\n✅ Run completed successfully.")

        # Verify file creation
        if os.path.exists(REMOTE_ARCHIVE_PATH):
            files = os.listdir(REMOTE_ARCHIVE_PATH)
            print(f"Files saved to Volume ({REMOTE_ARCHIVE_PATH}): {files}")

    except subprocess.CalledProcessError as e:
        print(f"\nRun failed with exit code {e.returncode}")
        raise e


# 5. Helper to check your volume contents without running training
@app.function(volumes={"/data": volume})
def list_archives():
    print("Archives in Volume:")
    try:
        files = os.listdir("/data")
        for f in files:
            print(f" - {f}")
    except FileNotFoundError:
        print("Volume is empty or path not found.")
