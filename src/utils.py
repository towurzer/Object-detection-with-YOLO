import json
import os
import pickle
import shutil

import matplotlib.pyplot as plt
def create_dir(config):
	"""
	Creates the necessary project directories if they do not already exist.
	:arg config: Configuration object containing directory paths.
	"""
	os.makedirs(config.DATA_DIR, exist_ok=True)
	os.makedirs(config.OUT_DIR, exist_ok=True)
	os.makedirs(config.MODEL_DIR, exist_ok=True)
	os.makedirs(os.path.join(config.OUT_DIR, "evaluation"), exist_ok=True)

def clear_dirs(config):
    """
    Deletes all folders and subfolders
    :arg config: Configuration object containing directory paths.
    """
    for dir_path in [config.DATA_DIR, config.OUT_DIR, config.MODEL_DIR]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            print(f"Deleted directory and all contents: {dir_path}")
        else:
            print(f"Directory does not exist, skipping: {dir_path}")


def saveResults(directory, filename, results):
	save_path = os.path.join(directory, filename)

	with open(save_path, "w", encoding="utf-8") as f:
		f.write(results)
	print(f"Saved eval results items to {save_path}")

def gracefulExit():
	"""Wait for all pots to be closed before exiting"""
	while plt.get_fignums():
		plt.pause(0.1)

	exit(0)

def has_flag(argv, flags):
	return any(flag in argv for flag in flags)

def confirm(message, tries=3):
	confirm = input(message).strip().lower()
	for _ in range(tries):
		if confirm == "y":
			return True
		elif confirm == "n" or confirm == "":
			break
		else:
			confirm = input("Please enter 'y' to confirm or 'n' to cancel.")
	return False