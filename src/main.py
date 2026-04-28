import sys
from config import Config, CLIConfig
import utils
import dataset

def main(pre_training_inference, perform_training, post_training_inference, evaluate_results):
	print("Starting ...")
	config = Config()  # Load Configuration
	utils.create_dir(config)  # create directories to store dataset as well as the scatter plots and clustering
	print("Loaded config, created directories")

	if pre_training_inference or perform_training or post_training_inference:
		# TODO: get Dataloader
		dataloader = dataset.get_dataloader(config)  # download the dataset, crate and return the dataloader

	if pre_training_inference:
		# TODO: Run Inference
		pass

	if perform_training:
		# TODO: Train model
		pass

	if post_training_inference:
		# TODO: Run Inference again on cached model
		# TODO: If no trained model found, throw an error with a describing message
		pass

	if evaluate_results:
		# TODO: Evaluate
		# TODO: If no pre and post training inference results found, throw an error with a describing message
		pass


if __name__ == '__main__':
	cli_config = CLIConfig()
	argv = sys.argv
	anyFlag = False

	do_pre_training_inference = utils.has_flag(argv, cli_config.pre_training_inference_flags)
	do_training = utils.has_flag(argv, cli_config.training_flags)
	do_post_training_inference = utils.has_flag(argv, cli_config.post_training_inference_flags)
	do_results_evaluation = utils.has_flag(argv, cli_config.evaluate_flags)
	do_pre_training_inference = do_pre_training_inference or utils.has_flag(argv, cli_config.inference_flags)
	do_post_training_inference = do_pre_training_inference or utils.has_flag(argv, cli_config.inference_flags)
	reset_cache = utils.has_flag(argv, cli_config.reset_flags)
	show_info_message = utils.has_flag(argv, cli_config.help_flags)

	anyFlag = do_pre_training_inference or do_training or do_post_training_inference or do_results_evaluation or reset_cache

	if show_info_message or not anyFlag:
		print(cli_config.help_string)
	elif reset_cache:
		if utils.confirm("Are you sure you want to delete all cached data? \nThis will also delete the dataset and cannot be undone! (y/N): "):
			utils.clear_dirs(Config())
			print("All caches have been reset.")
		else:
			print("Reset canceled.")
	else:
		main(do_pre_training_inference, do_training, do_post_training_inference, do_results_evaluation)

	utils.gracefulExit()
