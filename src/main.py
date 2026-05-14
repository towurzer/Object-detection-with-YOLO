import os.path
import sys
from config import Config, CLIConfig
import utils
import dataset
import inference
from model import train_model
import evaluation
import visualizations


def main(pre_training_inference, perform_training, post_training_inference, evaluate_results):
	print("Starting ...")
	config = Config()  # Load Configuration
	utils.create_dir(config)  # Create necessary directories if they don't exist
	print("Loaded config, created directories")

	if pre_training_inference or perform_training or post_training_inference:
		data_yaml = dataset.get_dataloader(config)  # download the dataset, create and return dataloader (path to data.yaml)

	if pre_training_inference:
		inference.run_pretrained_inference(config)

	if perform_training:
		train_model(config, data_yaml)

	if post_training_inference:
		inference.run_post_training_inference(config)

	if evaluate_results:
		results = evaluation.evaluate(config)
		print("\n\n\n")
		utils.saveResults(os.path.join(Config.OUT_DIR, "evaluation"), "evaluation_results.json", results)
		visualizations.plot_results_from_json(os.path.join(Config.OUT_DIR, "evaluation"), "evaluation_results.json")


if __name__ == '__main__':
	cli_config = CLIConfig()
	argv = sys.argv
	anyFlag = False

	do_pre_training_inference = utils.has_flag(argv, cli_config.pre_training_inference_flags)
	do_training = utils.has_flag(argv, cli_config.training_flags)
	do_post_training_inference = utils.has_flag(argv, cli_config.post_training_inference_flags)
	do_results_evaluation = utils.has_flag(argv, cli_config.evaluate_flags)
	do_pre_training_inference = do_pre_training_inference or utils.has_flag(argv, cli_config.inference_flags)
	do_post_training_inference = do_post_training_inference or utils.has_flag(argv, cli_config.inference_flags)
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
