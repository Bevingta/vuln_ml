import logging

def create_logger(run_history_save_path, timestamp):
    logger = logging.getLogger("run_logger")
    logger.setLevel(logging.INFO)

    # Define log file path
    log_file_path = f"{run_history_save_path}/run_{timestamp}_log.log"
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)

    # Create a log format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    # Log a message to indicate the start of a run
    logger.info("Starting run at %s", timestamp)

    return logger
