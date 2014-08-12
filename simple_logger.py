import logging

#This can be set in main
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", filename="example.log", level=logging.DEBUG,
                    filemode="w", datefmt='%m/%d/%Y %I:%M:%S %p')

#This code appears in each of the submodules, under the class
logger = logging.getLogger(__name__)

logger.warning("Test")
logger.warning("Watch out")
logger.info("Puppies")
logger.debug("debug message")
