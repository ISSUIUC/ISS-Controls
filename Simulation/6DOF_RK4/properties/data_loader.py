import yaml
import numpy
import os
from properties.yaml_datatypes import get_loader
import properties.properties as prop

config = None

# Exception to handle errors raised by the config validator.
class ConfigurationException(Exception):
    pass

def rec_validate_config_section(traceback, cfg_section):
    """Recursively validates a config output by the yaml parser by checking every field in the dictionary
    for _None or _None:Required data fields, warning and erroring accordingly."""
    if type(cfg_section) is dict:
      # propagate recursion
      for field in cfg_section.keys():
        rec_validate_config_section(traceback+"."+field, cfg_section[field])
    elif (type(cfg_section) is list):
       for ind, listitem in enumerate(cfg_section):
          rec_validate_config_section(traceback+f"[{ind}]", listitem)
    else:
        if type(cfg_section) is str and cfg_section == "_None":
            print("PySim Configuration WARN: " + traceback + " is inherited from a structure but has an unset value.")
        if type(cfg_section) is str and cfg_section == "_None:Required":
            raise ConfigurationException("PySim Configuration ERR: " + traceback + " is inherited from a structure but has an unset:required value.")
        if type(cfg_section) is str and cfg_section == "_Outdated_config":
            raise ConfigurationException("PySim Configuration ERR: The configuration file loaded was marked with the OUTDATED_CONFIG flag, and the simulation will crash if it continues with this config. Please select a different configuration file.")
def validate_config(parsed_yaml):
   # Top level for config validation
   sections = parsed_yaml.keys()
   for section in sections:
        if(section == "define"):
            continue
        rec_validate_config_section(section, parsed_yaml[section])

def load_config(config_path):
  """Loads and validates a config given by the config path (relative to main.py)
  Returns config data and constants as a python dictionary.
  """

  if(config != None):
    print("PySim Configuration WARN: data_loader.load_config() has been called more than once in the same process.\nTo ensure data singularity, instead use dataloader.config")
 
  raw_yaml = ""

  # Load all data types and YAML structures
  with open(os.path.join(os.path.dirname(__file__), "../properties/typedef.yaml")) as typedef: 
    # Add yaml type definitions to current loaded file
    raw_yaml += typedef.read()

  # Load all ./templates
  template_directory = os.path.join(os.path.dirname(__file__), "../properties/templates")
  for template_filename in os.listdir(template_directory):
    template = os.path.join(template_directory, template_filename)
    # Check if template is a yaml file or not (All ./templates must be .yaml files.)
    if os.path.isfile(template) and template.endswith(".yaml"):
      with open(template) as template_raw: 
        # Append templates to the yaml header.
        raw_yaml += "\n" + template_raw.read()
    else:
      print("PySim Configuration WARN: Non-yaml File " + template + " is present in the /templates directory. \nEnsure that only YAML templates are present in /templates.")

  with open(os.path.join(os.path.dirname(__file__), config_path)) as cfg_raw: 
    # Append the config file to the typedef header
    raw_yaml += "\n" + cfg_raw.read()
 
  yaml_content = list(yaml.load_all(raw_yaml, Loader=get_loader()))[0] # PyYaml parser
  validate_config(yaml_content)
  del yaml_content['define'] # Remove header data from file, leaves only config data.
  print("PySim Configuration loaded with no fatal errors.")
  return yaml_content

config = load_config(prop.sim_config)